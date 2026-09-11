"""Backtest walk-forward sin look-ahead (Fase 3).

Diseño aprobado en ``docs/fase0_data_gate.md`` (D8/D9, Opción B):

* Retornos log diarios sobre calendario hábil colombiano desde ``estimation_start`` = 2018-01-01.
* Ventana de estimación **rolling 24 meses** calendario: en la fecha de decisión ``t`` el optimizador
  ve exactamente ``returns[(t - 24m, t]]`` — nunca una fila posterior a ``t``. ``expanding`` (mín.
  24 m, recortado al soporte común de los fondos activos) queda como sensibilidad.
* Rebalanceo **trimestral** (base) o mensual (sensibilidad). Fecha de decisión = último día hábil
  del periodo; la primera es el último día hábil anterior a ``estimation_start + 24m`` (2019-12-30),
  y los pesos viven desde el siguiente día hábil (2020-01-02). Entre rebalanceos el portafolio es
  buy-and-hold (los pesos derivan con los precios).
* Universo point-in-time: un fondo entra cuando acumula ``window_months`` de historia observada a la
  fecha de decisión y sigue reportando (último dato dentro de ``max_staleness_bdays``); sale si deja
  de reportar. Nada de la composición futura del universo se usa en el pasado.
* NaN intra-vida (``manifest.json → gap_log``): retorno 0 ese día; el retorno puenteado del día
  siguiente ya recoge el movimiento acumulado (política del manifest).
* Turnover one-way ``½ Σ|w_target − w_drift|``; costo ``Σ|w_target − w_drift| × cost_bps / 1e4``
  descontado del NAV el primer día del nuevo periodo. ``cost_bps = 10`` es **[ESPECULATIVO]**: el gate
  documenta 0 comisiones explícitas de entrada/salida en FICs abiertos (la comisión de administración
  ya está en el VU); 10 bps es un proxy conservador de fricción operativa (cash drag de liquidación
  T+n, penalidades de pactos de permanencia). Parametrizable con ``--cost-bps``.

Salida: ``results/backtest_summary.parquet`` (una fila por metodología × perfil con métricas y la
serie de pesos anidada), ``results/backtest_weights.parquet`` y ``results/backtest_daily.parquet``
(formato largo), ``results/backtest_summary.md`` y la sección ``backtest`` de ``manifest.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping

import numpy as np
import pandas as pd

log = logging.getLogger("walk_forward")

OptimizerFn = Callable[[pd.DataFrame, str, str], pd.Series]
DEFAULT_METHODS: tuple[str, ...] = ("markowitz", "risk_parity", "hrp")
DEFAULT_PROFILES: tuple[str, ...] = ("conservador", "moderado", "agresivo")
COST_LABEL = "[ESPECULATIVO]"
COST_RATIONALE = (
    "El gate (docs/fase0_data_gate.md §4.3) documenta 0 comisiones explícitas de entrada/salida en FICs "
    "abiertos (la comisión de administración ya está descontada en el valor de unidad). 10 bps por unidad "
    "de notional transado es un proxy conservador de fricción operativa (cash drag de liquidación T+n, "
    "penalidades por pactos de permanencia) sin fuente confirmada; sensibilidad con --cost-bps."
)
NAN_POLICY = (
    "Celdas NaN intra-vida (manifest.json → gap_log) se tratan como retorno 0 tanto en la ventana de "
    "estimación como en el P&L; el retorno puenteado del primer día tras la brecha ya acumula el movimiento."
)
TURNOVER_DEFINITION = "one-way: 0.5 * sum(|w_target - w_drift|), w_drift = pesos previos derivados por precio"
COST_DEFINITION = "sum(|w_target - w_drift|) * cost_bps / 1e4, descontado del NAV el primer día del nuevo periodo"
WEIGHT_TOL = 1e-6
# docs/fase0_data_gate.md §3.2: vol anualizada esperada por perfil — validación ex-post (no target del
# optimizador). Regla del gate: fuera de la banda ±2 pp ⇒ se revisan los límites.
EXPECTED_VOL_BANDS: dict[str, tuple[float, float]] = {
    "conservador": (0.03, 0.05), "moderado": (0.07, 0.10), "agresivo": (0.11, 0.15),
}
VOL_BAND_SLACK = 0.02


class BacktestError(RuntimeError):
    """Invalid inputs, empty universe or an optimizer output that is not a portfolio."""


@dataclass(frozen=True)
class BacktestConfig:
    estimation_start: str = "2018-01-01"
    end: str | None = None                     # last date used (default: last row of the returns matrix)
    window_months: int = 24
    window_kind: Literal["rolling", "expanding"] = "rolling"
    rebalance: Literal["quarterly", "monthly"] = "quarterly"
    cost_bps: float = 10.0                     # [ESPECULATIVO] — see COST_RATIONALE
    rf_annual: float = 0.0                     # [ESPECULATIVO] Sharpe with rf = 0 (standard, comparable)
    periods_per_year: int = 252
    methods: tuple[str, ...] = DEFAULT_METHODS
    profiles: tuple[str, ...] = DEFAULT_PROFILES
    max_staleness_bdays: int = 5               # fund must have reported within the last k business days ≤ t
    min_active_funds: int = 2
    cov_method: str = "ledoit"                 # D8
    min_rv: tuple[float, ...] | None = None    # optional RV floor per profile (sensitivity; base D6 = None)
    tag: str = ""                              # suffix for sensitivity runs (files + manifest key)

    def __post_init__(self):
        if self.min_rv is not None and len(self.min_rv) != len(self.profiles):
            raise ValueError(f"min_rv needs one value per profile ({len(self.profiles)}), got {len(self.min_rv)}")
        if self.window_kind not in ("rolling", "expanding"):
            raise ValueError(f"window_kind must be 'rolling' or 'expanding', got {self.window_kind!r}")
        if self.rebalance not in ("quarterly", "monthly"):
            raise ValueError(f"rebalance must be 'quarterly' or 'monthly', got {self.rebalance!r}")
        if self.window_months < 1 or self.cost_bps < 0:
            raise ValueError("window_months must be >= 1 and cost_bps >= 0")

    @property
    def oos_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.estimation_start) + pd.DateOffset(months=self.window_months)

    @property
    def min_rv_by_profile(self) -> dict[str, float] | None:
        return dict(zip(self.profiles, self.min_rv)) if self.min_rv is not None else None


@dataclass
class CombinationResult:
    method: str
    profile: str
    weights: pd.DataFrame          # index = decision dates, columns = every fund (0 when inactive)
    active: pd.DataFrame           # same shape, bool
    n_active: pd.Series
    turnovers: pd.Series           # one-way, by decision date (first entry = deployment from cash)
    costs: pd.Series               # fraction of NAV, by decision date
    daily_returns: pd.Series       # simple returns, net of costs, OOS period
    nav: pd.Series
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Calendar / windows / universe
# ---------------------------------------------------------------------------
def rebalance_dates(index: pd.DatetimeIndex, cfg: BacktestConfig) -> list[pd.Timestamp]:
    """Decision dates: last business day before ``oos_start``, then the last business day of each period.

    Every date has at least one business day after it inside ``index`` (a holding period).
    """
    start = pd.Timestamp(cfg.estimation_start)
    end = pd.Timestamp(cfg.end) if cfg.end else index[-1]
    idx = index[(index >= start) & (index <= end)]
    before = idx[idx < cfg.oos_start]
    if len(before) == 0 or len(idx) < 2:
        raise BacktestError(f"no business days between {start.date()} and oos_start {cfg.oos_start.date()}")
    first = before[-1]
    freq = "Q" if cfg.rebalance == "quarterly" else "M"
    period_last = pd.Series(idx, index=idx).groupby(idx.to_period(freq)).last()
    dates = [first] + [d for d in period_last if d > first]
    return [d for d in dates if d < idx[-1]]


def active_universe(returns: pd.DataFrame, t: pd.Timestamp, cfg: BacktestConfig) -> list[str]:
    """Funds with >= ``window_months`` of observed history at ``t`` that are still reporting at ``t``."""
    past = returns.loc[returns.index <= t]
    min_first = t - pd.DateOffset(months=cfg.window_months)
    recent_cutoff = past.index[-cfg.max_staleness_bdays:][0]
    out = []
    for col in past.columns:
        obs = past.index[past[col].notna()]
        if len(obs) == 0 or obs[0] > min_first or obs[-1] < recent_cutoff:
            continue
        out.append(col)
    return out


def estimation_window(returns: pd.DataFrame, t: pd.Timestamp, cfg: BacktestConfig) -> pd.DataFrame:
    """Rows the optimizer may see at decision date ``t``: ``(t - window, t]`` (rolling) or
    ``[estimation_start, t]`` (expanding), always cut to the common support of the columns; intra-life
    NaN → 0 (see NAN_POLICY). Never contains a row after ``t``.
    """
    idx = returns.index
    mask = (idx <= t) & (idx >= pd.Timestamp(cfg.estimation_start))
    if cfg.window_kind == "rolling":
        mask &= idx > t - pd.DateOffset(months=cfg.window_months)
    win = returns.loc[mask]
    first_obs = win.apply(lambda s: s.first_valid_index())
    if first_obs.isna().any():
        raise BacktestError(f"columns without data in the window ending {t.date()}: {list(first_obs[first_obs.isna()].index)}")
    win = win.loc[win.index >= first_obs.max()]
    return win.fillna(0.0)


# ---------------------------------------------------------------------------
# Turnover / drift / simulation
# ---------------------------------------------------------------------------
def turnover(w_target: pd.Series, w_pre: pd.Series) -> float:
    """One-way turnover over the union of both index sets (missing = 0)."""
    cols = w_target.index.union(w_pre.index)
    delta = w_target.reindex(cols, fill_value=0.0) - w_pre.reindex(cols, fill_value=0.0)
    return 0.5 * float(delta.abs().sum())


def drift_weights(w: pd.Series, simple_returns: pd.DataFrame) -> pd.Series:
    """Weights after buy-and-hold through ``simple_returns`` (rows = days, NaN = no price change)."""
    growth = (1.0 + simple_returns[w.index].fillna(0.0)).prod(axis=0)
    values = w * growth
    return values / values.sum()


def simulate_period(w_target: pd.Series, simple_returns: pd.DataFrame, cost: float) -> tuple[pd.Series, pd.Series]:
    """Daily portfolio returns for one holding period (cost charged on its first day) and end weights."""
    growth = (1.0 + simple_returns[w_target.index].fillna(0.0)).cumprod(axis=0)
    values = growth.mul(w_target, axis=1) * (1.0 - cost)
    nav = values.sum(axis=1)
    nav_prev = nav.shift(1)
    nav_prev.iloc[0] = 1.0
    daily = nav / nav_prev - 1.0
    w_end = values.iloc[-1] / nav.iloc[-1]
    return daily, w_end


def _validate_weights(w: pd.Series, active: list[str], t: pd.Timestamp, method: str, profile: str) -> pd.Series:
    w = pd.Series(w, dtype=float).reindex(active)
    if w.isna().any():
        raise BacktestError(f"{method}/{profile} @ {t.date()}: optimizer returned NaN or missing weights")
    if abs(float(w.sum()) - 1.0) > WEIGHT_TOL or (w < -WEIGHT_TOL).any():
        raise BacktestError(f"{method}/{profile} @ {t.date()}: weights must be >= 0 and sum to 1 (sum={w.sum():.6f})")
    return w.clip(lower=0.0)


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def run_combination(returns: pd.DataFrame, optimizer: OptimizerFn, method: str, profile: str,
                    cfg: BacktestConfig, dates: list[pd.Timestamp]) -> CombinationResult:
    simple = np.expm1(returns)                               # log → simple; NaN stays NaN
    end = pd.Timestamp(cfg.end) if cfg.end else returns.index[-1]
    cols = list(returns.columns)
    weights, active_rows, n_active, turnovers, costs, daily_parts = {}, {}, {}, {}, {}, []
    w_pre = pd.Series(dtype=float)                           # all cash before the first decision
    for i, t in enumerate(dates):
        active = active_universe(returns, t, cfg)
        if len(active) < cfg.min_active_funds:
            raise BacktestError(f"only {len(active)} active funds at {t.date()} (< min_active_funds={cfg.min_active_funds})")
        window = estimation_window(returns[active], t, cfg)
        if window.index.max() > t:                           # invariant: no look-ahead, ever
            raise BacktestError(f"look-ahead: window ending {window.index.max().date()} > decision {t.date()}")
        w_target = _validate_weights(optimizer(window, method, profile), active, t, method, profile)
        tw = turnover(w_target, w_pre)
        cost = 2.0 * tw * cfg.cost_bps / 1e4                 # Σ|Δw| × bps
        t_next = dates[i + 1] if i + 1 < len(dates) else end
        period = simple.loc[(simple.index > t) & (simple.index <= t_next)]
        daily, w_pre = simulate_period(w_target, period, cost)
        daily_parts.append(daily)
        weights[t] = w_target.reindex(cols, fill_value=0.0)
        active_rows[t] = pd.Series([c in active for c in cols], index=cols)
        n_active[t], turnovers[t], costs[t] = len(active), tw, cost
        log.info("%s/%s %s: %d funds, turnover %.3f, window %s..%s", method, profile, t.date(), len(active), tw,
                 window.index.min().date(), window.index.max().date())
    daily_returns = pd.concat(daily_parts)
    daily_returns.name = "ret"
    res = CombinationResult(
        method=method, profile=profile,
        weights=pd.DataFrame(weights).T.rename_axis("date"),
        active=pd.DataFrame(active_rows).T.rename_axis("date"),
        n_active=pd.Series(n_active, name="n_active").rename_axis("date"),
        turnovers=pd.Series(turnovers, name="turnover").rename_axis("date"),
        costs=pd.Series(costs, name="cost").rename_axis("date"),
        daily_returns=daily_returns,
        nav=(1.0 + daily_returns).cumprod().rename("nav"),
    )
    res.metrics = compute_metrics(res.daily_returns, res.turnovers, res.costs, cfg)
    return res


def run_walk_forward(returns: pd.DataFrame, categories: Mapping[str, str], cfg: BacktestConfig,
                     optimizer: OptimizerFn | None = None) -> list[CombinationResult]:
    """Re-optimise every ``method × profile`` at every decision date; returns one result per combination."""
    if not isinstance(returns.index, pd.DatetimeIndex) or not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise BacktestError("returns must be indexed by a sorted, unique DatetimeIndex")
    optimizer = optimizer or default_optimizer(categories, rf=cfg.rf_annual, periods_per_year=cfg.periods_per_year,
                                               cov_method=cfg.cov_method, min_rv=cfg.min_rv_by_profile)
    dates = rebalance_dates(returns.index, cfg)
    log.info("%d rebalance dates: %s .. %s", len(dates), dates[0].date(), dates[-1].date())
    return [run_combination(returns, optimizer, m, p, cfg, dates) for m in cfg.methods for p in cfg.profiles]


def default_optimizer(categories: Mapping[str, str], *, rf: float = 0.0, periods_per_year: int = 252,
                      cov_method: str = "ledoit", min_rv: Mapping[str, float] | None = None) -> OptimizerFn:
    """Adapter over ``optimization.portfolio_optimizer.optimize`` (bands D6 as solver constraints).

    ``min_rv`` (profile -> floor) is a sensitivity knob: it adds an RV floor on top of the approved D6
    maxima; ``None`` keeps the approved bands untouched.
    """
    from dataclasses import replace

    from optimization.portfolio_optimizer import PROFILE_CONSTRAINTS, optimize

    def _opt(window: pd.DataFrame, method: str, profile: str) -> pd.Series:
        constraints = None
        if min_rv and profile in min_rv:
            constraints = replace(PROFILE_CONSTRAINTS[profile], min_rv=float(min_rv[profile]))
        return optimize(window, method, profile, constraints, categories=categories, rf=rf,
                        periods_per_year=periods_per_year, cov_method=cov_method).weights

    return _opt


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(daily: pd.Series, turnovers: pd.Series, costs: pd.Series, cfg: BacktestConfig) -> dict:
    """ann_return (geométrico), volatility, sharpe (exceso diario sobre rf), max_drawdown, cvar_95
    (media de los peores 5 % de días, negativo = pérdida), avg_turnover (excluye el despliegue inicial)."""
    ppy = cfg.periods_per_year
    n = len(daily)
    nav = (1.0 + daily).cumprod()
    std = float(daily.std(ddof=1))
    rf_daily = (1.0 + cfg.rf_annual) ** (1.0 / ppy) - 1.0
    q = daily.quantile(0.05)
    return {
        "ann_return": float(nav.iloc[-1] ** (ppy / n) - 1.0),
        "volatility": std * np.sqrt(ppy),
        "sharpe": float((daily - rf_daily).mean() / std * np.sqrt(ppy)) if std > 0 else float("nan"),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "cvar_95": float(daily[daily <= q].mean()),
        "avg_turnover": float(turnovers.iloc[1:].mean()) if len(turnovers) > 1 else float("nan"),
        "total_cost": float(costs.sum()),
        "n_rebalances": int(len(turnovers)),
        "n_days": int(n),
        "total_return": float(nav.iloc[-1] - 1.0),
    }


# ---------------------------------------------------------------------------
# Outputs + manifest
# ---------------------------------------------------------------------------
def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_entry(path: Path, **extra) -> dict:
    return {"sha256": sha256_of(path), "bytes": path.stat().st_size, **extra}


def _suffix(cfg: BacktestConfig) -> str:
    return f"_{cfg.tag}" if cfg.tag else ""


def _summary_frame(results: list[CombinationResult], cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    for r in results:
        w_long = r.weights.where(r.active).stack().reset_index()
        w_long.columns = ["date", "fund_id", "weight"]
        rows.append({
            "method": r.method, "profile": r.profile, **r.metrics,
            "window_months": cfg.window_months, "window_kind": cfg.window_kind, "rebalance": cfg.rebalance,
            "cost_bps": cfg.cost_bps, "rf_annual": cfg.rf_annual,
            "oos_start": str(r.daily_returns.index[0].date()), "oos_end": str(r.daily_returns.index[-1].date()),
            "weights": [{"date": str(d.date()), "fund_id": f, "weight": float(w)} for d, f, w in w_long.itertuples(index=False)],
        })
    return pd.DataFrame(rows)


def _weights_long(results: list[CombinationResult]) -> pd.DataFrame:
    parts = []
    for r in results:
        w = r.weights.stack().rename("weight").reset_index()
        w.columns = ["date", "fund_id", "weight"]
        a = r.active.stack().rename("active").reset_index()
        w["active"] = a["active"].to_numpy()
        w["turnover"] = w["date"].map(r.turnovers).to_numpy()
        w["cost"] = w["date"].map(r.costs).to_numpy()
        w["n_active"] = w["date"].map(r.n_active).to_numpy()
        w.insert(0, "profile", r.profile)
        w.insert(0, "method", r.method)
        parts.append(w)
    return pd.concat(parts, ignore_index=True)


def _daily_long(results: list[CombinationResult]) -> pd.DataFrame:
    parts = []
    for r in results:
        d = pd.DataFrame({"method": r.method, "profile": r.profile, "date": r.daily_returns.index,
                          "ret": r.daily_returns.to_numpy(), "nav": r.nav.to_numpy()})
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def render_markdown(results: list[CombinationResult], cfg: BacktestConfig, dates: list[pd.Timestamp],
                    n_active: pd.Series, out_dir: Path | None = None) -> str:
    r0 = results[0]
    lines = [
        f"# Backtest walk-forward{' — ' + cfg.tag if cfg.tag else ''} — {len(results)} combinaciones",
        "",
        f"Generado {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} por `backtest/walk_forward.py`.",
        "",
        "## Parámetros",
        "",
        f"- Ventana de estimación: **{cfg.window_kind} {cfg.window_months} meses** (retornos log diarios hábiles desde {cfg.estimation_start}), covarianza `{cfg.cov_method}` (D8).",
        f"- Rebalanceo: **{cfg.rebalance}** — {len(dates)} fechas de decisión, {dates[0].date()} → {dates[-1].date()}; OOS {r0.daily_returns.index[0].date()} → {r0.daily_returns.index[-1].date()} ({len(r0.daily_returns)} días hábiles) (D9).",
        f"- Universo point-in-time: {int(n_active.iloc[0])} fondos en la primera decisión, {int(n_active.iloc[-1])} en la última (entrada tras {cfg.window_months} m de historia; salida si deja de reportar).",
        f"- Costo de transacción: **{cfg.cost_bps:.1f} bps** por unidad de notional transado {COST_LABEL}. {COST_RATIONALE}",
        f"- Turnover: {TURNOVER_DEFINITION}. `Turnover prom.` excluye el despliegue inicial desde efectivo.",
        f"- Bandas: D6 (máximos RV / mínimos RF / máx. RV intl / máx. por fondo) como restricciones del solver"
        + (f"; **piso RV (min_rv) por perfil: {cfg.min_rv_by_profile}** [ESPECULATIVO — sensibilidad, no aprobado en D6]." if cfg.min_rv else "; sin piso de RV (min_rv = 0)."),
        f"- Sharpe: exceso diario sobre rf = {cfg.rf_annual:.2%} anual [ESPECULATIVO: rf = 0 es el estándar comparable; usa `--rf` para otra tasa], anualizado con √{cfg.periods_per_year}.",
        "- CVaR 95 %: media de los peores 5 % de retornos diarios (negativo = pérdida). Max DD sobre el NAV neto de costos.",
        f"- NaN: {NAN_POLICY}",
        "",
        "## Resumen por combinación",
        "",
        "| Metodología | Perfil | Ret. anual | Vol. anual | Sharpe | Max DD | CVaR 95 % (diario) | Turnover prom. | Costo total | N rebal. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.method} | {r.profile} | {m['ann_return']:.2%} | {m['volatility']:.2%} | {m['sharpe']:.2f} | "
            f"{m['max_drawdown']:.2%} | {m['cvar_95']:.2%} | {m['avg_turnover']:.2%} | {m['total_cost']:.2%} | {m['n_rebalances']} |"
        )
    lines += ["", "## Validación ex-post de volatilidad (gate §3.2)", "",
              f"Banda esperada por perfil ±{VOL_BAND_SLACK:.0%}; fuera de banda ⇒ el gate pide revisar los límites (D6).", "",
              "| Metodología | Perfil | Vol. realizada | Banda esperada | Veredicto |", "|---|---|---:|---:|---|"]
    for r in results:
        band = EXPECTED_VOL_BANDS.get(r.profile)
        if band is None:
            verdict, band_txt = "n/a", "n/a"
        else:
            lo, hi = band
            inside = lo - VOL_BAND_SLACK <= r.metrics["volatility"] <= hi + VOL_BAND_SLACK
            verdict, band_txt = ("dentro" if inside else "**FUERA**"), f"{lo:.2%}–{hi:.2%}"
        lines.append(f"| {r.method} | {r.profile} | {r.metrics['volatility']:.2%} | {band_txt} | {verdict} |")
    siblings = sorted(p.name for p in (Path(out_dir) / ".").glob(f"backtest_summary_*.md") if p.name != f"backtest_summary{_suffix(cfg)}.md") if out_dir else []
    if siblings:
        lines += ["", "## Corridas de sensibilidad disponibles", ""] + [f"- `results/{n}`" for n in siblings]
    lines += ["", "## Fondos activos por fecha de decisión", "", "| Fecha | N activos |", "|---|---:|"]
    lines += [f"| {d.date()} | {int(n)} |" for d, n in n_active.items()]
    lines.append("")
    return "\n".join(lines)


def write_outputs(results: list[CombinationResult], cfg: BacktestConfig, root: Path | str) -> dict:
    """Write ``results/backtest_*`` and add the ``backtest`` section (SHA256 + exact params) to ``manifest.json``."""
    root = Path(root)
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = _suffix(cfg)
    r0 = results[0]
    dates = list(r0.weights.index)

    summary_path = out_dir / f"backtest_summary{sfx}.parquet"
    weights_path = out_dir / f"backtest_weights{sfx}.parquet"
    daily_path = out_dir / f"backtest_daily{sfx}.parquet"
    md_path = out_dir / f"backtest_summary{sfx}.md"
    summary = _summary_frame(results, cfg)
    summary.to_parquet(summary_path, index=False)
    weights_long = _weights_long(results)
    weights_long.to_parquet(weights_path, index=False)
    daily_long = _daily_long(results)
    daily_long.to_parquet(daily_path, index=False)
    md_path.write_text(render_markdown(results, cfg, dates, r0.n_active, out_dir), encoding="utf-8")

    inputs = {}
    for rel in ("data/cleaned/returns_matrix.parquet", "data/universe.json"):
        if (root / rel).exists():
            inputs[rel] = _file_entry(root / rel)
    try:
        import cvxpy, pypfopt, riskfolio
        libs = {"riskfolio-lib": riskfolio.__version__, "PyPortfolioOpt": pypfopt.__version__, "cvxpy": cvxpy.__version__}
    except Exception:  # pragma: no cover - optional in test environments
        libs = {}
    section = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "module": "backtest/walk_forward.py",
        "version": "0.1.0",
        "libraries": {"pandas": pd.__version__, "numpy": np.__version__, **libs},
        "params": {
            "estimation_start": str(pd.Timestamp(cfg.estimation_start).date()),
            "oos_start": str(cfg.oos_start.date()),
            "first_rebalance": str(dates[0].date()),
            "last_rebalance": str(dates[-1].date()),
            "first_oos_day": str(r0.daily_returns.index[0].date()),
            "oos_end": str(r0.daily_returns.index[-1].date()),
            "end_requested": cfg.end,
            "window_months": cfg.window_months,
            "window_kind": cfg.window_kind,
            "rebalance": cfg.rebalance,
            "n_rebalances": len(dates),
            "n_oos_days": int(len(r0.daily_returns)),
            "cost_bps": float(cfg.cost_bps),
            "cost_label": COST_LABEL,
            "cost_rationale": COST_RATIONALE,
            "cost_definition": COST_DEFINITION,
            "turnover_definition": TURNOVER_DEFINITION,
            "rf_annual": float(cfg.rf_annual),
            "periods_per_year": cfg.periods_per_year,
            "cov_method": cfg.cov_method,
            "max_staleness_bdays": cfg.max_staleness_bdays,
            "min_active_funds": cfg.min_active_funds,
            "methods": list(cfg.methods),
            "profiles": list(cfg.profiles),
            "min_rv": cfg.min_rv_by_profile,
            "nan_policy": NAN_POLICY,
            "tag": cfg.tag,
        },
        "inputs": inputs,
        "rebalance_dates": [str(d.date()) for d in dates],
        "n_active_by_rebalance": {str(d.date()): int(n) for d, n in r0.n_active.items()},
        "outputs": {
            str(summary_path.relative_to(root)): _file_entry(summary_path, rows=len(summary)),
            str(weights_path.relative_to(root)): _file_entry(weights_path, rows=len(weights_long)),
            str(daily_path.relative_to(root)): _file_entry(daily_path, rows=len(daily_long)),
            str(md_path.relative_to(root)): _file_entry(md_path),
        },
        "metrics": {f"{r.method}|{r.profile}": r.metrics for r in results},
    }
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[f"backtest{sfx}"] = section
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote %s, %s, %s, %s and manifest section backtest%s", summary_path, weights_path, daily_path, md_path, sfx)
    return section


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Walk-forward backtest sin look-ahead (Fase 3)")
    ap.add_argument("--root", default=".", help="repo root (reads <root>/data, writes <root>/results and <root>/manifest.json)")
    ap.add_argument("--returns", default=None, help="parquet with daily log returns (default <root>/data/cleaned/returns_matrix.parquet)")
    ap.add_argument("--universe", default=None, help="universe.json with fund categories (default <root>/data/universe.json)")
    ap.add_argument("--estimation-start", default="2018-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--window-months", type=int, default=24)
    ap.add_argument("--window-kind", choices=("rolling", "expanding"), default="rolling")
    ap.add_argument("--rebalance", choices=("quarterly", "monthly"), default="quarterly")
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate for the Sharpe ratio")
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--profiles", nargs="+", default=list(DEFAULT_PROFILES))
    ap.add_argument("--min-active-funds", type=int, default=2)
    ap.add_argument("--max-staleness-bdays", type=int, default=5)
    ap.add_argument("--min-rv", nargs="+", type=float, default=None,
                    help="optional RV floor per profile, same order as --profiles (sensitivity; base case has none)")
    ap.add_argument("--tag", default="", help="suffix for sensitivity runs (results/backtest_*_<tag>, manifest key backtest_<tag>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    root = Path(args.root)
    cfg = BacktestConfig(estimation_start=args.estimation_start, end=args.end, window_months=args.window_months,
                         window_kind=args.window_kind, rebalance=args.rebalance, cost_bps=args.cost_bps,
                         rf_annual=args.rf, methods=tuple(args.methods), profiles=tuple(args.profiles),
                         min_active_funds=args.min_active_funds, max_staleness_bdays=args.max_staleness_bdays,
                         min_rv=tuple(args.min_rv) if args.min_rv else None, tag=args.tag)
    returns = pd.read_parquet(args.returns or root / "data" / "cleaned" / "returns_matrix.parquet")
    universe = json.loads(Path(args.universe or root / "data" / "universe.json").read_text(encoding="utf-8"))
    categories = {f["fund_id"]: f["cat"] for f in universe["funds"]}
    optimizer = default_optimizer(categories, rf=cfg.rf_annual, periods_per_year=cfg.periods_per_year,
                                  cov_method=cfg.cov_method, min_rv=cfg.min_rv_by_profile)
    try:
        results = run_walk_forward(returns, categories, cfg, optimizer=optimizer)
        write_outputs(results, cfg, root)
    except BacktestError as exc:
        log.error("backtest failed: %s", exc)
        return 2
    for r in results:
        m = r.metrics
        print(f"{r.method:12s} {r.profile:12s} ret {m['ann_return']:7.2%} vol {m['volatility']:6.2%} sharpe {m['sharpe']:5.2f} "
              f"mdd {m['max_drawdown']:7.2%} cvar95 {m['cvar_95']:7.2%} turnover {m['avg_turnover']:6.2%}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
