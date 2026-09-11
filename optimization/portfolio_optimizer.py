"""Capa matemática del Optimizador Multi-Perfil (Fase 2).

Interfaz única::

    optimize(returns, method, profile, constraints=None, *, categories) -> OptimizationResult

Tres metodologías:

* ``"markowitz"``   — portafolio tangente (máx. Sharpe) con ``EfficientFrontier`` de PyPortfolioOpt.
* ``"risk_parity"`` — paridad de riesgo (``Portfolio.rp_optimization``) de Riskfolio-Lib.
* ``"hrp"``         — Hierarchical Risk Parity (``HCPortfolio``) de Riskfolio-Lib, seguido de la
  proyección QP exacta ``min ||w - w_hrp||²`` sobre el poliedro de bandas (decisión D5 del gate).

Las bandas por perfil (D6, look-through de mixtos D7) entran a Markowitz y Risk Parity como
restricciones del solver. HRP no tiene solver, así que las bandas se imponen con una proyección
convexa exacta (cvxpy). No hay recorte ni renormalización heurística posterior: si el solver
devuelve pesos fuera de banda se lanza ``OptimizationError``.

Firmas verificadas contra la versión instalada (``requirements.txt``) — ver etiquetas [VERIFICADO]
en cada ``_solve_*``.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.covariance import LedoitWolf

Method = Literal["markowitz", "risk_parity", "hrp"]
Profile = Literal["conservador", "moderado", "agresivo"]
CovMethod = Literal["ledoit", "hist"]

METHODS: tuple[str, ...] = ("markowitz", "risk_parity", "hrp")
PROFILES: tuple[str, ...] = ("conservador", "moderado", "agresivo")
CATEGORIES: frozenset[str] = frozenset({"RF_CORTO", "RF_LARGO", "RV_LOCAL", "RV_INTL", "MIXTO"})
RV_CATEGORIES = frozenset({"RV_LOCAL", "RV_INTL"})
RF_CATEGORIES = frozenset({"RF_CORTO", "RF_LARGO"})

BAND_TOL = 1e-6          # tolerance when verifying solver output against the bands
QP_SOLVER = "CLARABEL"
MARKOWITZ_SOLVER_OPTIONS = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 500}


class OptimizationError(RuntimeError):
    """The solver failed or returned weights that violate the bands."""


class InfeasibleConstraintsError(OptimizationError):
    """The band polyhedron for this universe/profile is empty."""


@dataclass(frozen=True)
class ProfileConstraints:
    """Bandas por perfil (docs/fase0_data_gate.md §3.2, D6/D7). Fracciones, no porcentajes.

    ``max_rv``      máx. exposición RV combinada = RV_LOCAL + RV_INTL + mixto_rv_share × MIXTO
    ``min_rf``      mín. exposición RF = RF_CORTO + RF_LARGO + (1 - mixto_rv_share) × MIXTO
    ``max_rv_intl`` máx. RV internacional
    ``max_weight``  máx. por fondo; ``min_weight`` mín. por fondo (0 → HRP/QP pueden apagar fondos)
    ``min_rv``      piso opcional de RV combinada (0 = sin piso, base D6). Sensibilidad [ESPECULATIVO]:
                    los pisos del Decreto 2555 Art. 2.6.12.1.4 (Moderado ≥ 20 %, Mayor Riesgo ≥ 45 %).
    """
    max_rv: float
    min_rf: float
    max_rv_intl: float
    max_weight: float
    min_weight: float = 0.0
    mixto_rv_share: float = 0.5
    min_rv: float = 0.0


# D6 aprobada (Opción 1): Conservador 20/70, Moderado 45/40, Agresivo 70/20; intl 10/20/35; máx fondo 25/20/20.
PROFILE_CONSTRAINTS: dict[str, ProfileConstraints] = {
    "conservador": ProfileConstraints(max_rv=0.20, min_rf=0.70, max_rv_intl=0.10, max_weight=0.25),
    "moderado": ProfileConstraints(max_rv=0.45, min_rf=0.40, max_rv_intl=0.20, max_weight=0.20),
    "agresivo": ProfileConstraints(max_rv=0.70, min_rf=0.20, max_rv_intl=0.35, max_weight=0.20),
}


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series                 # fund_id -> weight, sums to 1, inside the bands
    expected_return: float             # annualised, w·mu
    volatility: float                  # annualised, sqrt(w'Σw)
    sharpe: float                      # (expected_return - rf) / volatility
    method: str
    profile: str
    exposures: dict[str, float]        # {"rv", "rf", "rv_intl"} after look-through
    raw_weights: pd.Series | None = None        # HRP: weights before the QP projection
    linkage_matrix: np.ndarray | None = None    # HRP: scipy linkage (N-1, 4) for the dendrogram
    leaf_order: list[int] | None = None         # HRP: quasi-diagonalisation order (column positions)
    cluster_labels: np.ndarray | None = None    # HRP: fcluster labels, k by two-diff gap statistic


# ---------------------------------------------------------------------------
# Categories / exposures
# ---------------------------------------------------------------------------
def load_categories(path: Path | str = "data/universe.json") -> dict[str, str]:
    """``fund_id -> categoría`` desde el universo aprobado (D3)."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f["fund_id"]: f["cat"] for f in cfg["funds"]}


def _exposure_vectors(columns: list[str], categories: Mapping[str, str], c: ProfileConstraints):
    cats = [categories[col] for col in columns]
    rv = np.array([1.0 if k in RV_CATEGORIES else c.mixto_rv_share if k == "MIXTO" else 0.0 for k in cats])
    rf = np.array([1.0 if k in RF_CATEGORIES else 1.0 - c.mixto_rv_share if k == "MIXTO" else 0.0 for k in cats])
    intl = np.array([1.0 if k == "RV_INTL" else 0.0 for k in cats])
    return rv, rf, intl


def exposures(weights: pd.Series, categories: Mapping[str, str], c: ProfileConstraints) -> dict[str, float]:
    """Exposición RV / RF / RV internacional con look-through de mixtos (D7)."""
    rv, rf, intl = _exposure_vectors(list(weights.index), categories, c)
    w = weights.to_numpy(dtype=float)
    return {"rv": float(rv @ w), "rf": float(rf @ w), "rv_intl": float(intl @ w)}


def _linear_constraints(columns: list[str], categories: Mapping[str, str], c: ProfileConstraints):
    """Bandas como ``A w <= b`` (grupos + techos y pisos por fondo)."""
    n = len(columns)
    rv, rf, intl = _exposure_vectors(columns, categories, c)
    eye = np.eye(n)
    a = np.vstack([rv, -rf, intl, -rv, eye, -eye])
    b = np.concatenate([[c.max_rv, -c.min_rf, c.max_rv_intl, -c.min_rv],
                        np.full(n, c.max_weight), np.full(n, -c.min_weight)])
    return a, b


def _check_feasible(a: np.ndarray, b: np.ndarray) -> None:
    n = a.shape[1]
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0), [cp.sum(w) == 1, a @ w <= b])
    prob.solve(solver=QP_SOLVER)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise InfeasibleConstraintsError(f"band polyhedron is empty for {n} funds (status={prob.status})")


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def _estimate(returns: pd.DataFrame, cov_method: str, periods_per_year: int) -> tuple[pd.Series, pd.DataFrame]:
    """Media histórica y covarianza (Ledoit-Wolf por D8, o muestral) anualizadas."""
    mu = returns.mean() * periods_per_year
    if cov_method == "ledoit":
        # [VERIFICADO] riskfolio-lib 7.3.0 ``method_cov='ledoit'`` usa el mismo sklearn LedoitWolf.
        cov = LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_
    elif cov_method == "hist":
        cov = returns.cov().to_numpy()
    else:
        raise ValueError(f"unknown cov_method {cov_method!r} (expected 'ledoit' or 'hist')")
    cov = pd.DataFrame(cov * periods_per_year, index=returns.columns, columns=returns.columns)
    return mu, cov


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def _solve_markowitz(mu: pd.Series, cov: pd.DataFrame, a: np.ndarray, b: np.ndarray,
                     c: ProfileConstraints, rf: float) -> np.ndarray:
    """Máx. Sharpe con PyPortfolioOpt.

    [VERIFICADO] PyPortfolioOpt 1.6.0: ``EfficientFrontier(expected_returns, cov_matrix,
    weight_bounds=(0, 1), solver=None, verbose=False, solver_options=None)``;
    ``add_constraint(new_constraint)``; ``max_sharpe(risk_free_rate=0.0)``. ``max_sharpe`` reescribe
    cada restricción ``A w <= b`` como ``A w <= b·k`` (homogénea), por lo que las bandas sobreviven la
    transformación de variable. Sin activo con retorno > rf lanza ``ValueError``.

    Solver: CLARABEL con tolerancias 1e-10 (``solver``/``solver_options`` de ``EfficientFrontier``).
    Con el solver por defecto la transformación ``w/k`` deja violaciones de banda de hasta 1e-5 sobre
    ventanas reales (medido en 648 ventanas); con CLARABEL ajustado el máximo es 1e-13, así que las
    bandas se verifican a ``BAND_TOL`` sin ningún post-proceso.
    """
    from pypfopt import EfficientFrontier
    from pypfopt import exceptions as pf_exc

    ef = EfficientFrontier(mu, cov, weight_bounds=(c.min_weight, c.max_weight),
                           solver=QP_SOLVER, solver_options=MARKOWITZ_SOLVER_OPTIONS)
    ef.add_constraint(lambda w: a @ w <= b)
    try:
        ef.max_sharpe(risk_free_rate=rf)
    except (ValueError, pf_exc.OptimizationError) as exc:
        raise OptimizationError(f"markowitz: {exc}") from exc
    return np.array([ef.weights[i] for i in range(len(mu))], dtype=float)


def _solve_risk_parity(returns: pd.DataFrame, a: np.ndarray, b: np.ndarray, cov_method: str) -> np.ndarray:
    """Paridad de riesgo con Riskfolio-Lib.

    [VERIFICADO] riskfolio-lib 7.3.0: ``Portfolio(returns=...)``;
    ``assets_stats(method_mu='hist', method_cov='hist', method_kurt=None, ...)`` con
    ``method_cov in {'hist','ledoit',...}``; ``rp_optimization(model='Classic', rm='MV', rf=0,
    b=None, b_f=None, hist=True)``. Restricciones lineales vía ``ainequality``/``binequality`` con la
    convención ``A w <= b`` (código: ``constraints += [A @ w - B @ k <= 0]``, homogénea en ``k``, así
    que sobrevive la normalización final). ``upperlng``/``lowerlng`` NO se aplican en
    ``rp_optimization`` → los techos por fondo van dentro de ``A``.
    """
    import riskfolio as rp

    # Percent units: daily variances ~1e-7 trip riskfolio's is_pos_def(threshold=1e-6) although Σ is PD;
    # risk-parity weights are invariant to a common scaling of Σ.
    port = rp.Portfolio(returns=returns * 100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        port.assets_stats(method_mu="hist", method_cov=cov_method)
        port.ainequality = a
        port.binequality = b.reshape(-1, 1)
        w = port.rp_optimization(model="Classic", rm="MV", rf=0, b=None, hist=True)
    if w is None:
        raise OptimizationError("risk_parity: riskfolio returned no solution")
    return w["weights"].reindex(returns.columns).to_numpy(dtype=float)


def _solve_hrp(returns: pd.DataFrame, cov_method: str):
    """HRP sin restricciones con Riskfolio-Lib; devuelve también los datos del dendrograma.

    [VERIFICADO] riskfolio-lib 7.3.0: ``HCPortfolio(returns=...)``;
    ``optimization(model='HRP', codependence='pearson', obj='MinRisk', rm='MV', rf=0, l=2,
    method_mu='hist', method_cov='hist', ..., linkage='single', ..., leaf_order=True, ...)`` →
    ``DataFrame`` (N×1, columna ``weights``). Atributos poblados: ``clustering`` (linkage scipy
    (N-1, 4)), ``sort_order`` (orden de hojas), ``codep`` (matriz de correlación). Distancia HRP para
    ``codependence='pearson'``: ``sqrt((1 - corr) / 2)``.
    """
    import riskfolio as rp

    hc = rp.HCPortfolio(returns=returns * 100.0)   # percent units, see _solve_risk_parity; HRP is scale-invariant
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = hc.optimization(model="HRP", codependence="pearson", rm="MV", method_cov=cov_method,
                            linkage="single", leaf_order=True)
    raw = w["weights"].reindex(returns.columns).to_numpy(dtype=float)
    linkage = np.asarray(hc.clustering, dtype=float)
    leaf_order = [int(i) for i in np.asarray(hc.sort_order).ravel()]
    codep = np.asarray(hc.codep, dtype=float)
    return raw, linkage, leaf_order, codep


def _cluster_labels(codep: np.ndarray, linkage: np.ndarray) -> np.ndarray:
    """k óptimo por el estadístico two-diff gap de riskfolio, luego ``fcluster`` (etiquetas 1..k).

    [VERIFICADO] riskfolio-lib 7.3.0: ``AuxFunctions.two_diff_gap_stat(dist, clustering, max_k=10)``
    con ``dist`` DataFrame → devuelve ``(k, clustering_inds)``.
    """
    import riskfolio as rp

    n = codep.shape[0]
    if n < 3:
        return np.ones(n, dtype=int)
    dist = pd.DataFrame(np.sqrt(np.clip((1.0 - codep) / 2.0, 0.0, 1.0)))
    k, _ = rp.AuxFunctions.two_diff_gap_stat(dist, linkage, max_k=min(10, n - 1))
    return fcluster(linkage, max(1, int(k)), criterion="maxclust")


def _project_to_bands(w_raw: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Proyección euclidiana exacta de ``w_raw`` sobre {w : 1'w = 1, A w <= b} (D5)."""
    n = len(w_raw)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(w - w_raw)), [cp.sum(w) == 1, a @ w <= b])
    prob.solve(solver=QP_SOLVER)
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return None
    return np.asarray(w.value, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _validate(returns: pd.DataFrame, method: str, profile: str, categories: Mapping[str, str]) -> None:
    if not isinstance(returns, pd.DataFrame) or returns.shape[1] < 2:
        raise ValueError("returns must be a DataFrame with at least 2 columns")
    if returns.shape[0] < 2:
        raise ValueError("returns must have at least 2 rows")
    if returns.isna().any().any():
        raise ValueError("returns contain NaN — slice/mask point-in-time before calling optimize()")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    for col in returns.columns:
        if col not in categories:
            raise ValueError(f"no category for fund {col!r}")
        if categories[col] not in CATEGORIES:
            raise ValueError(f"unknown category {categories[col]!r} for {col!r}; expected one of {sorted(CATEGORIES)}")


def _check_bands(w: np.ndarray, a: np.ndarray, b: np.ndarray, method: str) -> None:
    if not np.isfinite(w).all() or abs(w.sum() - 1.0) > BAND_TOL or (a @ w - b > BAND_TOL).any():
        raise OptimizationError(f"{method}: solver returned weights outside the bands (sum={w.sum():.8f})")


def optimize(
    returns: pd.DataFrame,
    method: Method,
    profile: Profile,
    constraints: ProfileConstraints | None = None,
    *,
    categories: Mapping[str, str],
    rf: float = 0.0,
    periods_per_year: int = 252,
    cov_method: CovMethod = "ledoit",
) -> OptimizationResult:
    """Pesos óptimos para ``method`` × ``profile`` sobre la ventana ``returns`` (sin NaN).

    Args:
        returns: retornos diarios (log o simples) T×N, columnas = ``fund_id``. Sin NaN; el recorte
            point-in-time es responsabilidad del caller.
        method: ``"markowitz" | "risk_parity" | "hrp"``.
        profile: ``"conservador" | "moderado" | "agresivo"``.
        constraints: bandas; por defecto ``PROFILE_CONSTRAINTS[profile]`` (D6).
        categories: ``fund_id -> categoría`` (ver ``load_categories``).
        rf: tasa libre de riesgo anual para el Sharpe / tangencia.
        periods_per_year: anualización (252 por el gate §4.3).
        cov_method: ``"ledoit"`` (D8, default) o ``"hist"`` (muestral; útil para casos analíticos).

    Raises:
        ValueError: inputs inválidos.
        InfeasibleConstraintsError: bandas incompatibles con el universo.
        OptimizationError: fallo del solver o pesos fuera de banda.
    """
    _validate(returns, method, profile, categories)
    c = constraints if constraints is not None else PROFILE_CONSTRAINTS[profile]
    cols = list(returns.columns)
    a, b = _linear_constraints(cols, categories, c)
    _check_feasible(a, b)
    mu, cov = _estimate(returns, cov_method, periods_per_year)

    extra: dict = {}
    try:
        if method == "markowitz":
            w = _solve_markowitz(mu, cov, a, b, c, rf)
        elif method == "risk_parity":
            w = _solve_risk_parity(returns, a, b, cov_method)
        else:
            raw, linkage, leaf_order, codep = _solve_hrp(returns, cov_method)
            w = _project_to_bands(raw, a, b)
            if w is None:
                raise OptimizationError("hrp: QP projection onto the bands failed")
            extra = {
                "raw_weights": pd.Series(raw, index=cols, name="raw_weights"),
                "linkage_matrix": linkage,
                "leaf_order": leaf_order,
                "cluster_labels": _cluster_labels(codep, linkage),
            }
    except OptimizationError:
        raise
    except Exception as exc:  # solver-level failures of any flavour
        raise OptimizationError(f"{method}: {exc}") from exc

    _check_bands(w, a, b, method)
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    weights = pd.Series(w, index=cols, name="weights")
    er = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov.to_numpy() @ weights))
    return OptimizationResult(
        weights=weights,
        expected_return=er,
        volatility=vol,
        sharpe=(er - rf) / vol if vol > 0 else float("nan"),
        method=method,
        profile=profile,
        exposures=exposures(weights, categories, c),
        **extra,
    )
