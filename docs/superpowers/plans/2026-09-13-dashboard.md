# Fase 4 Dashboard (dendrograma HRP ↔ AG Grid) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dash app where hovering a leaf/link of the HRP dendrogram highlights the matching fund rows in an AG Grid, with profile/method/window switches served from a Flask-Caching memo of the distance matrix.

**Architecture:** `dashboard/` package with a `create_app()` factory (no mutable module globals; all session state in `dcc.Store`). Pure data/cache/figure/grid modules feed thin Dash callbacks. `distance_bundle` (corr → distance → single linkage → leaf order → cluster labels → Ledoit-Wolf Σ) is memoized on `(universe, start, end, fingerprint)` — profile and method are *not* in the key. Callbacks are tested by POSTing to `/_dash-update-component` with Flask's test client (real dispatch, no browser).

**Tech Stack:** Python 3.14 (`.venv/`), dash 4.4.1, dash-ag-grid 35.3.0 (AG Grid 35.3.1), flask-caching 2.5.1, dash-bootstrap-components 2.0.4, plotly 6.8.0, scipy 1.17.1, existing `optimization.portfolio_optimizer` + `backtest.walk_forward`.

**Spec:** `docs/superpowers/specs/2026-09-13-dashboard-design.md`

## Global Constraints

- Pins (exact, already installed in `.venv/`): `dash[testing]==4.4.1`, `dash-ag-grid==35.3.0`, `flask-caching==2.5.1`, `dash-bootstrap-components==2.0.4`. Add them to `requirements.txt` verbatim.
- No mutable module-level session state; per-client state only in `dcc.Store`. Cache bound to `app.server` via `cache.init_app(app.server, cfg)`.
- Distance cache key = `(universe, start, end, fingerprint)`; TTL constant `DIST_TTL_SECONDS = 24 * 3600` with the justification comment quoted in Task 2.
- Bands D6 / methods / profiles come from `optimization.portfolio_optimizer` (`PROFILES`, `METHODS`, `optimize`). Never re-implement the optimizer.
- Window slicing reuses `backtest.walk_forward.active_universe` / `estimation_window` / `rebalance_dates` with `BacktestConfig(estimation_start=str(index[0].date()), window_months=lookback)`.
- Out of scope: PDF/tear sheet export, URL state persistence, grid → dendrogram reverse sync.
- All commands run from the repo root with `.venv/bin/python`. Tests: `.venv/bin/python -m pytest tests/test_dashboard.py -q`.
- Commit after every task; commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Code comments/docstrings in Spanish (matches `optimization/`, `backtest/`); identifiers in English.

## File map

| File | Responsibility |
|---|---|
| `dashboard/__init__.py` | package docstring |
| `dashboard/data.py` | `load_returns`, `load_labels`, `fingerprint`, `asof_options`, `Window`, `window_slice` |
| `dashboard/cache.py` | `cache`, `DIST_TTL_SECONDS`, `cache_config`, `DistanceBundle`, `_compute_distance`, `distance_bundle`, `PortfolioBundle`, `_compute_portfolio`, `portfolio_bundle` |
| `dashboard/figures.py` | `Link`, `dendrogram_geometry`, `build_dendrogram` |
| `dashboard/grid.py` | `column_defs`, `DEFAULT_COL_DEF`, `risk_contributions`, `fund_metrics`, `build_rows` |
| `dashboard/app.py` | `DashboardData`, `load_data`, `build_layout`, `register_callbacks`, `create_app`, `__main__` |
| `dashboard/assets/dashboard.css` | selected-row colour, cursor |
| `dashboard/README.md` | how to run / test / cache env vars |
| `tests/test_dashboard.py` | all tests (fixtures at top) |
| `requirements.txt`, `pyproject.toml` | pins / optional group `dashboard` |

---

### Task 1: Dependencies + data layer (`dashboard/data.py`)

**Files:**
- Modify: `requirements.txt` (append 4 lines), `pyproject.toml` (optional group)
- Create: `dashboard/__init__.py`, `dashboard/data.py`
- Test: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `backtest.walk_forward.BacktestConfig, BacktestError, active_universe, estimation_window, rebalance_dates`.
- Produces:
  - `load_returns(path=RETURNS_PATH) -> pd.DataFrame` (sorted DatetimeIndex)
  - `load_labels(path=FUNDS_MASTER_PATH) -> dict[str, str]` (`fund_id -> nombre`, `{}` if file missing)
  - `fingerprint(returns: pd.DataFrame, path: Path | None = None) -> str` (16 hex)
  - `asof_options(index: pd.DatetimeIndex, lookback_months: int) -> list[pd.Timestamp]` (ends with `index[-1]`)
  - `Window(universe: tuple[str, ...], frame: pd.DataFrame)` with `.start`, `.end` (ISO strings)
  - `window_slice(returns, asof: pd.Timestamp, lookback_months: int) -> Window` (raises `BacktestError` if `< 2` funds)
  - Constants `LOOKBACK_OPTIONS = (12, 24, 36)`, `DEFAULT_LOOKBACK = 24`.

- [ ] **Step 1: Pin dependencies**

Append to `requirements.txt` (after `pytest-cov==7.1.0`):

```
# Fase 4 dashboard (dash[testing] pins selenium<=4.2.0 → no Selenium Manager; chromedriver must be on PATH for E2E)
dash[testing]==4.4.1
dash-ag-grid==35.3.0
flask-caching==2.5.1
dash-bootstrap-components==2.0.4
```

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
dashboard = ["dash[testing]==4.4.1", "dash-ag-grid==35.3.0", "flask-caching==2.5.1", "dash-bootstrap-components==2.0.4"]
```

Run: `.venv/bin/pip install -r requirements.txt -q && .venv/bin/python -c "import dash, dash_ag_grid, flask_caching, dash_bootstrap_components as dbc; print(dash.__version__, dash_ag_grid.__version__)"`
Expected: `4.4.1 35.3.0`

- [ ] **Step 2: Write the failing tests (fixtures + data layer)**

Create `tests/test_dashboard.py`:

```python
"""Fase 4 — dashboard: datos, cache, figura, grid, callbacks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = json.loads((ROOT / "data/universe.json").read_text(encoding="utf-8"))["funds"]
FUND_IDS = [f["fund_id"] for f in UNIVERSE]
CATS = {f["fund_id"]: f["cat"] for f in UNIVERSE}
BLOCK = {"RF_CORTO": 0, "RF_LARGO": 0, "MIXTO": 1, "RV_LOCAL": 2, "RV_INTL": 2}
DAILY_VOL = {"RF_CORTO": 0.0005, "RF_LARGO": 0.002, "MIXTO": 0.005, "RV_LOCAL": 0.012, "RV_INTL": 0.010}


def make_returns(seed: int = 7, start: str = "2021-01-04", periods: int = 1040) -> pd.DataFrame:
    """21 fondos del universo aprobado, ~4 años hábiles, 3 bloques correlacionados (ρ intra-bloque ≈ 0.64)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    factors = rng.standard_normal((periods, 3))
    cols = {}
    for fid in FUND_IDS:
        cat = CATS[fid]
        eps = rng.standard_normal(periods)
        drift = 0.0002 if cat.startswith("RF") else 0.0003
        cols[fid] = DAILY_VOL[cat] * (0.8 * factors[:, BLOCK[cat]] + 0.6 * eps) + drift
    return pd.DataFrame(cols, index=idx)


@pytest.fixture(scope="session")
def synthetic_returns() -> pd.DataFrame:
    return make_returns()


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------
def test_fingerprint_changes_with_data(synthetic_returns):
    from dashboard.data import fingerprint

    a = fingerprint(synthetic_returns)
    b = fingerprint(synthetic_returns * 1.0001)
    assert len(a) == 16 and a != b
    assert fingerprint(synthetic_returns) == a


def test_fingerprint_from_file(tmp_path, synthetic_returns):
    from dashboard.data import fingerprint

    p = tmp_path / "r.parquet"
    synthetic_returns.to_parquet(p)
    assert fingerprint(synthetic_returns, p) == fingerprint(synthetic_returns, p)
    assert fingerprint(synthetic_returns, p) != fingerprint(synthetic_returns, None)


def test_asof_options_end_with_last_date(synthetic_returns):
    from dashboard.data import asof_options

    idx = synthetic_returns.index
    opts = asof_options(idx, 24)
    assert opts[-1] == idx[-1]
    assert all(a < b for a, b in zip(opts, opts[1:]))
    assert len(opts) > 1                      # quarter ends after 24 m of history
    assert asof_options(idx, 36)[-1] == idx[-1]


def test_window_slice_matches_backtest_semantics(synthetic_returns):
    from dashboard.data import window_slice

    asof = synthetic_returns.index[-1]
    win = window_slice(synthetic_returns, asof, 24)
    assert set(win.universe) == set(FUND_IDS)
    assert win.frame.index[-1] == asof
    assert win.frame.index[0] > asof - pd.DateOffset(months=24)
    assert not win.frame.isna().any().any()
    assert win.start == str(win.frame.index[0].date()) and win.end == str(asof.date())
    # reproducible from the key: loc[start:end, universe].fillna(0) == frame
    again = synthetic_returns.loc[win.start:win.end, list(win.universe)].fillna(0.0)
    assert again.equals(win.frame)


def test_window_slice_rejects_tiny_universe(synthetic_returns):
    from backtest.walk_forward import BacktestError
    from dashboard.data import window_slice

    two = synthetic_returns[FUND_IDS[:2]].copy()
    two[FUND_IDS[1]] = np.nan                # only one live fund
    with pytest.raises(BacktestError):
        window_slice(two, two.index[-1], 12)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: 5 failures with `ModuleNotFoundError: No module named 'dashboard'`.

- [ ] **Step 4: Implement `dashboard/__init__.py` and `dashboard/data.py`**

`dashboard/__init__.py`:

```python
"""Fase 4 — dashboard Dash: dendrograma HRP interactivo sincronizado con un AG Grid de pesos y riesgo."""
```

`dashboard/data.py`:

```python
"""Carga de datos y recorte de ventana para el dashboard (Fase 4).

Reutiliza la semántica point-in-time del backtest (``active_universe`` / ``estimation_window``) para que la
ventana que ve el dashboard sea exactamente la que vería el walk-forward en esa fecha.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.walk_forward import BacktestConfig, BacktestError, active_universe, estimation_window, rebalance_dates

RETURNS_PATH = Path("data/cleaned/returns_matrix.parquet")
FUNDS_MASTER_PATH = Path("data/cleaned/funds_master.parquet")
LOOKBACK_OPTIONS: tuple[int, ...] = (12, 24, 36)
DEFAULT_LOOKBACK = 24  # D8


def load_returns(path: Path | str = RETURNS_PATH) -> pd.DataFrame:
    """Matriz de retornos log diarios (índice DatetimeIndex ordenado, columnas = fund_id)."""
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_labels(path: Path | str = FUNDS_MASTER_PATH) -> dict[str, str]:
    """``fund_id -> nombre`` desde ``funds_master``; vacío si el archivo no existe (tests / entornos sin ingest)."""
    if not Path(path).exists():
        return {}
    fm = pd.read_parquet(path, columns=["fund_id", "nombre"])
    return dict(zip(fm["fund_id"], fm["nombre"]))


def fingerprint(returns: pd.DataFrame, path: Path | str | None = None) -> str:
    """16 hex de sha256 del parquet (si existe) o del contenido del DataFrame. Forma parte de la clave de cache:
    un re-ingest cambia el fingerprint y por tanto invalida la cache sin reiniciar el proceso."""
    h = hashlib.sha256()
    if path is not None and Path(path).exists():
        h.update(Path(path).read_bytes())
    else:
        h.update(pd.util.hash_pandas_object(returns, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, returns.columns)).encode("utf-8"))
    return h.hexdigest()[:16]


def _config(index: pd.DatetimeIndex, lookback_months: int) -> BacktestConfig:
    return BacktestConfig(estimation_start=str(index[0].date()), window_months=lookback_months)


def asof_options(index: pd.DatetimeIndex, lookback_months: int) -> list[pd.Timestamp]:
    """Fechas de decisión del backtest (último día hábil de cada trimestre con ≥ lookback meses de historia)
    más la última fecha disponible."""
    try:
        dates = rebalance_dates(index, _config(index, lookback_months))
    except BacktestError:
        dates = []
    last = index[-1]
    return dates + [last] if not dates or dates[-1] != last else dates


@dataclass(frozen=True)
class Window:
    universe: tuple[str, ...]
    frame: pd.DataFrame  # (asof - lookback, asof], soporte común, NaN → 0 (política del backtest)

    @property
    def start(self) -> str:
        return str(self.frame.index[0].date())

    @property
    def end(self) -> str:
        return str(self.frame.index[-1].date())


def window_slice(returns: pd.DataFrame, asof: pd.Timestamp, lookback_months: int) -> Window:
    """Universo activo point-in-time y ventana de estimación en ``asof``. ``BacktestError`` si < 2 fondos."""
    cfg = _config(returns.index, lookback_months)
    universe = tuple(active_universe(returns, asof, cfg))
    if len(universe) < 2:
        raise BacktestError(f"solo {len(universe)} fondo(s) activo(s) en {asof.date()} con {lookback_months} m de historia")
    frame = estimation_window(returns[list(universe)], asof, cfg)
    return Window(universe=universe, frame=frame)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml dashboard/__init__.py dashboard/data.py tests/test_dashboard.py
git commit -m "feat(dashboard): data layer + pinned Dash deps

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Cache layer (`dashboard/cache.py`)

**Files:**
- Create: `dashboard/cache.py`
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `Window` from Task 1; `optimization.portfolio_optimizer.optimize, _estimate, _cluster_labels`.
- Produces:
  - `cache: flask_caching.Cache` (uninitialised; `create_app` calls `cache.init_app(app.server, cfg)`)
  - `DIST_TTL_SECONDS: int`, `PERIODS_PER_YEAR = 252`
  - `cache_config(env: Mapping[str, str] = os.environ) -> dict`
  - `DistanceBundle(fund_ids, corr, dist, linkage, leaf_order, cluster_labels, cov, mu)`
  - `_compute_distance(window: pd.DataFrame) -> DistanceBundle` (pure; tests monkeypatch it)
  - `distance_bundle(window, universe, start, end, fingerprint) -> DistanceBundle` (memoized, `window` ignored in key)
  - `PortfolioBundle(method, profile, weights: dict, raw_weights: dict | None, expected_return, volatility, sharpe, exposures)`
  - `_compute_portfolio(window, categories, method, profile) -> PortfolioBundle`
  - `portfolio_bundle(window, categories, universe, start, end, fingerprint, method, profile) -> PortfolioBundle` (memoized, `window`/`categories` ignored)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------
@pytest.fixture
def flask_cache_app():
    """Flask app mínima con la cache del dashboard inicializada (SimpleCache fresca por test)."""
    from flask import Flask

    from dashboard.cache import cache, cache_config

    app = Flask("cache-test")
    cache.init_app(app, cache_config({}))
    with app.app_context():
        yield app


def test_cache_config_defaults_and_env():
    from dashboard.cache import DIST_TTL_SECONDS, cache_config

    cfg = cache_config({})
    assert cfg["CACHE_TYPE"] == "SimpleCache" and cfg["CACHE_DEFAULT_TIMEOUT"] == DIST_TTL_SECONDS == 86400
    fs = cache_config({"DASH_CACHE_TYPE": "FileSystemCache", "DASH_CACHE_DIR": "/tmp/x"})
    assert fs["CACHE_TYPE"] == "FileSystemCache" and fs["CACHE_DIR"] == "/tmp/x"


def test_distance_bundle_replicates_optimizer_hrp(flask_cache_app, synthetic_returns):
    from dashboard.cache import distance_bundle
    from dashboard.data import fingerprint, window_slice
    from optimization.portfolio_optimizer import optimize

    win = window_slice(synthetic_returns, synthetic_returns.index[-1], 24)
    b = distance_bundle(win.frame, win.universe, win.start, win.end, fingerprint(synthetic_returns))
    res = optimize(win.frame, "hrp", "moderado", categories=CATS)
    assert np.allclose(b.linkage, res.linkage_matrix)
    assert b.leaf_order == res.leaf_order
    assert b.cluster_labels == [int(k) for k in res.cluster_labels]
    assert b.fund_ids == win.universe and b.cov.shape == (len(win.universe),) * 2
    assert np.allclose(np.diag(b.dist), 0.0) and (b.dist >= 0).all() and (b.dist <= 1).all()


def test_distance_cache_ignores_profile_and_method(flask_cache_app, synthetic_returns, monkeypatch):
    import dashboard.cache as dc
    from dashboard.data import fingerprint, window_slice

    calls = {"n": 0}
    real = dc._compute_distance

    def counting(window):
        calls["n"] += 1
        return real(window)

    monkeypatch.setattr(dc, "_compute_distance", counting)
    fp = fingerprint(synthetic_returns)
    win = window_slice(synthetic_returns, synthetic_returns.index[-1], 24)
    for profile in ("conservador", "moderado", "agresivo"):
        for method in ("markowitz", "risk_parity", "hrp"):
            dc.distance_bundle(win.frame, win.universe, win.start, win.end, fp)
            dc.portfolio_bundle(win.frame, CATS, win.universe, win.start, win.end, fp, method, profile)
    assert calls["n"] == 1, "9 combinaciones sobre la misma ventana deben reutilizar la matriz de distancia"

    win12 = window_slice(synthetic_returns, synthetic_returns.index[-1], 12)
    dc.distance_bundle(win12.frame, win12.universe, win12.start, win12.end, fp)
    assert calls["n"] == 2, "cambiar la ventana sí recalcula"
    dc.distance_bundle(win.frame, win.universe, win.start, win.end, fp)
    assert calls["n"] == 2, "volver a la ventana anterior lee de cache"
    dc.distance_bundle(win.frame, win.universe, win.start, win.end, "other-fingerprint")
    assert calls["n"] == 3, "otro fingerprint (re-ingest) invalida"


def test_portfolio_bundle_is_memoized_and_serialisable(flask_cache_app, synthetic_returns, monkeypatch):
    import dashboard.cache as dc
    from dashboard.data import fingerprint, window_slice

    calls = {"n": 0}
    real = dc._compute_portfolio

    def counting(window, categories, method, profile):
        calls["n"] += 1
        return real(window, categories, method, profile)

    monkeypatch.setattr(dc, "_compute_portfolio", counting)
    fp = fingerprint(synthetic_returns)
    win = window_slice(synthetic_returns, synthetic_returns.index[-1], 24)
    p1 = dc.portfolio_bundle(win.frame, CATS, win.universe, win.start, win.end, fp, "hrp", "agresivo")
    p2 = dc.portfolio_bundle(win.frame, CATS, win.universe, win.start, win.end, fp, "hrp", "agresivo")
    assert calls["n"] == 1 and p1 == p2
    assert abs(sum(p1.weights.values()) - 1) < 1e-6 and p1.raw_weights is not None
    json.dumps({"w": p1.weights, "raw": p1.raw_weights, "exp": p1.exposures})   # JSON-safe for dcc.Store
    p3 = dc.portfolio_bundle(win.frame, CATS, win.universe, win.start, win.end, fp, "markowitz", "agresivo")
    assert calls["n"] == 2 and p3.raw_weights is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k "cache or bundle"`
Expected: 4 failures with `ModuleNotFoundError: No module named 'dashboard.cache'`.

- [ ] **Step 3: Implement `dashboard/cache.py`**

```python
"""Capa de cache del dashboard (Flask-Caching).

Dos funciones memoizadas, ambas atadas a ``app.server`` mediante ``cache.init_app`` en ``create_app``:

* ``distance_bundle`` — correlación → distancia HRP → linkage single → orden de hojas → etiquetas de cluster
  → Σ Ledoit-Wolf. Clave = (universo, inicio, fin, fingerprint). **Perfil y metodología no forman parte de la
  clave**: cambiar de perfil reutiliza la matriz de distancia.
* ``portfolio_bundle`` — ``optimize()`` para (método, perfil) sobre la misma ventana.

Réplica exacta de Riskfolio-Lib 7.3.0 ``HCPortfolio._hierarchical_clustering`` (codependence='pearson',
linkage='single', leaf_order=True): ``dist = sqrt(clip((1 - corr)/2, 0, 1))``,
``linkage(squareform(dist), 'single', optimal_ordering=True)``, ``leaves_list``. Verificado con
``np.allclose`` contra ``OptimizationResult.linkage_matrix`` en ``tests/test_dashboard.py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from flask_caching import Cache
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from optimization.portfolio_optimizer import _cluster_labels, _estimate, optimize

# TTL de la matriz de distancia. Los datos SFC se publican con cadencia diaria y el ``fingerprint`` (sha256 del
# parquet) ya forma parte de la clave, así que un re-ingest nunca sirve una matriz obsoleta. El TTL por tanto no
# protege la corrección: acota la memoria del backend y garantiza que una entrada huérfana (fingerprint viejo)
# desaparezca al cabo de un día, la cadencia natural de refresco de los datos.
DIST_TTL_SECONDS = 24 * 3600
PERIODS_PER_YEAR = 252  # gate §4.3

cache = Cache()


def cache_config(env: Mapping[str, str] = os.environ) -> dict:
    """``SimpleCache`` (proceso único, default) o ``FileSystemCache`` (gunicorn multi-worker) vía
    ``DASH_CACHE_TYPE`` / ``DASH_CACHE_DIR``."""
    cfg = {
        "CACHE_TYPE": env.get("DASH_CACHE_TYPE", "SimpleCache"),
        "CACHE_DEFAULT_TIMEOUT": DIST_TTL_SECONDS,
        "CACHE_THRESHOLD": 500,
    }
    if cfg["CACHE_TYPE"] == "FileSystemCache":
        cfg["CACHE_DIR"] = env.get("DASH_CACHE_DIR", ".cache/dashboard")
    return cfg


@dataclass(frozen=True)
class DistanceBundle:
    fund_ids: tuple[str, ...]
    corr: np.ndarray            # Pearson N×N
    dist: np.ndarray            # sqrt((1 - corr) / 2)
    linkage: np.ndarray         # scipy (N-1, 4), single, optimal ordering
    leaf_order: list[int]       # posiciones de columna, orden de hojas del dendrograma
    cluster_labels: list[int]   # fcluster 1..k, k por two-diff gap (misma función del optimizador)
    cov: np.ndarray             # Σ Ledoit-Wolf anualizada (misma del optimizador) — contribución al riesgo
    mu: np.ndarray              # media anualizada


def _compute_distance(window: pd.DataFrame) -> DistanceBundle:
    """Función pura sobre la ventana (sin NaN). Separada de la memoización para poder contar llamadas en tests."""
    corr = window.corr().to_numpy(dtype=float)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    z = linkage(squareform(dist, checks=False), method="single", optimal_ordering=True)
    labels = _cluster_labels(corr, z)
    mu, cov = _estimate(window, "ledoit", PERIODS_PER_YEAR)
    return DistanceBundle(
        fund_ids=tuple(str(c) for c in window.columns),
        corr=corr,
        dist=dist,
        linkage=np.asarray(z, dtype=float),
        leaf_order=[int(i) for i in leaves_list(z)],
        cluster_labels=[int(k) for k in labels],
        cov=cov.to_numpy(dtype=float),
        mu=mu.to_numpy(dtype=float),
    )


@cache.memoize(timeout=DIST_TTL_SECONDS, args_to_ignore=["window"])
def distance_bundle(window: pd.DataFrame, universe: tuple[str, ...], start: str, end: str, fingerprint: str) -> DistanceBundle:
    """Memoizada por (universe, start, end, fingerprint). ``window`` se ignora en la clave porque es derivable de
    ella (``returns.loc[start:end, universe].fillna(0)``); se pasa para no recargar datos aquí."""
    return _compute_distance(window)


@dataclass(frozen=True)
class PortfolioBundle:
    method: str
    profile: str
    weights: dict[str, float]
    raw_weights: dict[str, float] | None   # HRP: pesos antes de la proyección QP (D5)
    expected_return: float
    volatility: float
    sharpe: float
    exposures: dict[str, float]


def _compute_portfolio(window: pd.DataFrame, categories: Mapping[str, str], method: str, profile: str) -> PortfolioBundle:
    res = optimize(window, method, profile, categories=categories, periods_per_year=PERIODS_PER_YEAR)
    return PortfolioBundle(
        method=method,
        profile=profile,
        weights={str(k): float(v) for k, v in res.weights.items()},
        raw_weights=None if res.raw_weights is None else {str(k): float(v) for k, v in res.raw_weights.items()},
        expected_return=float(res.expected_return),
        volatility=float(res.volatility),
        sharpe=float(res.sharpe),
        exposures={k: float(v) for k, v in res.exposures.items()},
    )


@cache.memoize(timeout=DIST_TTL_SECONDS, args_to_ignore=["window", "categories"])
def portfolio_bundle(window: pd.DataFrame, categories: Mapping[str, str], universe: tuple[str, ...], start: str,
                     end: str, fingerprint: str, method: str, profile: str) -> PortfolioBundle:
    """Memoizada por (universe, start, end, fingerprint, method, profile)."""
    return _compute_portfolio(window, categories, method, profile)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: `9 passed` (the two HRP-heavy tests take ~5–10 s).

- [ ] **Step 5: Commit**

```bash
git add dashboard/cache.py tests/test_dashboard.py
git commit -m "feat(dashboard): flask-caching memo of distance matrix + portfolio bundle

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Dendrogram figure (`dashboard/figures.py`)

**Files:**
- Create: `dashboard/figures.py`
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `DistanceBundle.linkage/leaf_order/cluster_labels/fund_ids` (Task 2).
- Produces:
  - `Link(node_id: int, height: float, pos: tuple[float, float, float, float], heights: tuple[float, ...], leaves: tuple[int, ...])`
  - `dendrogram_geometry(Z: np.ndarray) -> tuple[list[int], list[Link]]`
  - `build_dendrogram(Z, fund_ids: Sequence[str], labels: Mapping[str, str], cluster_labels: Sequence[int], weights: Mapping[str, float], highlight: Iterable[str] = (), title: str | None = None) -> go.Figure`
  - Trace contract used by the `sync` callback: every link trace has `customdata[i] == [fund_id, ...]` (list, subtree) for every point; the last trace (`name="leaves"`) has `customdata[i] == [fund_id]`.
  - Constants `LINK_WIDTH=2.0`, `LINK_WIDTH_HI=5.0`, `MARKER_SIZE=9`, `MARKER_SIZE_HI=15`, `MIXED_COLOR="#9aa0a6"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# figures.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def bundle(synthetic_returns):
    from dashboard.cache import _compute_distance
    from dashboard.data import window_slice

    return _compute_distance(window_slice(synthetic_returns, synthetic_returns.index[-1], 24).frame)


def test_dendrogram_geometry_maps_links_to_subtrees(bundle):
    from scipy.cluster.hierarchy import leaves_list

    from dashboard.figures import dendrogram_geometry

    leaves, links = dendrogram_geometry(bundle.linkage)
    n = len(bundle.fund_ids)
    assert leaves == [int(i) for i in leaves_list(bundle.linkage)] == bundle.leaf_order
    assert len(links) == n - 1
    root = max(links, key=lambda l: len(l.leaves))
    assert sorted(root.leaves) == list(range(n)) and root.height == pytest.approx(bundle.linkage[-1, 2])
    for lk in links:
        row = lk.node_id - n
        assert lk.height == pytest.approx(bundle.linkage[row, 2])
        pos = {leaf: 10 * i + 5 for i, leaf in enumerate(leaves)}
        xs = [pos[l] for l in lk.leaves]
        assert min(xs) <= min(lk.pos) and max(lk.pos) <= max(xs)   # U lies inside its subtree span


def test_build_dendrogram_trace_contract(bundle):
    from dashboard.figures import LINK_WIDTH, LINK_WIDTH_HI, MARKER_SIZE, MARKER_SIZE_HI, build_dendrogram

    ids = list(bundle.fund_ids)
    weights = {f: 1 / len(ids) for f in ids}
    fig = build_dendrogram(bundle.linkage, ids, {}, bundle.cluster_labels, weights)
    n = len(ids)
    assert len(fig.data) == n            # n-1 links + 1 leaf-marker trace
    leaves = fig.data[-1]
    assert leaves.name == "leaves" and len(leaves.x) == n
    assert sorted(cd[0] for cd in leaves.customdata) == sorted(ids)
    for tr in fig.data[:-1]:
        assert tr.mode == "lines" and len(tr.customdata) == len(tr.x) > 4      # densified U
        subtree = tr.customdata[0]
        assert isinstance(subtree, (list, tuple)) and set(subtree) <= set(ids)
        assert all(list(cd) == list(subtree) for cd in tr.customdata)
        assert tr.line.width == LINK_WIDTH
    assert all(s == MARKER_SIZE for s in leaves.marker.size)
    # horizontal orientation: leaf positions on y, distances on x
    assert list(fig.layout.yaxis.ticktext) == [ids[i] for i in bundle.leaf_order]
    assert max(max(tr.x) for tr in fig.data[:-1]) == pytest.approx(bundle.linkage[-1, 2])


def test_build_dendrogram_highlight(bundle):
    from dashboard.figures import LINK_WIDTH_HI, MARKER_SIZE_HI, build_dendrogram, dendrogram_geometry

    ids = list(bundle.fund_ids)
    _, links = dendrogram_geometry(bundle.linkage)
    small = min(links, key=lambda l: len(l.leaves))          # a 2-leaf link
    pinned = {ids[i] for i in small.leaves}
    fig = build_dendrogram(bundle.linkage, ids, {}, bundle.cluster_labels, {}, highlight=pinned)
    hi_links = [tr for tr in fig.data[:-1] if tr.line.width == LINK_WIDTH_HI]
    assert hi_links and all(set(tr.customdata[0]) <= pinned for tr in hi_links)
    leaves = fig.data[-1]
    sizes = dict(zip((cd[0] for cd in leaves.customdata), leaves.marker.size))
    assert all(sizes[f] == MARKER_SIZE_HI for f in pinned)
    assert sum(s == MARKER_SIZE_HI for s in sizes.values()) == len(pinned)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k dendrogram`
Expected: 3 failures, `ModuleNotFoundError: No module named 'dashboard.figures'`.

- [ ] **Step 3: Implement `dashboard/figures.py`**

```python
"""Dendrograma interactivo del HRP.

``plotly.figure_factory.create_dendrogram`` se evaluó y se descartó: sus trazas no exponen la relación
nodo → hojas (necesaria para que el hover de un enlace resalte todo su subárbol) y solo colorean por
``color_threshold``. Se usa la misma geometría base de scipy: ``dendrogram(Z, no_plot=True,
link_color_func=str)`` devuelve en ``color_list`` el id de nodo de cada enlace **en el mismo orden** que
``icoord``/``dcoord`` (scipy anexa los tres en la misma recursión), y ``to_tree(Z, rd=True)`` da las hojas
de cada nodo.

Contrato de trazas (lo consume el callback ``sync`` de ``app.py``):
* una traza ``mode="lines"`` por enlace, ``customdata[i] = [fund_id, ...]`` (subárbol) para todos sus puntos;
* última traza ``name="leaves"`` con un marcador por hoja, ``customdata[i] = [fund_id]``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram, to_tree

LEAF_STEP = 10.0                      # scipy sitúa las hojas en 5, 15, 25, …
POINTS_PER_SEGMENT = 8                # densificación de la U para que el hover responda en todo el trazo
LINK_WIDTH, LINK_WIDTH_HI = 2.0, 5.0
MARKER_SIZE, MARKER_SIZE_HI = 9, 15
MIXED_COLOR = "#9aa0a6"               # enlace que une clusters distintos
CLUSTER_PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f")


@dataclass(frozen=True)
class Link:
    node_id: int                        # id de nodo scipy (≥ n)
    height: float                       # distancia de fusión
    pos: tuple[float, float, float, float]      # icoord: 4 posiciones sobre el eje de hojas
    heights: tuple[float, float, float, float]  # dcoord: 4 alturas (0/h_izq, h, h, h_der/0)
    leaves: tuple[int, ...]             # posiciones de columna bajo el nodo


def dendrogram_geometry(z: np.ndarray) -> tuple[list[int], list[Link]]:
    """(orden de hojas, enlaces) a partir de la matriz de linkage."""
    z = np.asarray(z, dtype=float)
    d = dendrogram(z, no_plot=True, link_color_func=str)
    _, nodes = to_tree(z, rd=True)
    links = [
        Link(node_id=int(k), height=float(dc[1]), pos=tuple(float(v) for v in ic), heights=tuple(float(v) for v in dc),
             leaves=tuple(int(i) for i in nodes[int(k)].pre_order()))
        for ic, dc, k in zip(d["icoord"], d["dcoord"], d["color_list"])
    ]
    return [int(i) for i in d["leaves"]], links


def _densify(xs: Sequence[float], ys: Sequence[float], per_segment: int = POINTS_PER_SEGMENT) -> tuple[list[float], list[float]]:
    px: list[float] = []
    py: list[float] = []
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        px.extend((x0 + (x1 - x0) * t).tolist())
        py.extend((y0 + (y1 - y0) * t).tolist())
    px.append(float(xs[-1]))
    py.append(float(ys[-1]))
    return px, py


def build_dendrogram(z: np.ndarray, fund_ids: Sequence[str], labels: Mapping[str, str], cluster_labels: Sequence[int],
                     weights: Mapping[str, float], highlight: Iterable[str] = (), title: str | None = None) -> go.Figure:
    """Dendrograma horizontal (hojas en y, distancia en x) coloreado por cluster, con los fondos de ``highlight``
    (y los enlaces cuyo subárbol está íntegramente en ``highlight``) resaltados."""
    fund_ids = [str(f) for f in fund_ids]
    n = len(fund_ids)
    hi = set(highlight)
    leaves, links = dendrogram_geometry(z)
    pos = {leaf: LEAF_STEP * i + LEAF_STEP / 2 for i, leaf in enumerate(leaves)}
    palette = {c: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, c in enumerate(sorted(set(int(c) for c in cluster_labels)))}
    name = lambda f: labels.get(f, f)  # noqa: E731

    fig = go.Figure()
    for lk in links:
        ids = [fund_ids[i] for i in lk.leaves]
        clusters = {int(cluster_labels[i]) for i in lk.leaves}
        color = palette[next(iter(clusters))] if len(clusters) == 1 else MIXED_COLOR
        is_hi = bool(hi) and set(ids) <= hi
        leaf_pos, heights = _densify(lk.pos, lk.heights)
        shown = ", ".join(name(f) for f in ids[:6]) + (" …" if len(ids) > 6 else "")
        fig.add_trace(go.Scatter(
            x=heights, y=leaf_pos, mode="lines", name=f"link-{lk.node_id}", showlegend=False,
            line=dict(color=color, width=LINK_WIDTH_HI if is_hi else LINK_WIDTH),
            customdata=[ids] * len(heights),
            hovertemplate=f"<b>{len(ids)} fondos</b> · distancia {lk.height:.3f}<br>{shown}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[0.0] * n, y=[pos[i] for i in leaves], mode="markers", name="leaves", showlegend=False,
        marker=dict(size=[MARKER_SIZE_HI if fund_ids[i] in hi else MARKER_SIZE for i in leaves],
                    color=[palette[int(cluster_labels[i])] for i in leaves], line=dict(width=1, color="white")),
        customdata=[[fund_ids[i]] for i in leaves],
        text=[f"<b>{name(fund_ids[i])}</b><br>cluster {int(cluster_labels[i])} · peso {weights.get(fund_ids[i], 0.0):.2%}"
              for i in leaves],
        hovertemplate="%{text}<extra></extra>",
    ))

    max_h = max((lk.height for lk in links), default=1.0)
    fig.update_layout(
        title=title, height=max(420, int(28 * n + 120)), margin=dict(l=10, r=20, t=50 if title else 10, b=45),
        hovermode="closest", hoverdistance=25, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="distancia √((1 − ρ) / 2)", range=[-0.03 * max_h, 1.05 * max_h], zeroline=False,
                   gridcolor="#eeeeee"),
        yaxis=dict(tickvals=[pos[i] for i in leaves], ticktext=[fund_ids[i] for i in leaves],
                   range=[0, LEAF_STEP * n], showgrid=False, zeroline=False, tickfont=dict(size=11)),
    )
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k dendrogram`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/figures.py tests/test_dashboard.py
git commit -m "feat(dashboard): interactive dendrogram figure with subtree customdata

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Grid rows and columns (`dashboard/grid.py`)

**Files:**
- Create: `dashboard/grid.py`
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: `DistanceBundle`, `PortfolioBundle`, `PERIODS_PER_YEAR` (Task 2).
- Produces:
  - `column_defs(method: str) -> list[dict]` (`raw_weight` column hidden unless `method == "hrp"`)
  - `DEFAULT_COL_DEF: dict`
  - `risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray` (sums to 1)
  - `fund_metrics(window: pd.DataFrame) -> pd.DataFrame` (columns `ann_return, ann_vol, sharpe, max_drawdown`)
  - `build_rows(dist: DistanceBundle, port: PortfolioBundle, window: pd.DataFrame, labels: Mapping[str, str], categories: Mapping[str, str]) -> list[dict]` — rows in `leaf_order`, keys: `fund_id, nombre, cat, cluster, weight, raw_weight, risk_contrib, cluster_risk_contrib, ann_return, ann_vol, sharpe, max_drawdown` (floats or `None`, JSON-safe).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# grid.py
# ---------------------------------------------------------------------------
def test_risk_contributions_sum_to_one():
    from dashboard.grid import risk_contributions

    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]])
    w = np.array([0.5, 0.3, 0.2])
    rc = risk_contributions(w, cov)
    assert rc.sum() == pytest.approx(1.0)
    assert np.allclose(rc, w * (cov @ w) / (w @ cov @ w))
    assert np.allclose(risk_contributions(np.zeros(3), cov), 0.0)


def test_fund_metrics_columns(synthetic_returns):
    from dashboard.grid import fund_metrics

    m = fund_metrics(synthetic_returns.iloc[-252:])
    assert list(m.columns) == ["ann_return", "ann_vol", "sharpe", "max_drawdown"]
    assert (m["ann_vol"] > 0).all() and (m["max_drawdown"] <= 0).all()
    assert m.loc[FUND_IDS[0], "sharpe"] == pytest.approx(m.loc[FUND_IDS[0], "ann_return"] / m.loc[FUND_IDS[0], "ann_vol"])


def test_build_rows_ordered_by_leaf_order_and_json_safe(bundle, synthetic_returns):
    from dashboard.cache import _compute_portfolio
    from dashboard.data import window_slice
    from dashboard.grid import build_rows, column_defs

    win = window_slice(synthetic_returns, synthetic_returns.index[-1], 24)
    port = _compute_portfolio(win.frame, CATS, "hrp", "moderado")
    rows = build_rows(bundle, port, win.frame, {FUND_IDS[0]: "Nombre Largo"}, CATS)
    assert [r["fund_id"] for r in rows] == [bundle.fund_ids[i] for i in bundle.leaf_order]
    assert rows[[r["fund_id"] for r in rows].index(FUND_IDS[0])]["nombre"] == "Nombre Largo"
    assert sum(r["weight"] for r in rows) == pytest.approx(1.0)
    assert sum(r["risk_contrib"] for r in rows) == pytest.approx(1.0)
    by_cluster = {}
    for r in rows:
        by_cluster.setdefault(r["cluster"], 0.0)
        by_cluster[r["cluster"]] += r["risk_contrib"]
    assert all(r["cluster_risk_contrib"] == pytest.approx(by_cluster[r["cluster"]]) for r in rows)
    assert all(r["raw_weight"] is not None for r in rows)
    json.dumps(rows)
    assert all(r["cat"] == CATS[r["fund_id"]] for r in rows)

    port_mk = _compute_portfolio(win.frame, CATS, "markowitz", "moderado")
    rows_mk = build_rows(bundle, port_mk, win.frame, {}, CATS)
    assert all(r["raw_weight"] is None for r in rows_mk)


def test_column_defs_toggle_raw_weight():
    from dashboard.grid import column_defs

    hrp = {c["field"]: c for c in column_defs("hrp")}
    mk = {c["field"]: c for c in column_defs("markowitz")}
    assert hrp["raw_weight"].get("hide", False) is False and mk["raw_weight"]["hide"] is True
    assert set(hrp) >= {"fund_id", "cat", "cluster", "weight", "risk_contrib", "cluster_risk_contrib",
                        "ann_return", "ann_vol", "sharpe", "max_drawdown"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k "risk_contrib or fund_metrics or build_rows or column_defs"`
Expected: 4 failures, `ModuleNotFoundError: No module named 'dashboard.grid'`.

- [ ] **Step 3: Implement `dashboard/grid.py`**

```python
"""Filas y columnas del AG Grid (pesos, contribución al riesgo, métricas por fondo)."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from dashboard.cache import PERIODS_PER_YEAR, DistanceBundle, PortfolioBundle

# dash-ag-grid expone d3-format en las funciones de formato
_PCT = {"function": "params.value == null ? '' : d3.format('.2%')(params.value)"}
_NUM = {"function": "params.value == null ? '' : d3.format('.2f')(params.value)"}
_NUMERIC = {"type": "numericColumn", "width": 130}

DEFAULT_COL_DEF = {"sortable": True, "filter": True, "resizable": True}


def column_defs(method: str) -> list[dict]:
    """Definición de columnas; ``raw_weight`` (pesos HRP antes de la proyección QP) solo visible con HRP."""
    return [
        {"field": "fund_id", "headerName": "Fondo", "pinned": "left", "minWidth": 220, "tooltipField": "nombre"},
        {"field": "cat", "headerName": "Categoría", "width": 110},
        {"field": "cluster", "headerName": "Cluster", "width": 95, "type": "numericColumn"},
        {"field": "weight", "headerName": "Peso", "valueFormatter": _PCT, **_NUMERIC},
        {"field": "raw_weight", "headerName": "Peso HRP pre-QP", "valueFormatter": _PCT, "hide": method != "hrp", **_NUMERIC},
        {"field": "risk_contrib", "headerName": "Contrib. varianza", "valueFormatter": _PCT, **_NUMERIC},
        {"field": "cluster_risk_contrib", "headerName": "Contrib. cluster", "valueFormatter": _PCT, **_NUMERIC},
        {"field": "ann_return", "headerName": "Ret. anual", "valueFormatter": _PCT, **_NUMERIC},
        {"field": "ann_vol", "headerName": "Vol. anual", "valueFormatter": _PCT, **_NUMERIC},
        {"field": "sharpe", "headerName": "Sharpe", "valueFormatter": _NUM, **_NUMERIC},
        {"field": "max_drawdown", "headerName": "Max DD", "valueFormatter": _PCT, **_NUMERIC},
    ]


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """RC_i = w_i (Σ w)_i / (wᵀ Σ w); suma 1. Cero si la varianza es nula."""
    marginal = cov @ weights
    var = float(weights @ marginal)
    return weights * marginal / var if var > 0 else np.zeros_like(weights, dtype=float)


def fund_metrics(window: pd.DataFrame) -> pd.DataFrame:
    """Sobre la ventana (retornos log diarios): media·252, σ·√252, Sharpe (rf 0), max drawdown de exp(cumsum)."""
    mu = window.mean() * PERIODS_PER_YEAR
    vol = window.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR)
    nav = np.exp(window.cumsum())
    max_dd = (nav / nav.cummax() - 1.0).min()
    sharpe = (mu / vol).replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame({"ann_return": mu, "ann_vol": vol, "sharpe": sharpe, "max_drawdown": max_dd})


def _num(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def build_rows(dist: DistanceBundle, port: PortfolioBundle, window: pd.DataFrame, labels: Mapping[str, str],
               categories: Mapping[str, str]) -> list[dict]:
    """Una fila por fondo, en el orden de las hojas del dendrograma (alineación visual con el panel izquierdo)."""
    ids = list(dist.fund_ids)
    w = np.array([port.weights.get(f, 0.0) for f in ids], dtype=float)
    rc = risk_contributions(w, dist.cov)
    cl = np.asarray(dist.cluster_labels, dtype=int)
    cluster_rc = {int(c): float(rc[cl == c].sum()) for c in np.unique(cl)}
    met = fund_metrics(window[ids])
    rows = []
    for pos in dist.leaf_order:
        f = ids[pos]
        rows.append({
            "fund_id": f,
            "nombre": labels.get(f, f),
            "cat": categories[f],
            "cluster": int(cl[pos]),
            "weight": _num(w[pos]),
            "raw_weight": None if port.raw_weights is None else _num(port.raw_weights.get(f, 0.0)),
            "risk_contrib": _num(rc[pos]),
            "cluster_risk_contrib": _num(cluster_rc[int(cl[pos])]),
            "ann_return": _num(met.at[f, "ann_return"]),
            "ann_vol": _num(met.at[f, "ann_vol"]),
            "sharpe": _num(met.at[f, "sharpe"]),
            "max_drawdown": _num(met.at[f, "max_drawdown"]),
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: `16 passed`.

- [ ] **Step 5: Commit**

```bash
git add dashboard/grid.py tests/test_dashboard.py
git commit -m "feat(dashboard): AG Grid rows with variance risk contributions and fund metrics

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: App factory, layout and callbacks (`dashboard/app.py`)

**Files:**
- Create: `dashboard/app.py`, `dashboard/assets/dashboard.css`
- Test: `tests/test_dashboard.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces:
  - `DashboardData(returns, categories, labels, fingerprint)`
  - `load_data(returns=None, labels=None, categories=None) -> DashboardData`
  - `build_layout(data) -> dbc.Container` with component ids `profile, method, lookback, asof, result, pinned, dendro, grid, error, badges`
  - `register_callbacks(app, data)` registering `asof_choices`, `compute`, `render_grid`, `render_figure`, `sync`
  - `create_app(returns=None, labels=None, categories=None, cache_cfg=None) -> dash.Dash`
  - Store payload (`result.data`): `{"method","profile","window":{"start","end","n_obs"},"fund_ids","linkage","leaf_order","cluster_labels","weights","rows","badges"}`.
  - `sync` semantics: hover → `selectedRows={"ids": hovered ∪ pinned}`; hover-out (`hoverData=None`) → `{"ids": pinned}`; click → toggles the clicked ids in `pinned` (all already pinned → remove; otherwise add) and returns the new pinned as selection.

- [ ] **Step 1: Write the failing tests (smoke, callbacks via dispatch, cache through the app)**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# app.py — dispatch helpers (POST /_dash-update-component == real Dash callback path, no browser)
# ---------------------------------------------------------------------------
def _output_key(app, contains: str) -> str:
    return next(k for k in app.callback_map if contains in k)


def _outputs(key: str) -> list[dict]:
    return [dict(zip(("id", "property"), s.split("."))) for s in key.strip(".").split("...")]


def dispatch(app, contains: str, inputs: dict[str, object], state: dict[str, object] | None = None,
             changed: list[str] | None = None) -> dict:
    """inputs/state: {"id.prop": value}. Returns the {"id": {"prop": value}} response dict."""
    key = _output_key(app, contains)
    body = {
        "output": key,
        "outputs": _outputs(key),
        "inputs": [{"id": k.split(".")[0], "property": k.split(".")[1], "value": v} for k, v in inputs.items()],
        "state": [{"id": k.split(".")[0], "property": k.split(".")[1], "value": v} for k, v in (state or {}).items()],
        "changedPropIds": changed if changed is not None else list(inputs),
    }
    resp = app.server.test_client().post("/_dash-update-component", json=body)
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]
    return resp.get_json()["response"]


@pytest.fixture
def app(synthetic_returns):
    from dashboard.app import create_app

    return create_app(returns=synthetic_returns, labels={}, cache_cfg={"CACHE_TYPE": "SimpleCache"})


def _compute(app, profile="moderado", method="hrp", asof=None, lookback=24):
    asof = asof or str(app._omp_last_date)
    return dispatch(app, "result.data", {"profile.value": profile, "method.value": method,
                                         "asof.value": asof, "lookback.value": lookback}, changed=["profile.value"])


def test_layout_validates_and_ids_present(app):
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    assert {"profile", "method", "lookback", "asof", "result", "pinned", "dendro", "grid", "error", "badges"} <= ids
    app.validation_layout  # noqa: B018 — raises if the layout is invalid
    assert {"asof_choices", "compute", "render_grid", "render_figure", "sync"} <= {
        cb["callback"].__wrapped__.__name__ if hasattr(cb["callback"], "__wrapped__") else cb["callback"].__name__
        for cb in app.callback_map.values()}


@pytest.mark.parametrize("method", ["markowitz", "risk_parity", "hrp"])
@pytest.mark.parametrize("profile", ["conservador", "moderado", "agresivo"])
def test_smoke_nine_combinations(app, method, profile):
    out = _compute(app, profile=profile, method=method)
    payload = out["result"]["data"]
    assert out["error"]["is_open"] is False
    assert payload["method"] == method and payload["profile"] == profile
    assert len(payload["rows"]) == len(payload["fund_ids"]) == 21
    assert len(payload["linkage"]) == 20 and len(payload["cluster_labels"]) == 21
    assert abs(sum(payload["weights"].values()) - 1) < 1e-6
    assert (payload["rows"][0]["raw_weight"] is not None) == (method == "hrp")


def test_compute_reports_errors_without_crashing(app):
    out = dispatch(app, "result.data", {"profile.value": "moderado", "method.value": "hrp",
                                        "asof.value": str(app._omp_last_date), "lookback.value": 480},
                   changed=["lookback.value"])
    assert out["error"]["is_open"] is True and "BacktestError" in out["error"]["children"]
    assert "result" not in out                                         # no_update


def test_profile_and_method_changes_hit_distance_cache(app, monkeypatch):
    import dashboard.cache as dc

    calls = {"n": 0}
    real = dc._compute_distance

    def counting(window):
        calls["n"] += 1
        return real(window)

    monkeypatch.setattr(dc, "_compute_distance", counting)
    for profile in ("conservador", "moderado", "agresivo"):
        for method in ("markowitz", "risk_parity", "hrp"):
            _compute(app, profile=profile, method=method)
    assert calls["n"] == 1
    _compute(app, lookback=12)
    assert calls["n"] == 2
    _compute(app, profile="agresivo", lookback=24)
    assert calls["n"] == 2


def test_profile_change_changes_rows_and_method_toggles_raw(app):
    a = _compute(app, profile="conservador", method="risk_parity")["result"]["data"]
    b = _compute(app, profile="agresivo", method="risk_parity")["result"]["data"]
    assert a["weights"] != b["weights"]
    assert a["badges"]["rv"] <= 0.20 + 1e-6 and b["badges"]["rf"] >= 0.20 - 1e-6
    grid_a = dispatch(app, "grid.rowData", {"result.data": a})
    grid_h = dispatch(app, "grid.rowData", {"result.data": _compute(app, method="hrp")["result"]["data"]})
    cols_a = {c["field"]: c for c in grid_a["grid"]["columnDefs"]}
    cols_h = {c["field"]: c for c in grid_h["grid"]["columnDefs"]}
    assert cols_a["raw_weight"]["hide"] is True and cols_h["raw_weight"]["hide"] is False
    assert [r["fund_id"] for r in grid_a["grid"]["rowData"]] == [a["fund_ids"][i] for i in a["leaf_order"]]
    assert grid_a["badges"]["children"]


def test_render_figure_uses_store_and_pins(app):
    from dashboard.figures import LINK_WIDTH_HI

    payload = _compute(app)["result"]["data"]
    fig = dispatch(app, "dendro.figure", {"result.data": payload, "pinned.data": []})["dendro"]["figure"]
    assert len(fig["data"]) == 21 and fig["data"][-1]["name"] == "leaves"
    two = [tr["customdata"][0] for tr in fig["data"][:-1] if len(tr["customdata"][0]) == 2][0]
    fig2 = dispatch(app, "dendro.figure", {"result.data": payload, "pinned.data": two})["dendro"]["figure"]
    assert any(tr["line"]["width"] == LINK_WIDTH_HI for tr in fig2["data"][:-1])
    empty = dispatch(app, "dendro.figure", {"result.data": None, "pinned.data": []})["dendro"]["figure"]
    assert empty["data"] == [] or "layout" in empty


def _hover(ids):
    return {"points": [{"customdata": list(ids)}]}


def test_sync_hover_leaf_selects_row(app):
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": _hover([FUND_IDS[3]]), "dendro.clickData": None},
                   state={"pinned.data": []}, changed=["dendro.hoverData"])
    assert out["grid"]["selectedRows"] == {"ids": [FUND_IDS[3]]}
    assert out["grid"]["scrollTo"]["rowId"] == FUND_IDS[3]
    assert "pinned" not in out                                          # hover never rewrites pins


def test_sync_hover_link_selects_subtree(app):
    from dashboard.figures import dendrogram_geometry

    payload = _compute(app)["result"]["data"]
    _, links = dendrogram_geometry(np.array(payload["linkage"]))
    link = max((l for l in links if len(l.leaves) < 21), key=lambda l: len(l.leaves))
    subtree = [payload["fund_ids"][i] for i in link.leaves]
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": _hover(subtree), "dendro.clickData": None},
                   state={"pinned.data": []}, changed=["dendro.hoverData"])
    assert sorted(out["grid"]["selectedRows"]["ids"]) == sorted(subtree)


def test_sync_hover_out_clears_unless_pinned(app):
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": None, "dendro.clickData": None},
                   state={"pinned.data": []}, changed=["dendro.hoverData"])
    assert out["grid"]["selectedRows"] == {"ids": []}
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": None, "dendro.clickData": None},
                   state={"pinned.data": [FUND_IDS[0]]}, changed=["dendro.hoverData"])
    assert out["grid"]["selectedRows"] == {"ids": [FUND_IDS[0]]}
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": _hover([FUND_IDS[5]]), "dendro.clickData": None},
                   state={"pinned.data": [FUND_IDS[0]]}, changed=["dendro.hoverData"])
    assert sorted(out["grid"]["selectedRows"]["ids"]) == sorted([FUND_IDS[0], FUND_IDS[5]])


def test_sync_click_toggles_pin(app):
    click = _hover([FUND_IDS[1], FUND_IDS[2]])
    out = dispatch(app, "grid.selectedRows", {"dendro.hoverData": click, "dendro.clickData": click},
                   state={"pinned.data": [FUND_IDS[0]]}, changed=["dendro.clickData"])
    assert sorted(out["pinned"]["data"]) == sorted([FUND_IDS[0], FUND_IDS[1], FUND_IDS[2]])
    assert sorted(out["grid"]["selectedRows"]["ids"]) == sorted(out["pinned"]["data"])
    out2 = dispatch(app, "grid.selectedRows", {"dendro.hoverData": click, "dendro.clickData": click},
                    state={"pinned.data": out["pinned"]["data"]}, changed=["dendro.clickData"])
    assert out2["pinned"]["data"] == [FUND_IDS[0]]                     # second click unpins the pair
    assert out2["grid"]["selectedRows"] == {"ids": [FUND_IDS[0]]}


def test_asof_choices_follow_lookback(app):
    out = dispatch(app, "asof.options", {"lookback.value": 36}, state={"asof.value": "1999-01-01"})
    opts = [o["value"] for o in out["asof"]["options"]]
    assert opts[-1] == str(app._omp_last_date) and out["asof"]["value"] == opts[-1]
    keep = dispatch(app, "asof.options", {"lookback.value": 24}, state={"asof.value": opts[-1]})
    assert keep["asof"]["value"] == opts[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k "app or smoke or sync or compute or render or asof_choices or layout"`
Expected: failures with `ModuleNotFoundError: No module named 'dashboard.app'`.

- [ ] **Step 3: Implement `dashboard/assets/dashboard.css`**

```css
/* Resaltado de filas seleccionadas desde el dendrograma (AG Grid 35 Theming API vars) */
.ag-root-wrapper {
  --ag-selected-row-background-color: rgba(255, 176, 0, 0.45);
  --ag-row-hover-color: rgba(0, 0, 0, 0.03);
}
#dendro .nsewdrag { cursor: crosshair; }
.omp-badge { font-size: 0.85rem; }
```

- [ ] **Step 4: Implement `dashboard/app.py`**

```python
"""Dashboard Dash (Fase 4): dendrograma HRP interactivo ↔ AG Grid de pesos y riesgo.

Arquitectura: ``create_app`` es una factoría. Los datos inmutables (retornos, categorías, etiquetas,
fingerprint) viajan en un ``DashboardData`` capturado por clausura en ``register_callbacks``; todo estado por
cliente vive en ``dcc.Store`` (``result``, ``pinned``). Los cálculos pesados pasan por ``dashboard.cache``.

Ejecutar: ``.venv/bin/python -m dashboard.app`` → http://127.0.0.1:8050
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from backtest.walk_forward import BacktestError
from dashboard import data as dd
from dashboard.cache import cache, cache_config, distance_bundle, portfolio_bundle
from dashboard.figures import build_dendrogram
from dashboard.grid import DEFAULT_COL_DEF, build_rows, column_defs
from optimization.portfolio_optimizer import METHODS, PROFILES, OptimizationError, load_categories

log = logging.getLogger(__name__)

METHOD_LABELS = {"markowitz": "Markowitz (máx. Sharpe)", "risk_parity": "Risk Parity", "hrp": "HRP"}
PROFILE_LABELS = {"conservador": "Conservador", "moderado": "Moderado", "agresivo": "Agresivo"}
# AG Grid ≥ 32.2: selección por objeto; sin checkboxes y sin selección por click (la selección la manda el dendrograma)
GRID_SELECTION = {"mode": "multiRow", "checkboxes": False, "headerCheckbox": False, "enableClickSelection": False}


@dataclass(frozen=True)
class DashboardData:
    returns: pd.DataFrame
    categories: Mapping[str, str]
    labels: Mapping[str, str]
    fingerprint: str


def load_data(returns: pd.DataFrame | None = None, labels: Mapping[str, str] | None = None,
              categories: Mapping[str, str] | None = None) -> DashboardData:
    """Datos reales de ``data/cleaned`` salvo que se inyecten (tests)."""
    from_disk = returns is None
    returns = dd.load_returns() if from_disk else returns
    return DashboardData(
        returns=returns,
        categories=categories if categories is not None else load_categories(),
        labels=labels if labels is not None else dd.load_labels(),
        fingerprint=dd.fingerprint(returns, dd.RETURNS_PATH if from_disk else None),
    )


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def build_layout(data: DashboardData) -> dbc.Container:
    last = str(data.returns.index[-1].date())
    dropdown = lambda id_, options, value: dcc.Dropdown(id=id_, options=options, value=value, clearable=False)  # noqa: E731
    return dbc.Container(fluid=True, className="py-3", children=[
        dcc.Store(id="result"),
        dcc.Store(id="pinned", data=[]),
        html.H4("Optimizador Multi-Perfil — FICs Colombia", className="mb-3"),
        dbc.Row(className="g-2 mb-3", children=[
            dbc.Col([dbc.Label("Perfil"), dropdown("profile", [{"label": PROFILE_LABELS[p], "value": p} for p in PROFILES], "moderado")], md=3),
            dbc.Col([dbc.Label("Metodología"), dropdown("method", [{"label": METHOD_LABELS[m], "value": m} for m in METHODS], "hrp")], md=3),
            dbc.Col([dbc.Label("Lookback (meses)"), dropdown("lookback", [{"label": f"{m} m", "value": m} for m in dd.LOOKBACK_OPTIONS], dd.DEFAULT_LOOKBACK)], md=2),
            dbc.Col([dbc.Label("Fecha as-of"), dropdown("asof", [{"label": last, "value": last}], last)], md=3),
        ]),
        dbc.Alert(id="error", color="danger", is_open=False, dismissable=True),
        dbc.Row(className="g-3", children=[
            dbc.Col(dcc.Graph(id="dendro", clear_on_unhover=True, config={"displayModeBar": False}), md=5),
            dbc.Col([
                html.Div(id="badges", className="mb-2 d-flex flex-wrap gap-2"),
                dag.AgGrid(
                    id="grid", rowData=[], columnDefs=column_defs("hrp"), defaultColDef=DEFAULT_COL_DEF,
                    getRowId="params.data.fund_id", columnSize="autoSize",
                    dashGridOptions={"rowSelection": GRID_SELECTION, "animateRows": True, "tooltipShowDelay": 300},
                    style={"height": "70vh", "width": "100%"},
                ),
            ], md=7),
        ]),
        html.Small("Hover sobre una hoja resalta su fila; sobre un enlace, todo el subárbol. Click fija/libera la selección.",
                   className="text-muted"),
    ])


def _ids_from_event(event: dict | None) -> list[str]:
    """fund_ids de un hoverData/clickData: cada punto lleva ``customdata = [fund_id, ...]`` (ver figures.py)."""
    out: list[str] = []
    for p in (event or {}).get("points", []):
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            out.extend(str(x) for x in cd)
        elif cd is not None:
            out.append(str(cd))
    return list(dict.fromkeys(out))


def _badges(payload: dict) -> list:
    b = payload["badges"]
    w = payload["window"]
    items = [
        ("RV", _fmt_pct(b["rv"]), "primary"), ("RF", _fmt_pct(b["rf"]), "secondary"), ("RV intl", _fmt_pct(b["rv_intl"]), "info"),
        ("Vol", _fmt_pct(b["volatility"]), "dark"), ("Ret. esp.", _fmt_pct(b["expected_return"]), "dark"),
        ("Sharpe", "—" if b["sharpe"] is None else f"{b['sharpe']:.2f}", "dark"),
        ("Fondos", str(b["n_funds"]), "light"), ("Ventana", f"{w['start']} → {w['end']} ({w['n_obs']} d)", "light"),
    ]
    return [dbc.Badge(f"{k}: {v}", color=c, className="omp-badge", text_color="dark" if c == "light" else None)
            for k, v, c in items]


def register_callbacks(app: Dash, data: DashboardData) -> None:
    @app.callback(Output("asof", "options"), Output("asof", "value"), Input("lookback", "value"), State("asof", "value"))
    def asof_choices(lookback, current):
        opts = [str(d.date()) for d in dd.asof_options(data.returns.index, int(lookback))]
        return [{"label": o, "value": o} for o in opts], current if current in opts else opts[-1]

    @app.callback(Output("result", "data"), Output("error", "children"), Output("error", "is_open"),
                  Input("profile", "value"), Input("method", "value"), Input("asof", "value"), Input("lookback", "value"))
    def compute(profile, method, asof, lookback):
        try:
            win = dd.window_slice(data.returns, pd.Timestamp(asof), int(lookback))
            dist = distance_bundle(win.frame, win.universe, win.start, win.end, data.fingerprint)
            port = portfolio_bundle(win.frame, data.categories, win.universe, win.start, win.end, data.fingerprint, method, profile)
        except (BacktestError, OptimizationError, ValueError) as exc:
            log.warning("compute(%s, %s, %s, %s) failed: %s", profile, method, asof, lookback, exc)
            return no_update, f"{type(exc).__name__}: {exc}", True
        payload = {
            "method": method, "profile": profile,
            "window": {"start": win.start, "end": win.end, "n_obs": int(len(win.frame))},
            "fund_ids": list(dist.fund_ids), "linkage": dist.linkage.tolist(),
            "leaf_order": dist.leaf_order, "cluster_labels": dist.cluster_labels,
            "weights": port.weights,
            "rows": build_rows(dist, port, win.frame, data.labels, data.categories),
            "badges": {"rv": port.exposures["rv"], "rf": port.exposures["rf"], "rv_intl": port.exposures["rv_intl"],
                       "volatility": _num(port.volatility), "expected_return": _num(port.expected_return),
                       "sharpe": _num(port.sharpe), "n_funds": len(dist.fund_ids)},
        }
        return payload, "", False

    @app.callback(Output("grid", "rowData"), Output("grid", "columnDefs"), Output("badges", "children"), Input("result", "data"))
    def render_grid(result):
        if not result:
            return [], column_defs("hrp"), []
        return result["rows"], column_defs(result["method"]), _badges(result)

    @app.callback(Output("dendro", "figure"), Input("result", "data"), Input("pinned", "data"))
    def render_figure(result, pinned):
        if not result:
            return go.Figure(layout=dict(height=420, xaxis=dict(visible=False), yaxis=dict(visible=False),
                                         annotations=[dict(text="Sin resultado", showarrow=False)]))
        title = f"{METHOD_LABELS[result['method']]} · {PROFILE_LABELS[result['profile']]} · jerarquía HRP (single linkage)"
        return build_dendrogram(np.asarray(result["linkage"]), result["fund_ids"], data.labels, result["cluster_labels"],
                                result["weights"], highlight=frozenset(pinned or []), title=title)

    @app.callback(Output("grid", "selectedRows"), Output("grid", "scrollTo"), Output("pinned", "data"),
                  Input("dendro", "hoverData"), Input("dendro", "clickData"), State("pinned", "data"))
    def sync(hover, click, pinned):
        """Hover → resalta (hojas hovered ∪ fijadas). Click → alterna el pin del subárbol/hoja clicada."""
        pinned = [str(p) for p in (pinned or [])]
        if "dendro.clickData" in ctx.triggered_prop_ids and click:
            ids = _ids_from_event(click)
            if set(ids) <= set(pinned):
                pinned = [p for p in pinned if p not in ids]
            else:
                pinned = list(dict.fromkeys(pinned + ids))
            return {"ids": pinned}, _scroll(pinned), pinned
        shown = list(dict.fromkeys(pinned + _ids_from_event(hover)))
        return {"ids": shown}, _scroll(shown), no_update


def _scroll(ids: list[str]):
    return {"rowId": ids[0], "rowPosition": "middle"} if ids else no_update


def create_app(returns: pd.DataFrame | None = None, labels: Mapping[str, str] | None = None,
               categories: Mapping[str, str] | None = None, cache_cfg: dict | None = None) -> Dash:
    data = load_data(returns, labels, categories)
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Optimizador Multi-Perfil")
    cache.init_app(app.server, cache_cfg or cache_config())
    app.layout = build_layout(data)
    register_callbacks(app, data)
    app._omp_last_date = data.returns.index[-1].date()   # convenience for tests / __main__ logging only
    return app


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    create_app().run(debug=False, host="127.0.0.1", port=8050)
```

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q`
Expected: all pass (`16 + 17 = 33 passed`; smoke ×9 dominates runtime, ~30–60 s).

If `test_layout_validates_and_ids_present` fails on the callback-name assertion because Dash wraps callbacks differently, replace the assertion with `assert len(app.callback_map) == 5` — the other tests already exercise every callback by output.

If `test_render_figure_uses_store_and_pins` fails because Dash serialises the figure with `plotly.io.to_json` and `customdata` arrives as nested lists, that is expected and the test already indexes `tr["customdata"][0]`; only adjust if the key path differs.

- [ ] **Step 6: Manual browser check (required — this is the core requirement)**

Run in the background: `.venv/bin/python -m dashboard.app`
Open http://127.0.0.1:8050 (Browser pane or Chrome). Verify, with a screenshot each:
1. Dendrogram left (horizontal, coloured by cluster), grid right, rows in leaf order.
2. Hover a leaf marker → exactly that row gets the amber background and the grid scrolls to it.
3. Hover a U-link → all funds of its subtree highlighted.
4. Click a link → highlight stays after moving the mouse away; click again → cleared.
5. Switch profile → grid weights change, no page reload; switch to Markowitz → `Peso HRP pre-QP` column disappears; dendrogram unchanged (method-independent).
6. Server log shows the first `compute` computing and subsequent switches returning in < 100 ms (cache hit) — add a temporary `log.info` around `distance_bundle` if needed and remove it afterwards.

If the highlight colour is not visible, inspect the `.ag-row-selected` element: AG Grid 35 legacy CSS themes are not loaded, so the Theming API variable in `assets/dashboard.css` must be on `.ag-root-wrapper` (already). Stop the server afterwards.

- [ ] **Step 7: Commit**

```bash
git add dashboard/app.py dashboard/assets/dashboard.css tests/test_dashboard.py
git commit -m "feat(dashboard): Dash app with dendrogram ↔ AG Grid hover sync and cached recompute

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: E2E (skippable) test, README, coverage gate

**Files:**
- Test: `tests/test_dashboard.py` (append)
- Create: `dashboard/README.md`

- [ ] **Step 1: Add the selenium E2E test (skips without chromedriver)**

Append to `tests/test_dashboard.py`:

```python
# ---------------------------------------------------------------------------
# E2E — real browser hover (dash[testing] pins selenium<=4.2.0: chromedriver must be on PATH)
# ---------------------------------------------------------------------------
import shutil  # noqa: E402

pytestmark_e2e = pytest.mark.skipif(shutil.which("chromedriver") is None,
                                    reason="chromedriver no está en PATH (brew install chromedriver)")


@pytestmark_e2e
def test_e2e_hover_leaf_highlights_grid_row(dash_duo, synthetic_returns):
    from selenium.webdriver.common.action_chains import ActionChains

    from dashboard.app import create_app

    app = create_app(returns=synthetic_returns, labels={}, cache_cfg={"CACHE_TYPE": "SimpleCache"})
    dash_duo.start_server(app)
    dash_duo.wait_for_element(".ag-row", timeout=60)
    markers = dash_duo.find_elements("#dendro .scatterlayer .trace:last-child .points path")
    assert len(markers) == 21
    ActionChains(dash_duo.driver).move_to_element(markers[0]).perform()
    dash_duo.wait_for_element(".ag-row-selected", timeout=10)
    assert len(dash_duo.find_elements(".ag-row-selected")) == 1
    assert dash_duo.get_logs() == []
```

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q -k e2e -rs`
Expected: `1 skipped` (reason printed) on this machine; if chromedriver is present, `1 passed`.

- [ ] **Step 2: Coverage gate**

Run: `.venv/bin/python -m pytest tests/test_dashboard.py -q --cov=dashboard --cov-report=term-missing --cov-fail-under=85`
Expected: pass with ≥ 85 % (the `__main__` block is `pragma: no cover`). If below, add a test for the uncovered branch (typically `load_labels` with a real parquet, or `render_figure` empty state) rather than lowering the gate.

- [ ] **Step 3: Write `dashboard/README.md`**

```markdown
# `dashboard/` — dendrograma HRP interactivo ↔ AG Grid (Fase 4)

| Ruta | Qué es |
|---|---|
| `app.py` | `create_app()` (factoría Dash), layout dbc, callbacks `asof_choices` / `compute` / `render_grid` / `render_figure` / `sync` |
| `data.py` | carga de `data/cleaned/*`, `fingerprint`, fechas as-of, `window_slice` (reusa `active_universe` / `estimation_window` del backtest) |
| `cache.py` | Flask-Caching: `distance_bundle` (corr → distancia → linkage → hojas → clusters → Σ LW) y `portfolio_bundle` (`optimize`) |
| `figures.py` | dendrograma horizontal desde la matriz de linkage; cada enlace lleva `customdata` = fondos de su subárbol |
| `grid.py` | filas del AG Grid: peso, peso HRP pre-QP, contribución a varianza (fondo y cluster), métricas por fondo |
| `assets/dashboard.css` | color de fila seleccionada (AG Grid Theming API) |

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m dashboard.app                 # http://127.0.0.1:8050
.venv/bin/python -m pytest tests/test_dashboard.py --cov=dashboard --cov-fail-under=85
```

**Interacción.** Hover sobre una hoja resalta su fila; sobre un enlace (U) resalta todo el subárbol; click
fija/libera la selección (persiste al retirar el cursor). Cambiar perfil/metodología/ventana recalcula sin
recargar. El dendrograma no depende de la metodología (solo de la matriz de distancia), así que se muestra
también con Markowitz y Risk Parity; con HRP el grid añade la columna de pesos antes de la proyección QP (D5).

**Cache.** `distance_bundle` se memoiza con clave `(universo, inicio, fin, fingerprint)` — perfil y metodología
no están en la clave, por lo que cambiar de perfil nunca recalcula la matriz de distancia
(`tests/test_dashboard.py::test_profile_and_method_changes_hit_distance_cache`). `fingerprint` = sha256 del
parquet: un re-ingest invalida la cache sin reiniciar. TTL `DIST_TTL_SECONDS = 24 h` (cadencia diaria SFC;
solo acota memoria, ver comentario en `cache.py`). Backend `SimpleCache` por defecto; para varios workers:
`DASH_CACHE_TYPE=FileSystemCache DASH_CACHE_DIR=.cache/dashboard`.

**Tests.** Los callbacks se prueban por `POST /_dash-update-component` con el test client de Flask (ruta real de
Dash, sin navegador). El test E2E con `dash_duo` se omite si no hay `chromedriver` en PATH (`dash[testing]`
pinea `selenium<=4.2.0`, sin Selenium Manager): `brew install chromedriver` para ejecutarlo.

Fuera de alcance (decisión de Fase 4): exportación PDF / tear sheet, estado en URL, sincronización grid → dendrograma.
```

- [ ] **Step 4: Full suite + commit**

Run: `.venv/bin/python -m pytest -q` (whole repo — ingest/optimizer/backtest suites must still pass)
Expected: all green (dashboard: 33 passed, 1 skipped).

```bash
git add tests/test_dashboard.py dashboard/README.md
git commit -m "test(dashboard): selenium E2E (skip w/o chromedriver), README, coverage gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** §2 stack → Task 1 pins; §3 modules → Tasks 1–5; §4 data/window → Task 1; §5 cache (key, TTL comment, fingerprint, env backend, instrumentation, linkage equality) → Task 2 + Task 5 cache-through-app test; §6 layout/callbacks/stores → Task 5 (`render` split into `render_grid` + `render_figure` so pin clicks do not rewrite `rowData`; `sync` no longer needs `result` as State because `customdata` already carries fund ids — both refinements, same behaviour); §7 dendrogram (scipy node-id trick, densified links, cluster colours, horizontal, highlight) → Task 3; §8 grid columns and RC formula → Task 4; §9 tests (smoke ×9, cache count, dispatch callbacks, E2E skip, coverage 85) → Tasks 2, 5, 6; §10 run commands → README Task 6. Out-of-scope items untouched.
- **Placeholders:** none; every step has code or an exact command with expected output.
- **Type consistency:** `Window.universe: tuple[str, ...]`, `distance_bundle(window, universe, start, end, fingerprint)`, `portfolio_bundle(window, categories, universe, start, end, fingerprint, method, profile)`, `build_rows(dist, port, window, labels, categories)`, `build_dendrogram(z, fund_ids, labels, cluster_labels, weights, highlight, title)`, `column_defs(method)` used identically in Tasks 2–6. `Link.pos`/`Link.heights` names match between `dendrogram_geometry`, `_densify` call and the geometry test.
