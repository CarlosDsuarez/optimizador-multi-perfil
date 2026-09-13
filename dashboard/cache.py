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
