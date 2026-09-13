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
