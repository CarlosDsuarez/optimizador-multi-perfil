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
        except (BacktestError, OptimizationError, ValueError, KeyError) as exc:
            log.warning("compute(%s, %s, %s, %s) failed: %s", profile, method, asof, lookback, exc)
            return no_update, f"{type(exc).__name__}: {exc}", True
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
