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


# ---------------------------------------------------------------------------
# app.py — dispatch helpers (POST /_dash-update-component == real Dash callback path, no browser)
# ---------------------------------------------------------------------------
def _output_key(app, contains: str) -> str:
    return next(k for k in app.callback_map if contains in k)


def _outputs(key: str) -> list[dict] | dict:
    items = [dict(zip(("id", "property"), s.split("."))) for s in key.strip(".").split("...")]
    return items if key.startswith("..") else items[0]


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


def test_compute_guards_payload_construction(app, monkeypatch):
    """Una excepción al construir el payload (build_rows) debe abrir la alerta, no devolver 500."""
    import dashboard.app as da

    def boom(*args, **kwargs):
        raise ValueError("fila rota")

    monkeypatch.setattr(da, "build_rows", boom)
    out = _compute(app)
    assert out["error"]["is_open"] is True and "ValueError: fila rota" in out["error"]["children"]
    assert "result" not in out
