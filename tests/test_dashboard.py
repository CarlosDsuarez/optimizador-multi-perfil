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
