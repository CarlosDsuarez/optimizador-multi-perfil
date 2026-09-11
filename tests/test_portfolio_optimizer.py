"""Tests for optimization/portfolio_optimizer.py — synthetic returns with analytic solutions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from optimization import portfolio_optimizer as po

REPO = Path(__file__).resolve().parents[1]

# Bands wide open: isolates the optimizer maths from the profile bands.
FREE = po.ProfileConstraints(max_rv=1.0, min_rf=0.0, max_rv_intl=1.0, max_weight=1.0)
CATS_10 = {
    "RFC1": "RF_CORTO", "RFC2": "RF_CORTO",
    "RFL1": "RF_LARGO", "RFL2": "RF_LARGO",
    "RVL1": "RV_LOCAL", "RVL2": "RV_LOCAL",
    "RVI1": "RV_INTL", "RVI2": "RV_INTL",
    "MIX1": "MIXTO", "MIX2": "MIXTO",
}
TOL = 1e-6


# ---------------------------------------------------------------------------
# Synthetic return builders with *exact* sample moments
# ---------------------------------------------------------------------------
def exact_returns(mu_daily, sigma_daily, corr=None, T=1500, seed=0, columns=None) -> pd.DataFrame:
    """Returns whose sample mean/std/corr equal the requested values exactly (QR orthogonalisation)."""
    n = len(mu_daily)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, n))
    x -= x.mean(axis=0)
    q, _ = np.linalg.qr(x)                      # orthonormal columns, mean ~0
    q -= q.mean(axis=0)
    q /= q.std(axis=0, ddof=1)                  # unit sample std, zero sample corr
    corr = np.eye(n) if corr is None else np.asarray(corr)
    z = q @ np.linalg.cholesky(corr).T
    r = z * np.asarray(sigma_daily) + np.asarray(mu_daily)
    cols = columns or [f"A{i}" for i in range(n)]
    idx = pd.bdate_range("2018-01-01", periods=T)
    return pd.DataFrame(r, index=idx, columns=cols)


def ten_asset_returns(seed=1) -> pd.DataFrame:
    sig = np.array([0.0003, 0.0003, 0.001, 0.001, 0.012, 0.012, 0.010, 0.010, 0.004, 0.004])
    mu = np.array([0.00025, 0.00025, 0.00028, 0.00028, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003])
    corr = np.full((10, 10), 0.1)
    np.fill_diagonal(corr, 1.0)
    corr[4, 5] = corr[5, 4] = 0.9
    corr[0, 1] = corr[1, 0] = 0.95
    return exact_returns(mu, sig, corr, T=1200, seed=seed, columns=list(CATS_10))


# ---------------------------------------------------------------------------
# Analytic cases
# ---------------------------------------------------------------------------
def test_risk_parity_two_uncorrelated_assets_gives_inverse_vol_weights():
    r = exact_returns([0.0004, 0.0004], [0.01, 0.03], columns=["A", "B"])
    res = po.optimize(r, "risk_parity", "moderado", FREE, categories={"A": "RF_CORTO", "B": "RF_CORTO"},
                      cov_method="hist")
    # ERC on uncorrelated assets: w_i ∝ 1/σ_i  → 0.75 / 0.25
    assert res.weights["A"] == pytest.approx(0.75, abs=1e-3)
    assert res.weights["B"] == pytest.approx(0.25, abs=1e-3)
    assert res.weights.sum() == pytest.approx(1.0, abs=TOL)


def test_risk_parity_respects_per_fund_cap():
    r = exact_returns([0.0004, 0.0004], [0.01, 0.03], columns=["A", "B"])
    capped = po.ProfileConstraints(max_rv=1.0, min_rf=0.0, max_rv_intl=1.0, max_weight=0.6)
    res = po.optimize(r, "risk_parity", "moderado", capped, categories={"A": "RF_CORTO", "B": "RF_CORTO"},
                      cov_method="hist")
    assert res.weights["A"] == pytest.approx(0.6, abs=1e-4)
    assert res.weights["B"] == pytest.approx(0.4, abs=1e-4)


def test_markowitz_two_uncorrelated_assets_gives_mu_over_variance_weights():
    ppy = 252
    mu_a, var_a = np.array([0.10, 0.05]), np.array([0.04, 0.01])
    r = exact_returns(mu_a / ppy, np.sqrt(var_a / ppy), columns=["A", "B"])
    res = po.optimize(r, "markowitz", "moderado", FREE, categories={"A": "RF_CORTO", "B": "RF_CORTO"},
                      cov_method="hist", rf=0.0)
    analytic = (mu_a / var_a) / (mu_a / var_a).sum()        # tangency portfolio, rf=0, Σ diagonal
    assert res.weights["A"] == pytest.approx(analytic[0], abs=1e-3)
    assert res.weights["B"] == pytest.approx(analytic[1], abs=1e-3)
    assert res.expected_return == pytest.approx(float(res.weights @ mu_a), abs=1e-3)
    assert res.volatility == pytest.approx(float(np.sqrt(res.weights**2 @ var_a)), abs=1e-3)
    assert res.sharpe == pytest.approx(res.expected_return / res.volatility, abs=1e-6)


def test_hrp_linkage_is_deterministic_and_dendrogram_data_is_kept():
    r = ten_asset_returns()
    a = po.optimize(r, "hrp", "agresivo", categories=CATS_10)
    b = po.optimize(r, "hrp", "agresivo", categories=CATS_10)
    n = r.shape[1]
    assert a.linkage_matrix.shape == (n - 1, 4)
    np.testing.assert_array_equal(a.linkage_matrix, b.linkage_matrix)
    assert sorted(a.leaf_order) == list(range(n))
    assert len(a.cluster_labels) == n and a.cluster_labels.min() >= 1
    assert a.raw_weights.sum() == pytest.approx(1.0, abs=TOL)
    pd.testing.assert_series_equal(a.weights, b.weights)
    # Highly correlated pair (RVL1, RVL2) must merge before anything joins them with RFC1
    from scipy.cluster.hierarchy import fcluster
    two = fcluster(a.linkage_matrix, 2, criterion="maxclust")
    lab = dict(zip(r.columns, two))
    assert lab["RVL1"] == lab["RVL2"]


def test_hrp_projection_moves_raw_weights_into_bands_and_keeps_raw():
    r = ten_asset_returns()
    res = po.optimize(r, "hrp", "conservador", categories=CATS_10)
    raw_exp = po.exposures(res.raw_weights, CATS_10, po.PROFILE_CONSTRAINTS["conservador"])
    # Raw HRP on this universe is not conservador-compliant (sanity: otherwise the test proves nothing)
    assert raw_exp["rv"] > 0.20 or res.raw_weights.max() > 0.25
    assert not np.allclose(res.raw_weights.values, res.weights.values)
    assert res.exposures["rv"] <= 0.20 + TOL


# ---------------------------------------------------------------------------
# Bands are hard constraints for every method × profile
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", po.METHODS)
@pytest.mark.parametrize("profile", po.PROFILES)
def test_bands_are_respected_strictly(method, profile):
    r = ten_asset_returns()
    c = po.PROFILE_CONSTRAINTS[profile]
    res = po.optimize(r, method, profile, categories=CATS_10)
    w = res.weights
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0, abs=TOL)
    assert (w >= -TOL).all()
    assert (w <= c.max_weight + TOL).all()
    exp = po.exposures(w, CATS_10, c)
    assert exp["rv"] <= c.max_rv + TOL
    assert exp["rf"] >= c.min_rf - TOL
    assert exp["rv_intl"] <= c.max_rv_intl + TOL
    assert res.exposures == pytest.approx(exp, abs=TOL)
    assert res.method == method and res.profile == profile
    assert res.volatility > 0 and np.isfinite(res.sharpe)


def test_exposures_apply_50pct_look_through_on_mixtos():
    w = pd.Series({"RFC1": 0.5, "RVL1": 0.2, "RVI1": 0.1, "MIX1": 0.2})
    cats = {k: CATS_10[k] for k in w.index}
    exp = po.exposures(w, cats, po.PROFILE_CONSTRAINTS["moderado"])
    assert exp["rv"] == pytest.approx(0.2 + 0.1 + 0.5 * 0.2)
    assert exp["rf"] == pytest.approx(0.5 + 0.5 * 0.2)
    assert exp["rv_intl"] == pytest.approx(0.1)


def test_profile_constraints_match_gate_table_d6():
    c, m, a = (po.PROFILE_CONSTRAINTS[p] for p in ("conservador", "moderado", "agresivo"))
    assert (c.max_rv, c.min_rf, c.max_rv_intl, c.max_weight) == (0.20, 0.70, 0.10, 0.25)
    assert (m.max_rv, m.min_rf, m.max_rv_intl, m.max_weight) == (0.45, 0.40, 0.20, 0.20)
    assert (a.max_rv, a.min_rf, a.max_rv_intl, a.max_weight) == (0.70, 0.20, 0.35, 0.20)
    assert c.min_weight == 0.0 and c.mixto_rv_share == 0.5


def test_load_categories_reads_repo_universe():
    cats = po.load_categories(REPO / "data" / "universe.json")
    assert len(cats) == 21
    assert set(cats.values()) <= po.CATEGORIES
    assert cats["ISHARES_COLCAP"] == "RV_LOCAL"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_nan_in_returns_is_rejected():
    r = ten_asset_returns()
    r.iloc[5, 2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        po.optimize(r, "hrp", "moderado", categories=CATS_10)


def test_single_asset_is_rejected():
    r = ten_asset_returns().iloc[:, :1]
    with pytest.raises(ValueError, match="2 columns"):
        po.optimize(r, "hrp", "moderado", categories=CATS_10)


def test_missing_or_unknown_category_is_rejected():
    r = ten_asset_returns()
    cats = dict(CATS_10)
    del cats["MIX2"]
    with pytest.raises(ValueError, match="MIX2"):
        po.optimize(r, "hrp", "moderado", categories=cats)
    cats["MIX2"] = "CRYPTO"
    with pytest.raises(ValueError, match="CRYPTO"):
        po.optimize(r, "hrp", "moderado", categories=cats)


def test_unknown_method_or_profile_is_rejected():
    r = ten_asset_returns()
    with pytest.raises(ValueError, match="method"):
        po.optimize(r, "black_litterman", "moderado", categories=CATS_10)
    with pytest.raises(ValueError, match="profile"):
        po.optimize(r, "hrp", "yolo", categories=CATS_10)


@pytest.mark.parametrize("method", po.METHODS)
def test_infeasible_bands_raise_before_solving(method):
    r = ten_asset_returns()
    tiny = po.ProfileConstraints(max_rv=0.2, min_rf=0.7, max_rv_intl=0.1, max_weight=0.05)  # 10 × 5% < 100%
    with pytest.raises(po.InfeasibleConstraintsError):
        po.optimize(r, method, "conservador", tiny, categories=CATS_10)


def test_markowitz_without_any_asset_beating_rf_raises_optimization_error():
    r = exact_returns([-0.001, -0.002], [0.01, 0.03], columns=["A", "B"])
    with pytest.raises(po.OptimizationError, match="risk-free"):
        po.optimize(r, "markowitz", "moderado", FREE, categories={"A": "RF_CORTO", "B": "RF_CORTO"})


def test_solver_failure_surfaces_as_optimization_error(monkeypatch):
    r = ten_asset_returns()

    def boom(*a, **k):
        raise RuntimeError("solver exploded")

    monkeypatch.setattr(po, "_solve_hrp", boom)
    with pytest.raises(po.OptimizationError, match="solver exploded"):
        po.optimize(r, "hrp", "moderado", categories=CATS_10)


def test_projection_returning_no_solution_is_an_optimization_error(monkeypatch):
    r = ten_asset_returns()
    monkeypatch.setattr(po, "_project_to_bands", lambda *a, **k: None)
    with pytest.raises(po.OptimizationError, match="projection"):
        po.optimize(r, "hrp", "moderado", categories=CATS_10)


# ---------------------------------------------------------------------------
# Optional RV floor (sensitivity; D6 base case keeps min_rv = 0)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", po.METHODS)
def test_min_rv_floor_is_respected_when_set(method):
    r = ten_asset_returns()
    base = po.PROFILE_CONSTRAINTS["agresivo"]
    assert base.min_rv == 0.0
    floored = po.ProfileConstraints(max_rv=base.max_rv, min_rf=base.min_rf, max_rv_intl=base.max_rv_intl,
                                    max_weight=base.max_weight, min_rv=0.45)
    res = po.optimize(r, method, "agresivo", floored, categories=CATS_10)
    assert res.exposures["rv"] >= 0.45 - TOL
    assert res.exposures["rv"] <= 0.70 + TOL
    free = po.optimize(r, method, "agresivo", categories=CATS_10)
    assert free.exposures["rv"] < 0.45          # sanity: the floor actually changes the answer here


def test_riskfolio_receives_percent_returns_without_pd_warning(capsys):
    # Daily variances ~1e-7 trip riskfolio's is_pos_def(threshold=1e-6) even though Σ is PD.
    r = exact_returns([0.0002] * 4, [0.0003, 0.0003, 0.0004, 0.0005], columns=list("ABCD"))
    cats = {c: "RF_CORTO" for c in r.columns}
    po.optimize(r, "risk_parity", "moderado", FREE, categories=cats)
    po.optimize(r, "hrp", "moderado", FREE, categories=cats)
    assert "positive definite" not in capsys.readouterr().out


@pytest.mark.skipif(not (REPO / "data" / "cleaned" / "returns_matrix.parquet").exists(), reason="needs Fase 1 outputs")
@pytest.mark.parametrize("profile", po.PROFILES)
def test_markowitz_bands_hold_to_1e8_on_a_real_window(profile):
    r = pd.read_parquet(REPO / "data" / "cleaned" / "returns_matrix.parquet")
    cats = po.load_categories(REPO / "data" / "universe.json")
    win = r.loc["2022-01-01":"2023-12-29"].dropna(axis=1, how="any")
    win = win.loc[:, win.notna().all()]
    c = po.PROFILE_CONSTRAINTS[profile]
    res = po.optimize(win, "markowitz", profile, categories=cats)
    a, b = po._linear_constraints(list(win.columns), cats, c)
    assert (a @ res.weights.to_numpy() - b).max() <= 1e-8
    assert abs(res.weights.sum() - 1) <= 1e-8
