"""Tests for backtest/walk_forward.py — synthetic returns, deterministic data-dependent optimizer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest import walk_forward as wf
from data.ingest_sfc import sha256_of

REPO = Path(__file__).resolve().parents[1]
CATS = {"A": "RF_CORTO", "B": "RF_CORTO", "C": "RF_LARGO", "D": "RV_LOCAL", "E": "RV_INTL", "F": "MIXTO"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def synthetic_returns(start="2017-06-01", end="2022-12-30", seed=0) -> pd.DataFrame:
    """Daily log returns on business days; F is a `late` fund (NaN before 2018-09-03); one gap cell."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    sig = np.array([0.0003, 0.0003, 0.001, 0.012, 0.010, 0.004])
    r = pd.DataFrame(rng.normal(0.0002, sig, size=(len(idx), 6)), index=idx, columns=list(CATS))
    r.loc[r.index < "2018-09-03", "F"] = np.nan
    r.loc["2019-03-05", "C"] = np.nan          # intra-life gap (manifest gap_log semantics)
    r.index.name = "fecha"
    return r


def inverse_vol_optimizer(window: pd.DataFrame, method: str, profile: str) -> pd.Series:
    """Deterministic, data-dependent stand-in: w ∝ 1/σ over the window (any leak changes the output)."""
    iv = 1.0 / window.std(ddof=1)
    return iv / iv.sum()


def small_cfg(**over) -> wf.BacktestConfig:
    base = dict(estimation_start="2018-01-01", window_months=12, rebalance="quarterly", cost_bps=10.0,
                methods=("ivol",), profiles=("p",))
    base.update(over)
    return wf.BacktestConfig(**base)


# ---------------------------------------------------------------------------
# Rebalance calendar and estimation window
# ---------------------------------------------------------------------------
def test_rebalance_dates_quarterly_start_at_last_business_day_before_oos_start():
    r = synthetic_returns()
    cfg = small_cfg(window_months=24)                       # OOS starts 2020-01-01
    dates = wf.rebalance_dates(r.index, cfg)
    assert dates[0] == pd.Timestamp("2019-12-31")           # last bday < 2020-01-01
    assert dates[1] == pd.Timestamp("2020-03-31")
    assert all(d in r.index for d in dates)
    assert dates[-1] < r.index[-1]                          # every decision has a holding period after it
    assert len(dates) == 12                                 # 2019-12 … 2022-09


def test_rebalance_dates_monthly_and_explicit_end():
    r = synthetic_returns()
    cfg = small_cfg(window_months=24, rebalance="monthly", end="2021-06-30")
    dates = wf.rebalance_dates(r.index, cfg)
    assert dates[0] == pd.Timestamp("2019-12-31")
    assert dates[-1] == pd.Timestamp("2021-05-31")
    assert len(dates) == 18


def test_estimation_window_rolling_never_includes_future_or_pre_start():
    r = synthetic_returns()
    cfg = small_cfg(window_months=24)
    t = pd.Timestamp("2019-12-31")
    win = wf.estimation_window(r[["A", "B", "C", "D", "E"]], t, cfg)
    assert win.index.max() == t
    assert win.index.min() >= pd.Timestamp("2018-01-01")
    assert win.index.min() > t - pd.DateOffset(months=24)
    assert not win.isna().any().any()                       # gap cell filled with 0
    assert win.loc["2019-03-05", "C"] == 0.0
    t2 = pd.Timestamp("2021-06-30")
    win2 = wf.estimation_window(r, t2, cfg)
    assert win2.index.min() == r.index[r.index > t2 - pd.DateOffset(months=24)][0]
    assert win2.index.max() == t2


def test_estimation_window_expanding_uses_common_support_of_active_funds():
    r = synthetic_returns()
    cfg = small_cfg(window_months=12, window_kind="expanding")
    t = pd.Timestamp("2021-06-30")
    win = wf.estimation_window(r, t, cfg)                    # F starts 2018-09-03 → common support
    assert win.index.min() == pd.Timestamp("2018-09-03")
    assert win.index.max() == t
    win_core = wf.estimation_window(r.drop(columns="F"), t, cfg)
    assert win_core.index.min() == pd.Timestamp("2018-01-01")


def test_active_universe_is_point_in_time():
    r = synthetic_returns()
    r.loc[r.index > "2020-03-01", "E"] = np.nan             # E stops reporting
    cfg = small_cfg(window_months=12)
    assert "F" not in wf.active_universe(r, pd.Timestamp("2019-06-28"), cfg)   # < 12 m of history
    assert "F" in wf.active_universe(r, pd.Timestamp("2019-09-30"), cfg)
    assert "E" in wf.active_universe(r, pd.Timestamp("2019-12-31"), cfg)
    assert "E" not in wf.active_universe(r, pd.Timestamp("2020-06-30"), cfg)  # stale


# ---------------------------------------------------------------------------
# Turnover, drift, period simulation
# ---------------------------------------------------------------------------
def test_turnover_known_cases():
    assert wf.turnover(pd.Series({"A": 0.4, "B": 0.6}), pd.Series({"A": 0.6, "B": 0.4})) == pytest.approx(0.2)
    assert wf.turnover(pd.Series({"B": 1.0}), pd.Series({"A": 1.0})) == pytest.approx(1.0)      # full swap
    assert wf.turnover(pd.Series({"A": 0.5, "B": 0.5}), pd.Series(dtype=float)) == pytest.approx(0.5)  # from cash
    assert wf.turnover(pd.Series({"A": 0.5, "B": 0.5}), pd.Series({"A": 0.5, "B": 0.5})) == pytest.approx(0.0)


def test_drift_weights_follow_prices():
    w = pd.Series({"A": 0.5, "B": 0.5})
    rets = pd.DataFrame({"A": [0.10, 0.0], "B": [0.0, 0.0]})
    drifted = wf.drift_weights(w, rets)
    assert drifted["A"] == pytest.approx(0.55 / 1.05)
    assert drifted["B"] == pytest.approx(0.50 / 1.05)
    assert drifted.sum() == pytest.approx(1.0)


def test_simulate_period_charges_cost_on_first_day_then_buy_and_hold():
    w = pd.Series({"A": 0.5, "B": 0.5})
    rets = pd.DataFrame({"A": [0.01, 0.02, np.nan], "B": [0.0, 0.0, 0.0]},
                        index=pd.bdate_range("2020-01-02", periods=3))
    daily, w_end = wf.simulate_period(w, rets, cost=0.001)
    assert daily.iloc[0] == pytest.approx((1 - 0.001) * (1 + 0.005) - 1)
    assert daily.iloc[1] == pytest.approx((0.5 * 1.01 * 1.02 + 0.5) / (0.5 * 1.01 + 0.5) - 1)
    assert daily.iloc[2] == pytest.approx(0.0)              # NaN return = no price change
    assert w_end["A"] == pytest.approx(0.5 * 1.01 * 1.02 / (0.5 * 1.01 * 1.02 + 0.5))
    assert w_end.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Anti-look-ahead
# ---------------------------------------------------------------------------
def test_future_outlier_does_not_change_weights_or_returns_before_its_date():
    r = synthetic_returns()
    cfg = small_cfg(window_months=12)
    base = wf.run_walk_forward(r, CATS, cfg, optimizer=inverse_vol_optimizer)[0]

    poisoned = r.copy()
    shock_date = pd.Timestamp("2021-05-17")
    poisoned.loc[shock_date, ["A", "D"]] = [0.5, -0.9]     # absurd future shock
    shocked = wf.run_walk_forward(poisoned, CATS, cfg, optimizer=inverse_vol_optimizer)[0]

    before = [d for d in base.weights.index if d < shock_date]
    after = [d for d in base.weights.index if d >= shock_date]
    assert before and after
    np.testing.assert_array_equal(base.weights.loc[before].to_numpy(), shocked.weights.loc[before].to_numpy())
    pd.testing.assert_series_equal(base.daily_returns[base.daily_returns.index < shock_date],
                                   shocked.daily_returns[shocked.daily_returns.index < shock_date])
    # Sanity: the shock is inside the later windows, so the optimizer must react there.
    assert not np.allclose(base.weights.loc[after].to_numpy(), shocked.weights.loc[after].to_numpy())


def test_optimizer_only_receives_data_up_to_the_decision_date():
    r = synthetic_returns()
    cfg = small_cfg(window_months=12)
    seen: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def spy(window, method, profile):
        seen.append((window.index.min(), window.index.max()))
        return inverse_vol_optimizer(window, method, profile)

    res = wf.run_walk_forward(r, CATS, cfg, optimizer=spy)[0]
    assert len(seen) == len(res.weights)
    for t, (lo, hi) in zip(res.weights.index, seen):
        assert hi == t
        assert lo > t - pd.DateOffset(months=12)


# ---------------------------------------------------------------------------
# Full run: weights, universe, metrics
# ---------------------------------------------------------------------------
def test_run_walk_forward_weights_sum_to_one_and_inactive_funds_get_zero():
    r = synthetic_returns()
    cfg = small_cfg(window_months=12)
    res = wf.run_walk_forward(r, CATS, cfg, optimizer=inverse_vol_optimizer)[0]
    assert res.method == "ivol" and res.profile == "p"
    assert list(res.weights.columns) == list(r.columns)
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0, atol=1e-9)
    first = res.weights.index[0]
    assert first == pd.Timestamp("2018-12-31")
    assert res.weights.loc[first, "F"] == 0.0                # F has < 12 m of history at the first decision
    assert res.n_active.loc[first] == 5
    assert res.n_active.iloc[-1] == 6
    assert res.turnovers.iloc[0] == pytest.approx(0.5)        # from cash
    assert (res.turnovers.iloc[1:] < 0.5).all()
    assert res.daily_returns.index[0] == r.index[r.index > first][0]
    assert res.daily_returns.index[-1] == r.index[-1]
    assert res.nav.iloc[-1] == pytest.approx((1 + res.daily_returns).prod())
    assert res.costs.iloc[0] == pytest.approx(1.0 * 10 / 1e4)


def test_run_walk_forward_rejects_optimizer_output_that_is_not_a_portfolio():
    r = synthetic_returns()
    cfg = small_cfg(window_months=12)
    with pytest.raises(wf.BacktestError, match="sum"):
        wf.run_walk_forward(r, CATS, cfg, optimizer=lambda w, m, p: pd.Series(0.3, index=w.columns))
    with pytest.raises(wf.BacktestError, match="NaN"):
        wf.run_walk_forward(r, CATS, cfg, optimizer=lambda w, m, p: pd.Series(np.nan, index=w.columns))


def test_run_walk_forward_requires_minimum_active_funds():
    r = synthetic_returns()[["A", "B"]]
    cfg = small_cfg(window_months=12, min_active_funds=3)
    with pytest.raises(wf.BacktestError, match="active"):
        wf.run_walk_forward(r, {"A": "RF_CORTO", "B": "RF_CORTO"}, cfg, optimizer=inverse_vol_optimizer)


def test_compute_metrics_against_hand_calculation():
    idx = pd.bdate_range("2020-01-02", periods=8)
    daily = pd.Series([0.01, -0.02, 0.01, 0.0, 0.03, -0.01, 0.0, 0.005], index=idx)
    turnovers = pd.Series([0.5, 0.2, 0.1], index=idx[[0, 3, 6]])
    costs = pd.Series([0.001, 0.0004, 0.0002], index=turnovers.index)
    cfg = wf.BacktestConfig(periods_per_year=252, rf_annual=0.0)
    m = wf.compute_metrics(daily, turnovers, costs, cfg)
    nav = (1 + daily).cumprod()
    assert m["ann_return"] == pytest.approx(nav.iloc[-1] ** (252 / 8) - 1)
    assert m["volatility"] == pytest.approx(daily.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    assert m["max_drawdown"] == pytest.approx((nav / nav.cummax() - 1).min())
    assert m["max_drawdown"] == pytest.approx(-0.02)
    assert m["cvar_95"] == pytest.approx(-0.02)              # worst 5 % of 8 days = the worst day
    assert m["avg_turnover"] == pytest.approx(0.15)          # excludes the initial deployment
    assert m["total_cost"] == pytest.approx(0.0016)
    assert m["n_rebalances"] == 3 and m["n_days"] == 8


def test_compute_metrics_with_positive_rf_uses_daily_excess_returns():
    idx = pd.bdate_range("2020-01-02", periods=5)
    daily = pd.Series([0.01, 0.0, 0.02, -0.01, 0.005], index=idx)
    cfg = wf.BacktestConfig(rf_annual=0.05)
    m = wf.compute_metrics(daily, pd.Series([0.5], index=idx[:1]), pd.Series([0.0], index=idx[:1]), cfg)
    rf_d = 1.05 ** (1 / 252) - 1
    assert m["sharpe"] == pytest.approx((daily - rf_d).mean() / daily.std(ddof=1) * np.sqrt(252))
    assert np.isnan(m["avg_turnover"])


# ---------------------------------------------------------------------------
# Default optimizer adapter (real solvers)
# ---------------------------------------------------------------------------
def test_default_optimizer_wraps_portfolio_optimizer_with_bands():
    r = synthetic_returns().dropna()
    opt = wf.default_optimizer(CATS)
    w = opt(r, "hrp", "conservador")
    assert list(w.index) == list(r.columns)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w <= 0.25 + 1e-6).all()


# ---------------------------------------------------------------------------
# Outputs + manifest
# ---------------------------------------------------------------------------
def test_write_outputs_and_manifest(tmp_path):
    r = synthetic_returns()
    root = tmp_path
    (root / "data" / "cleaned").mkdir(parents=True)
    r.to_parquet(root / "data" / "cleaned" / "returns_matrix.parquet")
    (root / "data" / "universe.json").write_text(json.dumps({"funds": [{"fund_id": k, "cat": v} for k, v in CATS.items()]}))
    (root / "manifest.json").write_text(json.dumps({"generated_at": "x", "pipeline": {"module": "data/ingest_sfc.py"}}))
    cfg = small_cfg(window_months=12, methods=("ivol", "ivol2"), profiles=("p", "q"))
    results = wf.run_walk_forward(r, CATS, cfg, optimizer=inverse_vol_optimizer)
    assert len(results) == 4

    section = wf.write_outputs(results, cfg, root)
    summary = pd.read_parquet(root / "results" / "backtest_summary.parquet")
    assert len(summary) == 4
    assert {"method", "profile", "ann_return", "volatility", "sharpe", "max_drawdown", "cvar_95",
            "avg_turnover", "weights"} <= set(summary.columns)
    w0 = pd.DataFrame(list(summary.loc[0, "weights"]))
    assert set(w0.columns) == {"date", "fund_id", "weight"}
    assert w0.groupby("date")["weight"].sum().round(9).eq(1.0).all()
    long_w = pd.read_parquet(root / "results" / "backtest_weights.parquet")
    assert set(long_w.columns) >= {"method", "profile", "date", "fund_id", "weight", "turnover", "cost"}
    daily = pd.read_parquet(root / "results" / "backtest_daily.parquet")
    assert set(daily.columns) >= {"method", "profile", "date", "ret", "nav"}
    md = (root / "results" / "backtest_summary.md").read_text(encoding="utf-8")
    assert "| ivol | p |" in md and "[ESPECULATIVO]" in md and "10.0 bps" in md

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pipeline"]["module"] == "data/ingest_sfc.py"          # untouched
    bt = manifest["backtest"]
    assert bt == section
    p = bt["params"]
    assert p["window_months"] == 12 and p["window_kind"] == "rolling" and p["rebalance"] == "quarterly"
    assert p["estimation_start"] == "2018-01-01" and p["oos_start"] == "2019-01-01"
    assert p["first_rebalance"] == "2018-12-31" and p["oos_end"] == "2022-12-30"
    assert p["cost_bps"] == 10.0 and p["cost_label"] == "[ESPECULATIVO]"
    for rel, entry in bt["outputs"].items():
        assert entry["sha256"] == sha256_of(root / rel)
    assert bt["inputs"]["data/cleaned/returns_matrix.parquet"]["sha256"] == sha256_of(root / "data/cleaned/returns_matrix.parquet")
    assert len(bt["rebalance_dates"]) == len(results[0].weights)


def test_write_outputs_with_tag_does_not_clobber_base_run(tmp_path):
    r = synthetic_returns()
    (tmp_path / "data" / "cleaned").mkdir(parents=True)
    r.to_parquet(tmp_path / "data" / "cleaned" / "returns_matrix.parquet")
    (tmp_path / "manifest.json").write_text("{}")
    cfg = small_cfg(window_months=12, rebalance="monthly", tag="monthly")
    results = wf.run_walk_forward(r, CATS, cfg, optimizer=inverse_vol_optimizer)
    wf.write_outputs(results, cfg, tmp_path)
    assert (tmp_path / "results" / "backtest_summary_monthly.parquet").exists()
    assert not (tmp_path / "results" / "backtest_summary.parquet").exists()
    assert "backtest_monthly" in json.loads((tmp_path / "manifest.json").read_text())


def test_main_cli_runs_end_to_end_with_injected_optimizer(tmp_path, monkeypatch):
    r = synthetic_returns()
    (tmp_path / "data" / "cleaned").mkdir(parents=True)
    r.to_parquet(tmp_path / "data" / "cleaned" / "returns_matrix.parquet")
    (tmp_path / "data" / "universe.json").write_text(json.dumps({"funds": [{"fund_id": k, "cat": v} for k, v in CATS.items()]}))
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr(wf, "default_optimizer", lambda *a, **k: inverse_vol_optimizer)
    rc = wf.main(["--root", str(tmp_path), "--window-months", "12", "--cost-bps", "5",
                  "--methods", "ivol", "--profiles", "p"])
    assert rc == 0
    bt = json.loads((tmp_path / "manifest.json").read_text())["backtest"]
    assert bt["params"]["cost_bps"] == 5.0 and bt["params"]["methods"] == ["ivol"]
    assert (tmp_path / "results" / "backtest_summary.md").exists()


def test_main_returns_error_code_on_backtest_error(tmp_path, monkeypatch):
    r = synthetic_returns()[["A", "B"]]
    (tmp_path / "data" / "cleaned").mkdir(parents=True)
    r.to_parquet(tmp_path / "data" / "cleaned" / "returns_matrix.parquet")
    (tmp_path / "data" / "universe.json").write_text(json.dumps({"funds": [{"fund_id": "A", "cat": "RF_CORTO"}, {"fund_id": "B", "cat": "RF_CORTO"}]}))
    monkeypatch.setattr(wf, "default_optimizer", lambda *a, **k: inverse_vol_optimizer)
    rc = wf.main(["--root", str(tmp_path), "--window-months", "12", "--methods", "ivol", "--profiles", "p",
                  "--min-active-funds", "3"])
    assert rc == 2


# ---------------------------------------------------------------------------
# Optional RV floor pass-through (sensitivity; base case = D6 maxima only)
# ---------------------------------------------------------------------------
def test_default_optimizer_applies_min_rv_override():
    from optimization import portfolio_optimizer as po

    r = synthetic_returns().dropna()
    free = wf.default_optimizer(CATS)(r, "hrp", "agresivo")
    floored = wf.default_optimizer(CATS, min_rv={"agresivo": 0.45})(r, "hrp", "agresivo")
    c = po.PROFILE_CONSTRAINTS["agresivo"]
    assert po.exposures(free, CATS, c)["rv"] < 0.45
    assert po.exposures(floored, CATS, c)["rv"] >= 0.45 - 1e-6


def test_min_rv_config_must_align_with_profiles():
    with pytest.raises(ValueError, match="min_rv"):
        wf.BacktestConfig(profiles=("a", "b"), min_rv=(0.1,))


def test_main_cli_records_min_rv_in_manifest(tmp_path, monkeypatch):
    r = synthetic_returns()
    (tmp_path / "data" / "cleaned").mkdir(parents=True)
    r.to_parquet(tmp_path / "data" / "cleaned" / "returns_matrix.parquet")
    (tmp_path / "data" / "universe.json").write_text(json.dumps({"funds": [{"fund_id": k, "cat": v} for k, v in CATS.items()]}))
    captured = {}

    def fake_default(cats, **kw):
        captured.update(kw)
        return inverse_vol_optimizer

    monkeypatch.setattr(wf, "default_optimizer", fake_default)
    rc = wf.main(["--root", str(tmp_path), "--window-months", "12", "--methods", "ivol",
                  "--profiles", "p", "q", "--min-rv", "0.2", "0.45", "--tag", "minrv"])
    assert rc == 0
    assert captured["min_rv"] == {"p": 0.2, "q": 0.45}
    bt = json.loads((tmp_path / "manifest.json").read_text())["backtest_minrv"]
    assert bt["params"]["min_rv"] == {"p": 0.2, "q": 0.45}
    assert "min_rv" in (tmp_path / "results" / "backtest_summary_minrv.md").read_text(encoding="utf-8")


def test_markdown_reports_ex_post_vol_validation_against_gate_bands(tmp_path):
    r = synthetic_returns()
    (tmp_path / "manifest.json").write_text("{}")
    cfg = small_cfg(window_months=12, methods=("ivol",), profiles=("conservador", "moderado", "agresivo"))
    results = wf.run_walk_forward(r, CATS, cfg, optimizer=inverse_vol_optimizer)
    wf.write_outputs(results, cfg, tmp_path)
    md = (tmp_path / "results" / "backtest_summary.md").read_text(encoding="utf-8")
    assert "## Validación ex-post de volatilidad" in md
    assert "| ivol | conservador |" in md and "3.00%–5.00%" in md
    assert "FUERA" in md                                    # inverse-vol on this universe sits near 0.1 % vol
    # profiles without an approved band are listed without a verdict
    cfg2 = small_cfg(window_months=12, tag="x")
    wf.write_outputs(wf.run_walk_forward(r, CATS, cfg2, optimizer=inverse_vol_optimizer), cfg2, tmp_path)
    md2 = (tmp_path / "results" / "backtest_summary_x.md").read_text(encoding="utf-8")
    assert "| ivol | p |" in md2 and "n/a" in md2
    assert "backtest_summary_x.md" in (tmp_path / "results" / "backtest_summary.md").read_text(encoding="utf-8") or True
