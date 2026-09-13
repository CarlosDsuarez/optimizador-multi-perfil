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
