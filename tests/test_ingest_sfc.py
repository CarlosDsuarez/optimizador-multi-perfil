"""Tests for data/ingest_sfc.py — no network: SODA responses are synthetic fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data import ingest_sfc as ing

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures: synthetic SODA rows (everything arrives as strings, like the API)
# ---------------------------------------------------------------------------
def soda_row(date: str, code: int, clase: int, vu: float, **extra) -> dict:
    row = {
        "fecha_corte": f"{date}T00:00:00.000",
        "codigo_negocio": str(code),
        "codigo_entidad": "5",
        "nombre_entidad": "ENTIDAD TEST",
        "nombre_patrimonio": f"FONDO {code}",
        "nombre_subtipo_patrimonio": "FIC DE TIPO GENERAL",
        "tipo_participacion": str(clase),
        "valor_unidad_operaciones": f"{vu:.6f}",
        "valor_fondo_cierre_dia_t": "1000000.00",
        "numero_inversionistas": "120",
        "rentabilidad_diaria": "5.1",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Universe config
# ---------------------------------------------------------------------------
def test_load_universe_reads_repo_config_with_21_funds():
    uni = ing.load_universe(REPO / "data" / "universe.json")
    assert len(uni.funds) == 21
    assert sum(f.rol == "core" for f in uni.funds) == 18
    assert sum(f.rol == "late" for f in uni.funds) == 3
    gx = next(f for f in uni.funds if f.fund_id == "GLOBALX_COLSELECT")
    assert gx.codes == [43502, 129433]
    assert gx.clase == 800
    assert uni.datasets["series"] == "qhpu-8ixx"


def test_load_universe_rejects_duplicate_fund_ids(tmp_path):
    cfg = json.loads((REPO / "data" / "universe.json").read_text())
    cfg["funds"] = cfg["funds"][:2]
    cfg["funds"][1]["fund_id"] = cfg["funds"][0]["fund_id"]
    p = tmp_path / "u.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ing.ConfigError, match="fund_id"):
        ing.load_universe(p)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_soda_rows_casts_types():
    rows = [soda_row("2024-01-02", 2852, 800, 12345.678901),
            soda_row("2024-01-03", 2852, 800, 12350.1)]
    df = ing.parse_soda_rows(rows)
    assert list(df["fecha_corte"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert df["codigo_negocio"].dtype.kind == "i" and df["codigo_negocio"].iloc[0] == 2852
    assert df["tipo_participacion"].dtype.kind == "i" and df["tipo_participacion"].iloc[0] == 800
    assert df["valor_unidad_operaciones"].dtype.kind == "f"
    assert df["valor_unidad_operaciones"].iloc[0] == pytest.approx(12345.678901)


def test_parse_soda_rows_empty_returns_typed_empty_frame():
    df = ing.parse_soda_rows([])
    assert len(df) == 0
    assert "valor_unidad_operaciones" in df.columns


def test_parse_soda_rows_rejects_nonpositive_unit_value():
    rows = [soda_row("2024-01-02", 2852, 800, 0.0)]
    with pytest.raises(ing.DataQualityError, match="valor_unidad"):
        ing.parse_soda_rows(rows)


# ---------------------------------------------------------------------------
# SODA client — fake transport, no network
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else []
        self.headers = headers or {}
        self.text = text
        self.url = "https://fake"

    def json(self):
        return self._payload


class FakeSession:
    """Scripted transport: each call pops the next item; Exception instances are raised."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(script, **kw):
    sleeps = []
    session = FakeSession(script)
    client = ing.SodaClient("www.datos.gov.co", session=session, sleep=sleeps.append, **kw)
    return client, session, sleeps


def test_get_all_paginates_until_short_page():
    page1 = [soda_row("2024-01-01", 1, 800, 1.0)] * 3
    page2 = [soda_row("2024-01-04", 1, 800, 1.0)]
    client, session, _ = make_client([FakeResponse(payload=page1), FakeResponse(payload=page2)], page_size=3)
    rows = client.get_all("qhpu-8ixx", {"$where": "codigo_negocio=1"})
    assert len(rows) == 4
    assert [c["params"]["$offset"] for c in session.calls] == [0, 3]
    assert all(c["params"]["$limit"] == 3 for c in session.calls)
    assert session.calls[0]["url"].endswith("/resource/qhpu-8ixx.json")


def test_get_retries_on_timeout_then_succeeds():
    import requests
    client, session, sleeps = make_client(
        [requests.Timeout("t1"), requests.ConnectionError("t2"), FakeResponse(payload=[{"a": "1"}])]
    )
    assert client.get("qhpu-8ixx", {}) == [{"a": "1"}]
    assert len(session.calls) == 3
    assert len(sleeps) == 2 and all(s > 0 for s in sleeps)


def test_get_raises_ingest_error_after_max_attempts():
    import requests
    client, session, sleeps = make_client([requests.Timeout("x")] * 3, max_attempts=3)
    with pytest.raises(ing.IngestError, match="3 attempts"):
        client.get("qhpu-8ixx", {"$limit": 1})
    assert len(session.calls) == 3
    assert len(sleeps) == 2


def test_get_honors_retry_after_on_429():
    client, session, sleeps = make_client(
        [FakeResponse(status=429, headers={"Retry-After": "7"}), FakeResponse(payload=[{"ok": "1"}])]
    )
    assert client.get("qhpu-8ixx", {}) == [{"ok": "1"}]
    assert sleeps == [7.0]


def test_get_retries_on_5xx():
    client, _, sleeps = make_client([FakeResponse(status=503, text="down"), FakeResponse(payload=[])])
    assert client.get("qhpu-8ixx", {}) == []
    assert len(sleeps) == 1


def test_get_does_not_retry_on_client_error():
    client, session, sleeps = make_client([FakeResponse(status=400, text="bad soql")])
    with pytest.raises(ing.IngestError, match="400"):
        client.get("qhpu-8ixx", {"$where": "nonsense"})
    assert len(session.calls) == 1 and sleeps == []


def test_app_token_sets_header_and_flag():
    session = FakeSession([])
    client = ing.SodaClient("www.datos.gov.co", app_token="abc", session=session)
    assert session.headers["X-App-Token"] == "abc"
    assert client.app_token_used is True
    assert ing.SodaClient("www.datos.gov.co", session=FakeSession([])).app_token_used is False


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------
def test_dedupe_removes_exact_duplicates_and_reports_stats():
    r = soda_row("2024-01-02", 3644, 501, 100.0)
    df = ing.parse_soda_rows([r, r, r, soda_row("2024-01-03", 3644, 501, 101.0)])
    out, stats = ing.dedupe(df)
    assert len(out) == 2
    assert stats == {"rows_in": 4, "rows_out": 2, "duplicates_removed": 2, "by_code": {3644: 2}}


def test_dedupe_raises_on_conflicting_duplicate_key():
    df = ing.parse_soda_rows([soda_row("2024-01-02", 1, 800, 100.0), soda_row("2024-01-02", 1, 800, 100.5)])
    with pytest.raises(ing.DataQualityError, match="conflicting"):
        ing.dedupe(df)


# ---------------------------------------------------------------------------
# Code-chain splicing (D1)
# ---------------------------------------------------------------------------
def fund(fund_id="GX", codes=(43502, 129433), clase=800, inicio="2016-01-01", rol="core", cat="RV_LOCAL"):
    return ing.Fund(fund_id=fund_id, cat=cat, rol=rol, codes=list(codes), clase=clase,
                    nombre="x", admin="y", inicio_esperado=pd.Timestamp(inicio))


def chain_frame(last_a="2025-12-31", first_b="2026-01-01", vu_a=22826.99, vu_b=22826.99):
    rows = [soda_row("2025-12-30", 43502, 800, 22800.0), soda_row(last_a, 43502, 800, vu_a),
            soda_row(first_b, 129433, 800, vu_b), soda_row("2026-01-02", 129433, 800, 22900.0)]
    return ing.parse_soda_rows(rows)


def test_splice_codes_chains_contiguous_codes_and_records_mapping():
    out, mappings = ing.splice_codes(chain_frame(), fund(), tolerance=0.02)
    assert len(out) == 4
    assert out["fecha_corte"].is_monotonic_increasing
    assert mappings == [{
        "fund_id": "GX", "from_code": 43502, "to_code": 129433,
        "last_date_from": "2025-12-31", "first_date_to": "2026-01-01",
        "vu_from": 22826.99, "vu_to": 22826.99, "rel_diff": 0.0, "gap_days": 1,
    }]


def test_splice_codes_single_code_has_no_mapping():
    df = ing.parse_soda_rows([soda_row("2024-01-02", 2852, 800, 1.0)])
    out, mappings = ing.splice_codes(df, fund(codes=(2852,)), tolerance=0.02)
    assert len(out) == 1 and mappings == []


def test_splice_codes_rejects_unit_value_discontinuity():
    with pytest.raises(ing.DataQualityError, match="continuity"):
        ing.splice_codes(chain_frame(vu_b=24000.0), fund(), tolerance=0.02)


def test_splice_codes_rejects_gap_longer_than_one_day():
    with pytest.raises(ing.DataQualityError, match="gap"):
        ing.splice_codes(chain_frame(first_b="2026-01-05"), fund(), tolerance=0.02)


def test_splice_codes_rejects_overlapping_codes():
    with pytest.raises(ing.DataQualityError, match="overlap"):
        ing.splice_codes(chain_frame(first_b="2025-12-31"), fund(), tolerance=0.02)


def test_splice_codes_rejects_missing_code():
    df = chain_frame()
    with pytest.raises(ing.DataQualityError, match="no rows"):
        ing.splice_codes(df[df.codigo_negocio == 43502], fund(), tolerance=0.02)


# ---------------------------------------------------------------------------
# Business calendar (Colombia) + price alignment
# ---------------------------------------------------------------------------
def test_business_calendar_excludes_weekends_and_colombian_holidays():
    cal = ing.business_calendar("2024-01-01", "2024-01-12")
    assert pd.Timestamp("2024-01-01") not in cal      # Año Nuevo (Monday)
    assert pd.Timestamp("2024-01-08") not in cal      # Reyes Magos (observado, Monday)
    assert pd.Timestamp("2024-01-06") not in cal      # Saturday
    assert pd.Timestamp("2024-01-02") in cal
    assert list(cal) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                                             "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]))


def daily_series(start, values):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def test_build_price_matrix_aligns_series_to_business_calendar():
    cal = ing.business_calendar("2024-01-09", "2024-01-16")
    # 2024-01-12 Fri, 13 Sat, 14 Sun, 15 Mon
    s = daily_series("2024-01-12", [100.0, 100.1, 100.2, 100.3, 100.4])
    prices = ing.build_price_matrix({"MM": s}, cal)
    assert list(prices.index) == list(cal)
    assert list(prices.columns) == ["MM"]
    assert prices.loc["2024-01-12", "MM"] == 100.0
    assert prices.loc["2024-01-15", "MM"] == 100.3
    assert np.isnan(prices.loc["2024-01-09", "MM"])   # before inception → NaN


# ---------------------------------------------------------------------------
# Returns + NaN policy
# ---------------------------------------------------------------------------
def test_returns_fold_weekend_accrual_into_monday():
    cal = ing.business_calendar("2024-01-12", "2024-01-16")
    s = daily_series("2024-01-12", [100.0, 100.1, 100.2, 100.3, 100.4])
    rets, gaps = ing.compute_returns(ing.build_price_matrix({"MM": s}, cal))
    assert np.isnan(rets.loc["2024-01-12", "MM"])                         # first obs
    assert rets.loc["2024-01-15", "MM"] == pytest.approx(np.log(100.3 / 100.0))
    assert rets.loc["2024-01-16", "MM"] == pytest.approx(np.log(100.4 / 100.3))
    assert gaps == []


def test_returns_nan_before_inception_and_after_last_observation():
    cal = ing.business_calendar("2024-01-09", "2024-01-19")
    s = daily_series("2024-01-11", [10.0, 10.5, 11.0])                     # Thu..Sat
    rets, gaps = ing.compute_returns(ing.build_price_matrix({"F": s}, cal))
    assert rets.loc[:"2024-01-11", "F"].isna().all()
    assert rets.loc["2024-01-12", "F"] == pytest.approx(np.log(10.5 / 10.0))
    assert rets.loc["2024-01-15":, "F"].isna().all()
    assert gaps == []


def test_returns_bridge_intra_life_gap_and_log_it():
    cal = ing.business_calendar("2024-01-15", "2024-01-19")               # Mon..Fri
    s = pd.Series([100.0, 101.0, 104.0], index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-19"]))
    rets, gaps = ing.compute_returns(ing.build_price_matrix({"F": s}, cal))
    assert rets.loc["2024-01-16", "F"] == pytest.approx(np.log(101 / 100))
    assert np.isnan(rets.loc["2024-01-17", "F"]) and np.isnan(rets.loc["2024-01-18", "F"])
    assert rets.loc["2024-01-19", "F"] == pytest.approx(np.log(104 / 101))   # cumulative over the gap
    assert gaps == [{"fund_id": "F", "date": "2024-01-19", "prev_date": "2024-01-16",
                     "missing_bdays": 2, "bridged_log_return": pytest.approx(np.log(104 / 101))}]


# ---------------------------------------------------------------------------
# Sufficiency gate — halt, never substitute
# ---------------------------------------------------------------------------
SUFF = {"max_start_delay_days": 31, "min_completeness": 0.99, "max_gap_days": 31, "max_end_lag_days": 3,
        "splice_tolerance": 0.02}


def full_daily(start, end, v0=100.0):
    idx = pd.date_range(start, end, freq="D")
    return pd.Series(np.linspace(v0, v0 * 1.1, len(idx)), index=idx)


def test_check_sufficiency_passes_and_reports_metrics():
    s = full_daily("2016-01-01", "2016-12-31")
    report = ing.check_sufficiency({"A": s}, [fund("A", codes=(1,), inicio="2016-01-01")], SUFF,
                                   data_end=pd.Timestamp("2016-12-31"))
    assert report == [{"fund_id": "A", "first_date": "2016-01-01", "last_date": "2016-12-31", "n_obs": 366,
                       "completeness": 1.0, "max_gap_days": 0, "expected_start": "2016-01-01", "ok": True,
                       "failures": []}]


def test_check_sufficiency_halts_when_history_starts_too_late():
    s = full_daily("2016-06-01", "2016-12-31")
    with pytest.raises(ing.InsufficientDataError, match=r"A.*starts 2016-06-01"):
        ing.check_sufficiency({"A": s}, [fund("A", codes=(1,), inicio="2016-01-01")], SUFF,
                              data_end=pd.Timestamp("2016-12-31"))


def test_check_sufficiency_halts_when_fund_stopped_reporting():
    s = full_daily("2016-01-01", "2016-10-31")
    with pytest.raises(ing.InsufficientDataError, match=r"last observation 2016-10-31"):
        ing.check_sufficiency({"A": s}, [fund("A", codes=(1,), inicio="2016-01-01")], SUFF,
                              data_end=pd.Timestamp("2016-12-31"))


def test_check_sufficiency_halts_on_low_completeness_or_long_gap():
    s = full_daily("2016-01-01", "2016-12-31")
    s = s.drop(s.index[100:140])                       # 40-day hole → completeness 0.89, gap 40
    with pytest.raises(ing.InsufficientDataError) as ei:
        ing.check_sufficiency({"A": s}, [fund("A", codes=(1,), inicio="2016-01-01")], SUFF,
                              data_end=pd.Timestamp("2016-12-31"))
    assert "completeness" in str(ei.value) and "gap" in str(ei.value)


def test_check_sufficiency_reports_every_failing_fund_at_once():
    good = full_daily("2016-01-01", "2016-12-31")
    late = full_daily("2016-06-01", "2016-12-31")
    dead = full_daily("2016-01-01", "2016-06-30")
    funds = [fund("GOOD", codes=(1,), inicio="2016-01-01"), fund("LATE", codes=(2,), inicio="2016-01-01"),
             fund("DEAD", codes=(3,), inicio="2016-01-01")]
    with pytest.raises(ing.InsufficientDataError) as ei:
        ing.check_sufficiency({"GOOD": good, "LATE": late, "DEAD": dead}, funds, SUFF,
                              data_end=pd.Timestamp("2016-12-31"))
    msg = str(ei.value)
    assert "LATE" in msg and "DEAD" in msg and "GOOD" not in msg
    assert ei.value.report[0]["ok"] is True and [r["fund_id"] for r in ei.value.report if not r["ok"]] == ["LATE", "DEAD"]


# ---------------------------------------------------------------------------
# Orchestration: raw storage, cleaned outputs, manifest (routed fake transport)
# ---------------------------------------------------------------------------
class RoutedSession:
    """Fake transport that answers by dataset + $where, honoring $limit/$offset and $order DESC."""

    def __init__(self, series: dict[int, list[dict]], metadata: dict[int, list[dict]]):
        self.series, self.metadata = series, metadata
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append({"url": url, "params": params})
        import re
        code = int(re.search(r"codigo_negocio=(\d+)", params["$where"]).group(1))
        clase = re.search(r"tipo_participacion=(\d+)", params["$where"])
        if "qhpu-8ixx" in url:
            rows = self.series.get(code, [])
        else:
            rows = self.metadata.get(code, [])
        if clase:
            rows = [r for r in rows if r["tipo_participacion"] == clase.group(1)]
        if "DESC" in params.get("$order", ""):
            rows = sorted(rows, key=lambda r: r["fecha_corte"], reverse=True)
        off, lim = int(params.get("$offset", 0)), int(params.get("$limit", 50000))
        return FakeResponse(payload=rows[off:off + lim])


def synth_rows(code, clase, start, end, v0, extra_clase=None):
    idx = pd.date_range(start, end, freq="D")
    vals = v0 * np.cumprod(1 + 0.0002 * np.sin(np.arange(len(idx))))
    rows = [soda_row(str(d.date()), code, clase, v) for d, v in zip(idx, vals)]
    if extra_clase:
        rows += [soda_row(str(d.date()), code, extra_clase, v * 2) for d, v in zip(idx, vals)]
    return rows, vals[-1]


def write_universe(tmp_path, funds, start="2024-01-01"):
    cfg = {"domain": "www.datos.gov.co", "datasets": {"series": "qhpu-8ixx", "metadata": "djw7-ur7t"},
           "start_date": start, "sufficiency": SUFF, "funds": funds}
    p = tmp_path / "universe.json"
    p.write_text(json.dumps(cfg))
    return p


def fund_cfg(fund_id, codes, clase, inicio="2024-01-01", rol="core", cat="RF_CORTO"):
    return {"fund_id": fund_id, "cat": cat, "rol": rol, "codes": list(codes), "clase": clase,
            "nombre": f"Fondo {fund_id}", "admin": "Admin", "inicio_esperado": inicio}


@pytest.fixture
def synthetic_world(tmp_path):
    """Two funds: A (code 1, extra class 999, one duplicated row) and B (codes 2→3 spliced on 2024-07-01)."""
    a_rows, _ = synth_rows(1, 800, "2024-01-01", "2024-12-31", 100.0, extra_clase=999)
    a_rows.append(dict(a_rows[10]))                                   # exact duplicate
    b1_rows, last_vu = synth_rows(2, 501, "2024-01-01", "2024-06-30", 50.0)
    b2_rows, _ = synth_rows(3, 501, "2024-07-01", "2024-12-31", last_vu)
    series = {1: a_rows, 2: b1_rows, 3: b2_rows}
    metadata = {1: a_rows[-2:], 3: b2_rows[-1:]}                      # B mirrors A only for recent dates
    session = RoutedSession(series, metadata)
    uni_path = write_universe(tmp_path, [fund_cfg("A", [1], 800), fund_cfg("B", [2, 3], 501)])
    client = ing.SodaClient("www.datos.gov.co", session=session, sleep=lambda s: None)
    return {"tmp": tmp_path, "session": session, "client": client, "universe": ing.load_universe(uni_path),
            "last_vu_b1": last_vu}


def test_fetch_raw_stores_rows_untransformed_and_returns_query_params(synthetic_world):
    w = synthetic_world
    entries = ing.fetch_raw(w["client"], w["universe"], w["tmp"] / "raw")
    f = w["tmp"] / "raw" / "qhpu-8ixx_1.parquet"
    assert f.exists()
    raw = pd.read_parquet(f)
    assert len(raw) == 366 * 2 + 1                                    # both classes + duplicate, untouched
    assert isinstance(raw["valor_unidad_operaciones"].iloc[0], str)   # strings as delivered, not floats
    assert raw["fecha_corte"].iloc[0] == "2024-01-01T00:00:00.000"
    e = entries["qhpu-8ixx_1.parquet"]
    assert e["dataset"] == "qhpu-8ixx" and e["query_params"]["$where"] == "codigo_negocio=1"
    assert e["rows"] == 733 and e["pages"] == 1
    meta = pd.read_parquet(w["tmp"] / "raw" / "djw7-ur7t_metadata.parquet")
    assert set(meta["codigo_negocio"]) == {"1", "3"}
    assert entries["djw7-ur7t_metadata.parquet"]["query_params"]  # one param set per fund


def test_fetch_raw_halts_when_a_code_returns_no_rows(synthetic_world):
    w = synthetic_world
    w["session"].series.pop(3)
    with pytest.raises(ing.IngestError, match="codigo_negocio=3.*no rows"):
        ing.fetch_raw(w["client"], w["universe"], w["tmp"] / "raw")


def test_run_end_to_end_writes_cleaned_outputs_and_manifest(synthetic_world):
    w = synthetic_world
    root = w["tmp"]
    manifest = ing.run(w["universe"], root, client=w["client"])

    rets = pd.read_parquet(root / "data" / "cleaned" / "returns_matrix.parquet")
    prices = pd.read_parquet(root / "data" / "cleaned" / "prices_matrix.parquet")
    master = pd.read_parquet(root / "data" / "cleaned" / "funds_master.parquet")
    assert list(rets.columns) == ["A", "B"]
    assert rets.index[0] == pd.Timestamp("2024-01-02") and rets.index[-1] == pd.Timestamp("2024-12-31")
    assert rets.index.equals(prices.index)
    assert rets["B"].iloc[1:].notna().all()                           # splice left no hole
    # 2024-07-01 is a CO holiday: first business day under code 3 is 07-02; return across the splice is tiny
    assert abs(rets.loc["2024-07-02", "B"]) < 0.01
    assert np.isnan(prices.loc["2024-06-28":"2024-07-02", "B"]).sum() == 0
    assert list(master["fund_id"]) == ["A", "B"] and master.set_index("fund_id").loc["B", "codes"] == "2>3"

    assert (root / "manifest.json").exists()
    assert manifest["dedupe"]["duplicates_removed"] == 1
    assert manifest["code_mappings"][0]["from_code"] == 2 and manifest["code_mappings"][0]["to_code"] == 3
    assert manifest["returns"]["nan_policy"] == ing.NAN_POLICY
    assert manifest["sufficiency"]["report"][1]["ok"] is True
    assert manifest["source"]["app_token_used"] is False
    assert manifest["raw"]["data/raw/qhpu-8ixx_2.parquet"]["query_params"]["$where"] == "codigo_negocio=2"
    for rel, entry in {**manifest["raw"], **manifest["cleaned"]}.items():
        assert entry["sha256"] == ing.sha256_of(root / rel)
    assert manifest["universe"]["sha256"] == ing.sha256_of(w["universe"].config_path)
    assert manifest["generated_at"].endswith("Z")


def test_run_halts_before_writing_cleaned_when_data_insufficient(synthetic_world):
    w = synthetic_world
    w["session"].series[3] = w["session"].series[3][:60]              # B stops reporting in August
    with pytest.raises(ing.InsufficientDataError, match="B"):
        ing.run(w["universe"], w["tmp"], client=w["client"])
    assert not (w["tmp"] / "data" / "cleaned").exists()
    assert not (w["tmp"] / "manifest.json").exists()
    assert (w["tmp"] / "data" / "raw" / "qhpu-8ixx_3.parquet").exists()   # raw is kept for diagnosis


def test_run_skip_download_reuses_raw_without_network(synthetic_world):
    w = synthetic_world
    ing.run(w["universe"], w["tmp"], client=w["client"])
    n_calls = len(w["session"].calls)
    dead = ing.SodaClient("www.datos.gov.co", session=FakeSession([RuntimeError("network used")]), sleep=lambda s: None)
    manifest = ing.run(w["universe"], w["tmp"], client=dead, skip_download=True)
    assert len(w["session"].calls) == n_calls
    assert manifest["raw"]["data/raw/qhpu-8ixx_1.parquet"]["query_params"]["$where"] == "codigo_negocio=1"


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib
    f = tmp_path / "x.bin"
    f.write_bytes(b"hola mundo" * 1000)
    assert ing.sha256_of(f) == hashlib.sha256(f.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Remaining branches: config, metadata cross-check, raw reuse, CLI
# ---------------------------------------------------------------------------
def test_load_universe_rejects_code_assigned_to_two_funds(tmp_path):
    p = write_universe(tmp_path, [fund_cfg("A", [1], 800), fund_cfg("B", [1, 2], 800)])
    with pytest.raises(ing.ConfigError, match="codigo_negocio"):
        ing.load_universe(p)


def test_parse_soda_rows_rejects_null_key():
    with pytest.raises(ing.DataQualityError, match="null codigo_negocio"):
        ing.parse_soda_rows([soda_row("2024-01-02", 1, 800, 1.0, codigo_negocio="")])


def test_metadata_crosscheck_halts_on_unit_value_mismatch(synthetic_world):
    w = synthetic_world
    bad = dict(w["session"].metadata[1][-1])
    bad["valor_unidad_operaciones"] = "999.0"
    w["session"].metadata[1] = [bad]
    with pytest.raises(ing.DataQualityError, match="metadata dataset VU"):
        ing.run(w["universe"], w["tmp"], client=w["client"])


def test_metadata_crosscheck_reports_absent_and_matching_funds(synthetic_world):
    w = synthetic_world
    w["session"].metadata.pop(3)
    m = ing.run(w["universe"], w["tmp"], client=w["client"])
    assert [(r["fund_id"], r["status"]) for r in m["metadata_crosscheck"]] == [("A", "match"), ("B", "absent_in_metadata_dataset")]
    master = pd.read_parquet(w["tmp"] / "data" / "cleaned" / "funds_master.parquet").set_index("fund_id")
    assert master.loc["A", "api_nombre_patrimonio"] == "FONDO 1" and pd.isna(master.loc["B", "api_nombre_patrimonio"])


def test_load_raw_requires_downloaded_files(synthetic_world):
    w = synthetic_world
    with pytest.raises(ing.IngestError, match="raw file missing"):
        ing.run(w["universe"], w["tmp"], client=w["client"], skip_download=True)


def test_main_cli_exit_codes(synthetic_world, monkeypatch):
    w = synthetic_world
    monkeypatch.setattr(ing, "SodaClient", lambda *a, **k: w["client"])
    monkeypatch.setenv("SOCRATA_APP_TOKEN", "")
    argv = ["--universe", str(w["universe"].config_path), "--root", str(w["tmp"])]
    assert ing.main(argv) == 0
    assert (w["tmp"] / "manifest.json").exists()
    assert ing.main(argv + ["--skip-download", "-v"]) == 0

    w["session"].series[3] = w["session"].series[3][:60]          # B dies → InsufficientDataError → 2
    assert ing.main(argv) == 2

    w["session"].series.pop(3)                                     # no rows → IngestError → 1
    assert ing.main(argv) == 1


def test_parse_soda_rows_tolerates_non_integer_informational_fields():
    # Real quirk (2026-09-11, code 60678): numero_inversionistas = "1.000351". Not part of the row key → keep as float.
    df = ing.parse_soda_rows([soda_row("2024-01-02", 60678, 800, 1.0, numero_inversionistas="1.000351", codigo_entidad="")])
    assert df["numero_inversionistas"].iloc[0] == pytest.approx(1.000351)
    assert pd.isna(df["codigo_entidad"].iloc[0])


def test_invalid_unit_value_in_unused_class_is_recorded_not_fatal(synthetic_world):
    # Real quirk (11407 clase 520, 2018-02-07): VU = 0 on the class's launch day. Class 520 is not in the universe.
    w = synthetic_world
    w["session"].series[1].append(soda_row("2023-12-31", 1, 999, 0.0))
    m = ing.run(w["universe"], w["tmp"], client=w["client"])
    assert m["quality"]["invalid_vu_rows_ignored"] == [{"codigo_negocio": 1, "tipo_participacion": 999, "n": 1,
                                                        "dates": ["2023-12-31"]}]


def test_invalid_unit_value_in_selected_class_halts(synthetic_world):
    w = synthetic_world
    w["session"].series[1].append(soda_row("2023-12-31", 1, 800, 0.0))
    with pytest.raises(ing.DataQualityError, match=r"\[A\].*valor_unidad"):
        ing.run(w["universe"], w["tmp"], client=w["client"])
