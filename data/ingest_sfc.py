"""SFC FIC ingestion + cleaning layer (Fase 1, Proyecto 2 — Optimizador Multi-Perfil HRP).

Pipeline
--------
1. ``SodaClient``      — HTTP client for datos.gov.co SODA 2.1 with explicit retry/backoff.
2. ``fetch_raw``       — pulls ``qhpu-8ixx`` (series) + ``djw7-ur7t`` (metadata) for the approved
                         universe and stores them untouched in ``data/raw/*.parquet``.
3. ``build_clean``     — dedupe, class selection, code-chain splicing, business-day calendar,
                         log returns with an explicit NaN policy → ``data/cleaned/*.parquet``.
4. ``write_manifest``  — ``manifest.json`` with SHA256, timestamps, exact Socrata query params,
                         applied code mappings and data-quality checks.

Design decisions come from ``docs/fase0_data_gate.md`` (D1, D3, D8, D9, D11 approved 2026-09-11).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

import holidays
import numpy as np
import pandas as pd
import requests

log = logging.getLogger("ingest_sfc")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class IngestError(RuntimeError):
    """Base class for every failure raised by this module."""


class ConfigError(IngestError):
    """Universe configuration is malformed."""


class DataQualityError(IngestError):
    """Raw data violates an invariant (non-positive unit value, conflicting duplicates, ...)."""


class InsufficientDataError(IngestError):
    """A fund of the approved universe has less data than Fase 0 found. Pipeline halts; no substitution."""

    def __init__(self, message: str, report: list[dict]):
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Universe config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fund:
    fund_id: str
    cat: str
    rol: str
    codes: list[int]
    clase: int
    nombre: str
    admin: str
    inicio_esperado: pd.Timestamp


@dataclass(frozen=True)
class Universe:
    domain: str
    datasets: dict[str, str]
    start_date: pd.Timestamp
    sufficiency: dict[str, float]
    funds: list[Fund] = field(default_factory=list)
    config_path: Path | None = None

    @property
    def all_codes(self) -> list[int]:
        return sorted({c for f in self.funds for c in f.codes})


def load_universe(path: Path | str) -> Universe:
    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    funds = [
        Fund(
            fund_id=f["fund_id"],
            cat=f["cat"],
            rol=f["rol"],
            codes=[int(c) for c in f["codes"]],
            clase=int(f["clase"]),
            nombre=f["nombre"],
            admin=f["admin"],
            inicio_esperado=pd.Timestamp(f["inicio_esperado"]),
        )
        for f in cfg["funds"]
    ]
    ids = [f.fund_id for f in funds]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ConfigError(f"duplicate fund_id in {path}: {dupes}")
    codes = [c for f in funds for c in f.codes]
    if len(codes) != len(set(codes)):
        dupes = sorted({c for c in codes if codes.count(c) > 1})
        raise ConfigError(f"codigo_negocio assigned to more than one fund in {path}: {dupes}")
    return Universe(
        domain=cfg["domain"],
        datasets=dict(cfg["datasets"]),
        start_date=pd.Timestamp(cfg["start_date"]),
        sufficiency=dict(cfg["sufficiency"]),
        funds=funds,
        config_path=path,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
# Columns the cleaning layer depends on. Raw parquet keeps *every* column the API returns.
# Only the row key is cast strictly to int; informational numeric fields are tolerant floats because the
# source has occasional non-integer values (e.g. numero_inversionistas = 1.000351 for code 60678).
KEY_COLS = ["fecha_corte", "codigo_negocio", "tipo_participacion"]
FLOAT_COLS = ("valor_unidad_operaciones", "valor_fondo_cierre_dia_t", "rentabilidad_diaria",
              "codigo_entidad", "numero_inversionistas")


def parse_soda_rows(rows: list[dict], *, strict_vu: bool = True) -> pd.DataFrame:
    """Cast a list of SODA JSON rows (all strings) into a typed DataFrame.

    With ``strict_vu`` (default) raises ``DataQualityError`` if any ``valor_unidad_operaciones`` is
    missing or <= 0. ``build_clean`` parses with ``strict_vu=False`` and enforces the check only on the
    classes the universe actually uses, recording the rest in the manifest.
    """
    if not rows:
        cols = KEY_COLS + list(FLOAT_COLS)
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in FLOAT_COLS else "object") for c in cols})
    df = pd.DataFrame(rows)
    df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], format="ISO8601")
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    # codigo_negocio / tipo_participacion are the row key: must be non-null ints
    for c in ("codigo_negocio", "tipo_participacion"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any():
            raise DataQualityError(f"null {c} in {int(v.isna().sum())} rows")
        df[c] = v.astype("int64")
    if strict_vu:
        bad = _invalid_vu_mask(df)
        if bad.any():
            sample = df.loc[bad, KEY_COLS].head(5).to_dict("records")
            raise DataQualityError(f"valor_unidad_operaciones missing or <= 0 in {int(bad.sum())} rows, e.g. {sample}")
    return df


def _invalid_vu_mask(df: pd.DataFrame) -> pd.Series:
    vu = df["valor_unidad_operaciones"]
    return vu.isna() | (vu <= 0)


# ---------------------------------------------------------------------------
# SODA client with explicit retry/backoff
# ---------------------------------------------------------------------------
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class SodaClient:
    """Thin SODA 2.1 client. Every request is retried on timeouts, connection errors,
    HTTP 429 and 5xx with exponential backoff (+ jitter); ``Retry-After`` is honored.
    Exhausting ``max_attempts`` raises ``IngestError`` — never a silent empty result.
    """

    def __init__(
        self,
        domain: str,
        app_token: str | None = None,
        session: requests.Session | None = None,
        *,
        page_size: int = 50_000,
        timeout: float = 120.0,
        max_attempts: int = 6,
        base_wait: float = 2.0,
        max_wait: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = f"https://{domain}/resource/{{}}.json"
        self.session = session or requests.Session()
        self.app_token_used = bool(app_token)
        if app_token:
            self.session.headers["X-App-Token"] = app_token
        self.page_size = page_size
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.base_wait = base_wait
        self.max_wait = max_wait
        self._sleep = sleep

    def _backoff(self, attempt: int) -> float:
        wait = min(self.max_wait, self.base_wait * (2 ** (attempt - 1)))
        return wait * random.uniform(0.5, 1.0)

    def get(self, dataset: str, params: dict) -> list[dict]:
        """One request (one page). Retries per class docstring."""
        url = self.base_url.format(dataset)
        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code == 200:
                    return resp.json()
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRYABLE_STATUS:
                    raise IngestError(f"[{dataset}] non-retryable {last_err} params={params}")
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None and attempt < self.max_attempts:
                    log.warning("[%s] attempt %d/%d %s; Retry-After=%s", dataset, attempt, self.max_attempts, last_err, retry_after)
                    self._sleep(float(retry_after))
                    continue
            if attempt < self.max_attempts:
                wait = self._backoff(attempt)
                log.warning("[%s] attempt %d/%d failed (%s); retrying in %.1fs", dataset, attempt, self.max_attempts, last_err, wait)
                self._sleep(wait)
        raise IngestError(f"[{dataset}] gave up after {self.max_attempts} attempts; last error: {last_err}; params={params}")

    def get_all(self, dataset: str, params: dict) -> list[dict]:
        """Paginate with $limit/$offset until a short page is returned."""
        rows: list[dict] = []
        offset = 0
        while True:
            page = self.get(dataset, {**params, "$limit": self.page_size, "$offset": offset})
            rows.extend(page)
            if len(page) < self.page_size:
                return rows
            offset += self.page_size


# ---------------------------------------------------------------------------
# Cleaning: dedupe
# ---------------------------------------------------------------------------
def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows on the row key (fecha_corte, codigo_negocio, tipo_participacion).

    Same key with a *different* unit value is a data conflict → ``DataQualityError``.
    """
    rows_in = len(df)
    dup_key = df.duplicated(KEY_COLS, keep=False)
    if dup_key.any():
        nunique_vu = df[dup_key].groupby(KEY_COLS)["valor_unidad_operaciones"].nunique()
        conflicts = nunique_vu[nunique_vu > 1]
        if len(conflicts):
            raise DataQualityError(
                f"conflicting duplicates (same key, different valor_unidad) for {len(conflicts)} keys, "
                f"e.g. {conflicts.head(5).index.tolist()}"
            )
    out = df.drop_duplicates(KEY_COLS, keep="first").reset_index(drop=True)
    removed = df[df.duplicated(KEY_COLS, keep="first")]
    by_code = {int(k): int(v) for k, v in removed.groupby("codigo_negocio").size().items()}
    stats = {"rows_in": rows_in, "rows_out": len(out), "duplicates_removed": rows_in - len(out), "by_code": by_code}
    return out, stats


# ---------------------------------------------------------------------------
# Cleaning: code-chain splicing (D1)
# ---------------------------------------------------------------------------
def splice_codes(df: pd.DataFrame, fund: Fund, *, tolerance: float, max_gap_days: int = 1) -> tuple[pd.DataFrame, list[dict]]:
    """Concatenate the histories of ``fund.codes`` (in order) into one series.

    For each consecutive pair (A → B) the splice is accepted only if:
    * B starts strictly after A ends (no overlap),
    * the gap is at most ``max_gap_days`` calendar days (Fase 0: 69/71 handoffs are exactly 1 day),
    * |VU_B_first / VU_A_last − 1| ≤ ``tolerance`` (unit-value continuity).
    Anything else raises ``DataQualityError``. Every applied splice is returned as a mapping record
    so the manifest can document it — nothing is applied without a trace.
    """
    parts: list[pd.DataFrame] = []
    for code in fund.codes:
        part = df[df["codigo_negocio"] == code].sort_values("fecha_corte")
        if part.empty:
            raise DataQualityError(f"[{fund.fund_id}] no rows for codigo_negocio={code} (clase {fund.clase})")
        parts.append(part)
    mappings: list[dict] = []
    for prev, nxt in zip(parts, parts[1:]):
        a_last, b_first = prev.iloc[-1], nxt.iloc[0]
        gap = (b_first["fecha_corte"] - a_last["fecha_corte"]).days
        if gap <= 0:
            raise DataQualityError(
                f"[{fund.fund_id}] codes {a_last['codigo_negocio']}→{b_first['codigo_negocio']} overlap in time "
                f"(A ends {a_last['fecha_corte'].date()}, B starts {b_first['fecha_corte'].date()})"
            )
        if gap > max_gap_days:
            raise DataQualityError(
                f"[{fund.fund_id}] splice gap of {gap} days between {a_last['codigo_negocio']} "
                f"({a_last['fecha_corte'].date()}) and {b_first['codigo_negocio']} ({b_first['fecha_corte'].date()}) > {max_gap_days}"
            )
        vu_a, vu_b = float(a_last["valor_unidad_operaciones"]), float(b_first["valor_unidad_operaciones"])
        rel = vu_b / vu_a - 1.0
        if abs(rel) > tolerance:
            raise DataQualityError(
                f"[{fund.fund_id}] unit-value continuity check failed at splice "
                f"{a_last['codigo_negocio']}→{b_first['codigo_negocio']}: VU {vu_a} → {vu_b} ({rel:+.2%} > ±{tolerance:.0%})"
            )
        mappings.append({
            "fund_id": fund.fund_id,
            "from_code": int(a_last["codigo_negocio"]),
            "to_code": int(b_first["codigo_negocio"]),
            "last_date_from": str(a_last["fecha_corte"].date()),
            "first_date_to": str(b_first["fecha_corte"].date()),
            "vu_from": vu_a,
            "vu_to": vu_b,
            "rel_diff": round(rel, 8),
            "gap_days": int(gap),
        })
    out = pd.concat(parts, ignore_index=True)
    return out, mappings


# ---------------------------------------------------------------------------
# Cleaning: Colombian business-day calendar + price alignment (D8)
# ---------------------------------------------------------------------------
def business_calendar(start, end) -> pd.DatetimeIndex:
    """Mon–Fri excluding Colombian public holidays (``holidays.CO``)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    days = pd.bdate_range(start, end)
    co = holidays.CO(years=range(start.year, end.year + 1))
    mask = [d.date() not in co for d in days]
    return pd.DatetimeIndex(days[mask], name="fecha")


def build_price_matrix(series: dict[str, pd.Series], calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Align each fund's calendar-daily unit-value series to the business calendar.

    Weekend/holiday observations are dropped (their accrual is embedded in the next business day's
    unit value). Business days without an observation stay NaN — no forward fill here.
    """
    return pd.DataFrame({fid: s.reindex(calendar) for fid, s in series.items()}, index=calendar)


# ---------------------------------------------------------------------------
# Cleaning: log returns with explicit NaN policy
# ---------------------------------------------------------------------------
NAN_POLICY = (
    "returns[t] = log(VU[t] / VU[prev]) where prev is the most recent *observed* business day. "
    "(a) Business days before a fund's first observation or after its last are NaN (fund not in universe; "
    "point-in-time membership is handled downstream). "
    "(b) Business days inside a fund's life with no observation are NaN — no imputation. "
    "(c) On the first observed day after such a gap the return bridges the whole gap "
    "(cumulative log return since the last observation) so the NAV path stays continuous; every bridged "
    "cell is listed in manifest['gap_log'] with the number of missing business days so downstream can mask it. "
    "(d) Weekend/holiday accrual is folded into the next business day's return (not a gap)."
)


def compute_returns(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Log returns over consecutive observed business days + a log of bridged gaps (policy above)."""
    calendar = prices.index
    rets = pd.DataFrame(np.nan, index=calendar, columns=prices.columns, dtype="float64")
    gap_log: list[dict] = []
    for fid in prices.columns:
        obs = prices[fid].dropna()
        if len(obs) < 2:
            continue
        r = np.log(obs / obs.shift(1))
        rets.loc[obs.index, fid] = r.values
        pos = calendar.get_indexer(obs.index)
        jumps = np.diff(pos)
        for i in np.flatnonzero(jumps > 1):
            gap_log.append({
                "fund_id": fid,
                "date": str(obs.index[i + 1].date()),
                "prev_date": str(obs.index[i].date()),
                "missing_bdays": int(jumps[i] - 1),
                "bridged_log_return": float(r.iloc[i + 1]),
            })
    return rets, gap_log


# ---------------------------------------------------------------------------
# Sufficiency gate: the approved universe must match what Fase 0 measured
# ---------------------------------------------------------------------------
def check_sufficiency(series: dict[str, pd.Series], funds: list[Fund], cfg: dict, *, data_end: pd.Timestamp) -> list[dict]:
    """Per-fund checks against the Fase 0 evidence encoded in ``universe.json``:

    * first observation ≤ ``inicio_esperado`` + ``max_start_delay_days``
    * last observation ≥ ``data_end`` − ``max_end_lag_days`` (fund still reporting)
    * completeness (obs / calendar days between first and last) ≥ ``min_completeness``
    * longest hole between consecutive observations ≤ ``max_gap_days``

    Returns the full report. If any fund fails, raises ``InsufficientDataError`` naming *every*
    failing fund (with the report attached) — the caller must stop and escalate, not substitute.
    """
    report: list[dict] = []
    for f in funds:
        s = series[f.fund_id].dropna().sort_index()
        first, last = s.index[0], s.index[-1]
        span_days = (last - first).days + 1
        completeness = len(s) / span_days
        max_gap = int((s.index.to_series().diff().dt.days.fillna(1) - 1).max())
        failures = []
        if first > f.inicio_esperado + pd.Timedelta(days=cfg["max_start_delay_days"]):
            failures.append(f"starts {first.date()} but Fase 0 expected ≤ {f.inicio_esperado.date()} (+{cfg['max_start_delay_days']}d)")
        if last < data_end - pd.Timedelta(days=cfg["max_end_lag_days"]):
            failures.append(f"last observation {last.date()} < data end {data_end.date()} (−{cfg['max_end_lag_days']}d): stopped reporting")
        if completeness < cfg["min_completeness"]:
            failures.append(f"completeness {completeness:.2%} < {cfg['min_completeness']:.0%}")
        if max_gap > cfg["max_gap_days"]:
            failures.append(f"max gap {max_gap}d > {cfg['max_gap_days']}d")
        report.append({
            "fund_id": f.fund_id,
            "first_date": str(first.date()),
            "last_date": str(last.date()),
            "n_obs": int(len(s)),
            "completeness": round(float(completeness), 6),
            "max_gap_days": max_gap,
            "expected_start": str(f.inicio_esperado.date()),
            "ok": not failures,
            "failures": failures,
        })
    failing = [r for r in report if not r["ok"]]
    if failing:
        lines = "; ".join(f"{r['fund_id']}: {' | '.join(r['failures'])}" for r in failing)
        raise InsufficientDataError(
            f"{len(failing)} fund(s) of the approved universe contradict Fase 0 evidence — halting, no substitution. {lines}",
            report,
        )
    return report


# ---------------------------------------------------------------------------
# Raw layer: store what the API returns, untouched
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "0.1.0"


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_frame(rows: list[dict]) -> pd.DataFrame:
    """Rows exactly as delivered (every column, every value a string)."""
    df = pd.DataFrame(rows)
    return df.astype(object) if len(df) else df


def _series_query(code: int) -> dict:
    return {"$where": f"codigo_negocio={code}", "$order": "fecha_corte,tipo_participacion"}


def _metadata_query(code: int, clase: int) -> dict:
    return {"$where": f"codigo_negocio={code} AND tipo_participacion={clase}", "$order": "fecha_corte DESC", "$limit": 1}


def fetch_raw(client: SodaClient, universe: Universe, raw_dir: Path) -> dict[str, dict]:
    """Download the full history of every ``codigo_negocio`` in the universe (all classes, all
    columns) from the series dataset, plus the latest row per fund from the metadata dataset.

    Returns ``{filename: {dataset, query_params, page_size, pages, rows}}`` for the manifest.
    A code with zero rows halts the pipeline (that fund no longer exists in the API).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    series_ds, meta_ds = universe.datasets["series"], universe.datasets["metadata"]
    entries: dict[str, dict] = {}
    for code in universe.all_codes:
        params = _series_query(code)
        log.info("[%s] fetching codigo_negocio=%s", series_ds, code)
        rows = client.get_all(series_ds, params)
        if not rows:
            raise IngestError(f"[{series_ds}] codigo_negocio={code} returned no rows — fund missing from API; halting")
        name = f"{series_ds}_{code}.parquet"
        _raw_frame(rows).to_parquet(raw_dir / name, index=False)
        pages = -(-len(rows) // client.page_size) or 1
        entries[name] = {"dataset": series_ds, "query_params": params, "page_size": client.page_size,
                         "pages": pages, "rows": len(rows)}
    meta_rows: list[dict] = []
    meta_params: list[dict] = []
    for f in universe.funds:
        code = f.codes[-1]                         # current code carries the current metadata
        params = _metadata_query(code, f.clase)
        rows = client.get(meta_ds, params)
        meta_params.append(params)
        meta_rows.extend(rows)
    name = f"{meta_ds}_metadata.parquet"
    _raw_frame(meta_rows).to_parquet(raw_dir / name, index=False)
    entries[name] = {"dataset": meta_ds, "query_params": meta_params, "page_size": None, "pages": 1, "rows": len(meta_rows)}
    return entries


def load_raw(universe: Universe, raw_dir: Path) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, dict[str, dict]]:
    """Read the raw parquet files back (strings) and rebuild manifest entries from the same query builders."""
    raw_dir = Path(raw_dir)
    series_ds, meta_ds = universe.datasets["series"], universe.datasets["metadata"]
    frames: dict[int, pd.DataFrame] = {}
    entries: dict[str, dict] = {}
    for code in universe.all_codes:
        name = f"{series_ds}_{code}.parquet"
        path = raw_dir / name
        if not path.exists():
            raise IngestError(f"raw file missing: {path} (run without --skip-download)")
        frames[code] = pd.read_parquet(path)
        entries[name] = {"dataset": series_ds, "query_params": _series_query(code), "page_size": None,
                         "pages": None, "rows": len(frames[code]), "reused": True}
    name = f"{meta_ds}_metadata.parquet"
    meta = pd.read_parquet(raw_dir / name)
    entries[name] = {"dataset": meta_ds, "query_params": [_metadata_query(f.codes[-1], f.clase) for f in universe.funds],
                     "page_size": None, "pages": 1, "rows": len(meta), "reused": True}
    return frames, meta, entries


# ---------------------------------------------------------------------------
# Cleaning orchestration
# ---------------------------------------------------------------------------
@dataclass
class CleanResult:
    prices: pd.DataFrame
    returns: pd.DataFrame
    funds_master: pd.DataFrame
    dedupe_stats: dict
    quality: dict
    code_mappings: list[dict]
    gap_log: list[dict]
    sufficiency_report: list[dict]
    metadata_crosscheck: list[dict]
    data_end: pd.Timestamp
    calendar: pd.DatetimeIndex


def build_clean(raw_frames: dict[int, pd.DataFrame], raw_meta: pd.DataFrame, universe: Universe) -> CleanResult:
    parsed = parse_soda_rows([r for df in raw_frames.values() for r in df.to_dict("records")], strict_vu=False)
    parsed, dedupe_stats = dedupe(parsed)
    data_end = parsed["fecha_corte"].max()
    tol = universe.sufficiency["splice_tolerance"]

    # Invalid unit values are fatal only for the (code, class) pairs the universe uses; the rest are
    # dropped and listed in the manifest (e.g. a class reporting VU=0 on its launch day).
    used = {(c, f.clase) for f in universe.funds for c in f.codes}
    bad = _invalid_vu_mask(parsed)
    bad_used = bad & parsed.apply(lambda r: (r["codigo_negocio"], r["tipo_participacion"]) in used, axis=1)
    if bad_used.any():
        rows = parsed.loc[bad_used, KEY_COLS]
        fid = next(f.fund_id for f in universe.funds if int(rows.iloc[0]["codigo_negocio"]) in f.codes)
        raise DataQualityError(f"[{fid}] valor_unidad_operaciones missing or <= 0 in {int(bad_used.sum())} rows of a "
                               f"selected class, e.g. {rows.head(5).to_dict('records')}")
    ignored = [
        {"codigo_negocio": int(code), "tipo_participacion": int(clase), "n": int(len(g)),
         "dates": [str(d.date()) for d in g["fecha_corte"].head(10)]}
        for (code, clase), g in parsed[bad].groupby(["codigo_negocio", "tipo_participacion"])
    ]
    parsed = parsed[~bad]
    quality = {"invalid_vu_rows_ignored": ignored}

    series: dict[str, pd.Series] = {}
    mappings: list[dict] = []
    master_rows: list[dict] = []
    for f in universe.funds:
        sub = parsed[parsed["tipo_participacion"] == f.clase]
        spliced, m = splice_codes(sub, f, tolerance=tol)
        mappings.extend(m)
        s = spliced.set_index("fecha_corte")["valor_unidad_operaciones"].sort_index()
        if s.index.has_duplicates:                     # cannot happen after dedupe + no-overlap splice, but never trust
            raise DataQualityError(f"[{f.fund_id}] duplicate dates after splice")
        series[f.fund_id] = s
        master_rows.append({
            "fund_id": f.fund_id, "cat": f.cat, "rol": f.rol, "clase": f.clase,
            "codes": ">".join(str(c) for c in f.codes), "nombre": f.nombre, "admin": f.admin,
            "inicio_esperado": f.inicio_esperado, "first_date": s.index[0], "last_date": s.index[-1], "n_obs": int(len(s)),
        })

    report = check_sufficiency(series, universe.funds, universe.sufficiency, data_end=data_end)
    calendar = business_calendar(universe.start_date, data_end)
    prices = build_price_matrix(series, calendar)
    returns, gap_log = compute_returns(prices)

    master = pd.DataFrame(master_rows)
    meta = parse_soda_rows(raw_meta.to_dict("records"))
    crosscheck = _crosscheck_metadata(meta, series, universe)
    if len(meta):
        latest = meta.sort_values("fecha_corte").groupby("codigo_negocio").tail(1).set_index("codigo_negocio")
        code_of = {f.fund_id: f.codes[-1] for f in universe.funds}
        for col in ("nombre_patrimonio", "nombre_entidad", "nombre_subtipo_patrimonio", "numero_inversionistas"):
            if col in latest.columns:
                master[f"api_{col}"] = [latest[col].get(code_of[fid]) for fid in master["fund_id"]]
    return CleanResult(prices, returns, master, dedupe_stats, quality, mappings, gap_log, report, crosscheck, data_end, calendar)


def _crosscheck_metadata(meta: pd.DataFrame, series: dict[str, pd.Series], universe: Universe) -> list[dict]:
    """The metadata dataset is a mirror of the series dataset (Fase 0, H1): its latest unit value per
    fund must equal the series value on the same date. Mismatch = inconsistent sources → halt."""
    if not len(meta):
        return []
    out: list[dict] = []
    for f in universe.funds:
        rows = meta[(meta["codigo_negocio"] == f.codes[-1]) & (meta["tipo_participacion"] == f.clase)]
        if rows.empty:
            out.append({"fund_id": f.fund_id, "status": "absent_in_metadata_dataset"})
            continue
        r = rows.sort_values("fecha_corte").iloc[-1]
        d, vu_b = r["fecha_corte"], float(r["valor_unidad_operaciones"])
        vu_a = series[f.fund_id].get(d)
        rec = {"fund_id": f.fund_id, "date": str(d.date()), "vu_series": None if vu_a is None else float(vu_a), "vu_metadata": vu_b}
        if vu_a is None:
            rec["status"] = "date_missing_in_series"
        elif abs(vu_b / vu_a - 1) > 1e-9:
            raise DataQualityError(f"[{f.fund_id}] metadata dataset VU {vu_b} != series VU {vu_a} on {d.date()}")
        else:
            rec["status"] = "match"
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Outputs + manifest
# ---------------------------------------------------------------------------
def _file_entry(path: Path, **extra) -> dict:
    return {"sha256": sha256_of(path), "bytes": path.stat().st_size, **extra}


def write_outputs(result: CleanResult, cleaned_dir: Path) -> dict[str, dict]:
    cleaned_dir = Path(cleaned_dir)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, df in (("returns_matrix.parquet", result.returns), ("prices_matrix.parquet", result.prices)):
        path = cleaned_dir / name
        df.to_parquet(path)
        entries[name] = _file_entry(path, rows=len(df), columns=list(df.columns),
                                    index=f"{df.index[0].date()}..{df.index[-1].date()}")
    path = cleaned_dir / "funds_master.parquet"
    result.funds_master.to_parquet(path, index=False)
    entries["funds_master.parquet"] = _file_entry(path, rows=len(result.funds_master), columns=list(result.funds_master.columns))
    return entries


def build_manifest(universe: Universe, client: SodaClient, raw_entries: dict, cleaned_entries: dict,
                   result: CleanResult, root: Path) -> dict:
    root = Path(root)
    raw = {}
    for name, e in raw_entries.items():
        path = root / "data" / "raw" / name
        raw[str(path.relative_to(root))] = {**e, **_file_entry(path)}
    cleaned = {str((root / "data" / "cleaned" / n).relative_to(root)): e for n, e in cleaned_entries.items()}
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline": {"module": "data/ingest_sfc.py", "version": PIPELINE_VERSION,
                     "python": sys.version.split()[0], "pandas": pd.__version__, "holidays": holidays.__version__},
        "source": {"domain": universe.domain, "datasets": universe.datasets, "base_url": client.base_url,
                   "app_token_used": client.app_token_used,
                   "retry_policy": {"max_attempts": client.max_attempts, "base_wait_s": client.base_wait,
                                    "max_wait_s": client.max_wait, "timeout_s": client.timeout,
                                    "retry_on": ["Timeout", "ConnectionError"] + [f"HTTP {s}" for s in sorted(RETRYABLE_STATUS)]}},
        "universe": {"config_path": str(universe.config_path), "sha256": sha256_of(universe.config_path),
                     "n_funds": len(universe.funds), "fund_ids": [f.fund_id for f in universe.funds],
                     "start_date": str(universe.start_date.date())},
        "raw": raw,
        "cleaned": cleaned,
        "dedupe": {"key": KEY_COLS, **result.dedupe_stats},
        "quality": result.quality,
        "code_mappings": result.code_mappings,
        "calendar": {"type": "Colombian business days: Mon-Fri excluding holidays.CO", "holidays_version": holidays.__version__,
                     "start": str(result.calendar[0].date()), "end": str(result.calendar[-1].date()), "n_days": int(len(result.calendar)),
                     "data_end": str(result.data_end.date())},
        "returns": {"formula": "log(VU_t / VU_prev_observed_business_day)", "source_field": "valor_unidad_operaciones",
                    "class_selection": "tipo_participacion per fund from universe.json", "nan_policy": NAN_POLICY},
        "gap_log": result.gap_log,
        "sufficiency": {"thresholds": universe.sufficiency, "data_end": str(result.data_end.date()), "report": result.sufficiency_report},
        "metadata_crosscheck": result.metadata_crosscheck,
    }


def run(universe: Universe | Path | str, root: Path | str, *, client: SodaClient | None = None,
        skip_download: bool = False) -> dict:
    """Full pipeline. Writes ``<root>/data/raw``, ``<root>/data/cleaned`` and ``<root>/manifest.json``.

    On ``InsufficientDataError`` nothing under ``cleaned/`` and no manifest is written (raw is kept).
    """
    if not isinstance(universe, Universe):
        universe = load_universe(universe)
    root = Path(root)
    raw_dir = root / "data" / "raw"
    if client is None:
        client = SodaClient(universe.domain, app_token=os.environ.get("SOCRATA_APP_TOKEN") or None)
    if skip_download:
        raw_frames, raw_meta, raw_entries = load_raw(universe, raw_dir)
    else:
        raw_entries = fetch_raw(client, universe, raw_dir)
        raw_frames, raw_meta, _ = load_raw(universe, raw_dir)
    result = build_clean(raw_frames, raw_meta, universe)
    cleaned_entries = write_outputs(result, root / "data" / "cleaned")
    manifest = build_manifest(universe, client, raw_entries, cleaned_entries, result, root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("wrote %s", root / "manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--universe", default="data/universe.json")
    ap.add_argument("--root", default=".", help="repo root (outputs go to <root>/data/{raw,cleaned} and <root>/manifest.json)")
    ap.add_argument("--skip-download", action="store_true", help="reuse data/raw/*.parquet instead of calling the API")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        m = run(args.universe, args.root, skip_download=args.skip_download)
    except InsufficientDataError as exc:
        log.error("HALTED: %s", exc)
        for r in exc.report:
            if not r["ok"]:
                log.error("  %s: %s", r["fund_id"], " | ".join(r["failures"]))
        return 2
    except IngestError as exc:
        log.error("FAILED: %s", exc)
        return 1
    log.info("done: %d funds, calendar %s..%s, %d gap cells, %d code mappings",
             m["universe"]["n_funds"], m["calendar"]["start"], m["calendar"]["end"], len(m["gap_log"]), len(m["code_mappings"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
