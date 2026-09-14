"""Carga de datos y recorte de ventana para el dashboard (Fase 4).

Reutiliza la semántica point-in-time del backtest (``active_universe`` / ``estimation_window``) para que la
ventana que ve el dashboard sea exactamente la que vería el walk-forward en esa fecha.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from backtest.walk_forward import BacktestConfig, BacktestError, active_universe, estimation_window, rebalance_dates

RETURNS_PATH = Path("data/cleaned/returns_matrix.parquet")
FUNDS_MASTER_PATH = Path("data/cleaned/funds_master.parquet")
LOOKBACK_OPTIONS: tuple[int, ...] = (12, 24, 36)
DEFAULT_LOOKBACK = 24  # D8


def load_returns(path: Path | str = RETURNS_PATH) -> pd.DataFrame:
    """Matriz de retornos log diarios (índice DatetimeIndex ordenado, columnas = fund_id)."""
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_labels(path: Path | str = FUNDS_MASTER_PATH) -> dict[str, str]:
    """``fund_id -> nombre`` desde ``funds_master``; vacío si el archivo no existe (tests / entornos sin ingest)."""
    if not Path(path).exists():
        return {}
    fm = pd.read_parquet(path, columns=["fund_id", "nombre"])
    return dict(zip(fm["fund_id"], fm["nombre"]))


def fingerprint(returns: pd.DataFrame, path: Path | str | None = None) -> str:
    """16 hex de sha256 del parquet (si existe) o del contenido del DataFrame. Forma parte de la clave de cache:
    un re-ingest cambia el fingerprint y por tanto la clave. ``returns``/``fingerprint`` se calculan una sola
    vez en ``load_data`` al arrancar el proceso, así que un re-ingest en caliente no invalida nada mientras
    el proceso sigue vivo; lo que sí garantiza es que, tras reiniciar el proceso después de un re-ingest, un
    backend persistente (``FileSystemCache``) nunca sirva un bundle calculado con el parquet anterior."""
    h = hashlib.sha256()
    if path is not None and Path(path).exists():
        h.update(Path(path).read_bytes())
    else:
        h.update(pd.util.hash_pandas_object(returns, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, returns.columns)).encode("utf-8"))
    return h.hexdigest()[:16]


def _config(index: pd.DatetimeIndex, lookback_months: int) -> BacktestConfig:
    return BacktestConfig(estimation_start=str(index[0].date()), window_months=lookback_months)


def asof_options(returns: pd.DataFrame, lookback_months: int) -> list[pd.Timestamp]:
    """Fechas de decisión del backtest (último día hábil de cada trimestre) con universo activo suficiente
    (``len(active_universe(...)) >= cfg.min_active_funds``) más la última fecha disponible. Se filtra sobre
    ``returns`` (y no solo el índice) porque la primera fecha de ``rebalance_dates`` cae justo en el borde de
    ``lookback`` meses de historia, donde ningún fondo la cumple todavía; devolverla dejaría el primer as-of
    de cualquier lookback roto en ``window_slice``."""
    index = returns.index
    cfg = _config(index, lookback_months)
    try:
        candidates = rebalance_dates(index, cfg)
    except BacktestError:
        candidates = []
    dates = [d for d in candidates if len(active_universe(returns, d, cfg)) >= cfg.min_active_funds]
    last = index[-1]
    return dates + [last] if not dates or dates[-1] != last else dates


@dataclass(frozen=True)
class Window:
    universe: tuple[str, ...]
    frame: pd.DataFrame  # (asof - lookback, asof], soporte común, NaN → 0 (política del backtest)

    @property
    def start(self) -> str:
        return str(self.frame.index[0].date())

    @property
    def end(self) -> str:
        return str(self.frame.index[-1].date())


def window_slice(returns: pd.DataFrame, asof: pd.Timestamp, lookback_months: int) -> Window:
    """Universo activo point-in-time y ventana de estimación en ``asof``. ``BacktestError`` si < 2 fondos."""
    cfg = _config(returns.index, lookback_months)
    universe = tuple(active_universe(returns, asof, cfg))
    if len(universe) < cfg.min_active_funds:
        raise BacktestError(f"solo {len(universe)} fondo(s) activo(s) en {asof.date()} con {lookback_months} m de historia "
                             f"(mínimo {cfg.min_active_funds})")
    frame = estimation_window(returns[list(universe)], asof, cfg)
    return Window(universe=universe, frame=frame)
