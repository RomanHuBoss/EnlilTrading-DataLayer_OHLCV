# ohlcv/io/tail_cache.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class TailInfo:
    symbol: str
    tf: str
    latest_ts_ms: Optional[int]
    rows_total: Optional[int]
    data_hash: Optional[str]
    updated_at: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sidecar_path(root: Path | str, symbol: str, tf: str) -> Path:
    return Path(root).expanduser().resolve() / symbol / f"{tf}.latest.json"


def read(root: Path | str, symbol: str, tf: str) -> TailInfo:
    p = _sidecar_path(root, symbol, tf)
    if not p.exists():
        return TailInfo(symbol=symbol, tf=tf, latest_ts_ms=None, rows_total=None, data_hash=None, updated_at=_now_iso())
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
        return TailInfo(
            symbol=str(blob.get("symbol", symbol)),
            tf=str(blob.get("tf", tf)),
            latest_ts_ms=blob.get("latest_ts_ms"),
            rows_total=blob.get("rows_total"),
            data_hash=blob.get("data_hash"),
            updated_at=str(blob.get("updated_at", _now_iso())),
        )
    except Exception:
        # повреждённый sidecar — вернём пустой
        return TailInfo(symbol=symbol, tf=tf, latest_ts_ms=None, rows_total=None, data_hash=None, updated_at=_now_iso())


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def write(root: Path | str, info: TailInfo) -> Path:
    p = _sidecar_path(root, info.symbol, info.tf)
    _atomic_write_json(p, asdict(info))
    return p


def update(root: Path | str, symbol: str, tf: str, *, latest_ts_ms: Optional[int], rows_total: Optional[int], data_hash: Optional[str]) -> TailInfo:
    info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=latest_ts_ms, rows_total=rows_total, data_hash=data_hash, updated_at=_now_iso())
    write(root, info)
    return info


def refresh_from_parquet(root: Path | str, symbol: str, tf: str) -> TailInfo:
    """Обновляет sidecar, читая только столбец ts из parquet."""
    from .parquet_store import parquet_path  # локальный импорт, чтобы избежать циклов

    p = parquet_path(root, symbol, tf)
    if not Path(p).exists():
        info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=None, rows_total=None, data_hash=None, updated_at=_now_iso())
        write(root, info)
        return info
    # чтение только индекса ts для скорости
    df = pd.read_parquet(p, columns=["ts"])  # type: ignore[arg-type]
    if len(df) == 0:
        info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=None, rows_total=0, data_hash=None, updated_at=_now_iso())
        write(root, info)
        return info
    ts = pd.to_datetime(df["ts"], utc=True)
    latest = int(ts.iloc[-1].value // 1_000_000)
    info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=latest, rows_total=int(len(df)), data_hash=None, updated_at=_now_iso())
    write(root, info)
    return info


def update_from_df(root: Path | str, symbol: str, tf: str, df: pd.DataFrame, *, data_hash: Optional[str] = None) -> TailInfo:
    """Обновляет sidecar по свежезаписанному df (индекс tz-aware DatetimeIndex)."""
    if df.empty:
        # не перезаписываем хвост пустым
        return read(root, symbol, tf)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise TypeError("ожидается tz-aware DatetimeIndex")
    latest = int(df.index[-1].value // 1_000_000)
    # rows_total получить невозможно без чтения всего файла; оставляем None
    info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=latest, rows_total=None, data_hash=data_hash, updated_at=_now_iso())
    write(root, info)
    return info
