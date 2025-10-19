# ohlcv/io/tail_cache.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================
# Модель tail-сайдкара
# =============================
@dataclass(frozen=True)
class TailInfo:
    symbol: str
    tf: str
    latest_ts_ms: int
    updated_at_iso: str


# =============================
# Вспомогательные утилиты
# =============================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_dir(root: Path | str, symbol: str) -> Path:
    return Path(root) / symbol / "tail"


def _sidecar_path(root: Path | str, symbol: str, tf: str) -> Path:
    return _tail_dir(root, symbol) / f"{tf}.tail.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


# =============================
# API tail-сайдкара
# =============================

def write(root: Path | str, info: TailInfo) -> Path:
    """Пишет tail JSON атомарно и возвращает путь."""
    p = _sidecar_path(root, info.symbol, info.tf)
    _atomic_write_json(p, asdict(info))
    return p


def read(root: Path | str, symbol: str, tf: str) -> Optional[TailInfo]:
    """Читает tail JSON; возвращает TailInfo или None, если нет файла."""
    p = _sidecar_path(root, symbol, tf)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return TailInfo(
            symbol=obj.get("symbol", symbol),
            tf=obj.get("tf", tf),
            latest_ts_ms=int(obj["latest_ts_ms"]),
            updated_at_iso=str(obj.get("updated_at_iso") or _now_iso()),
        )
    except Exception:
        return None


def update(
    root: Path | str,
    symbol: str,
    tf: str,
    *,
    latest_ts_ms: Optional[int] = None,
) -> TailInfo:
    """Обновляет tail: монотонно повышает latest_ts_ms и пишет ISO-время обновления."""
    cur = read(root, symbol, tf)
    if cur is None:
        if latest_ts_ms is None:
            raise ValueError("latest_ts_ms обязателен при первом создании tail")
        info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=int(latest_ts_ms), updated_at_iso=_now_iso())
        write(root, info)
        return info
    new_ts = cur.latest_ts_ms if latest_ts_ms is None else max(int(latest_ts_ms), int(cur.latest_ts_ms))
    info = TailInfo(symbol=symbol, tf=tf, latest_ts_ms=new_ts, updated_at_iso=_now_iso())
    write(root, info)
    return info


# =============================
# Parquet-хвост последних N дней
# =============================

def write_parquet_tail(
    root: Path | str,
    symbol: str,
    tf: str,
    df: pd.DataFrame,
    *,
    days: int = 14,
) -> Path:
    """Пишет tail/<tf>.tail.parquet с последними N днями данных.

    Ожидает колонку 'ts' (мс). Если индекс DatetimeIndex — будет использован он.
    """
    out_dir = _tail_dir(root, symbol)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tf}.tail.parquet"

    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], unit="ms", utc=True)
        sub = df.copy()
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        sub = df.reset_index().rename(columns={df.index.name or "index": "ts"}).copy()
        sub["ts"] = (ts.view("int64") // 1_000_000).astype("int64")
    else:
        raise ValueError("ожидается колонка 'ts' или DatetimeIndex")

    cutoff = ts.max() - pd.Timedelta(days=days)
    mask = ts >= cutoff
    sub = sub.loc[mask].reset_index(drop=True)

    # запись parquet: предпочтительно через pyarrow
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pandas(sub, preserve_index=False)
        pq.write_table(table, path)
    except Exception:
        sub.to_parquet(path)

    # обновить JSON tail максимальным ts
    max_ts = int(sub["ts"].max()) if len(sub) else int((ts.max().value // 1_000_000))
    update(root, symbol, tf, latest_ts_ms=max_ts)

    return path
