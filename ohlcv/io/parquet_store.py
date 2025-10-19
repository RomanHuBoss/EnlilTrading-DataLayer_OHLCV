# ohlcv/io/parquet_store.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..version import __version__

__all__ = ["parquet_path", "write_idempotent"]


# =============================
# Путь хранения
# =============================

def parquet_path(root: Path | str, symbol: str, tf: str) -> Path:
    """Возвращает путь до parquet-файла: <root>/<symbol>/<tf>.parquet."""
    return Path(root) / symbol / f"{tf}.parquet"


# =============================
# Нормализация входных данных
# =============================
_DEF_ORDER = ["ts", "o", "h", "l", "c", "v", "t", "is_gap"]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Допускаем DatetimeIndex → ts(ms)
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        idx = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
        out["ts"] = (idx.view("int64") // 1_000_000).astype("int64")
    if "ts" not in out.columns:
        raise ValueError("ожидается колонка 'ts' (ms) или DatetimeIndex")

    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")

    for c in ["o", "h", "l", "c", "v", "t"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "is_gap" in out.columns:
        out["is_gap"] = out["is_gap"].astype(bool)

    # Сортировка и дедуп по ts (последний выигрывает)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    # Порядок колонок: только существующие из канона
    cols = [c for c in _DEF_ORDER if c in out.columns]
    out = out[cols]
    return out


# =============================
# Хэш данных и метаданные
# =============================

def _hash_df(df: pd.DataFrame) -> str:
    cols = [c for c in _DEF_ORDER if c in df.columns]
    h = blake2b(digest_size=20)
    for c in cols:
        v = df[c].to_numpy()
        if v.dtype == "float64":
            # стабильное представление
            h.update(np.nan_to_num(v, nan=np.nan).astype("float64").tobytes())
        elif v.dtype == "int64":
            h.update(v.astype("int64").tobytes())
        elif v.dtype == "bool":
            h.update(v.astype("uint8").tobytes())
        else:
            h.update(pd.Series(v).astype(str).str.encode("utf-8").sum())
    return h.hexdigest()


def _build_signature() -> dict:
    keys = [
        "GIT_COMMIT",
        "CI_COMMIT_SHA",
        "SOURCE_VERSION",
        "PYPROJECT_SHA",
        "GIT_DIRTY",
    ]
    env = {k: os.environ.get(k) for k in keys if os.environ.get(k)}
    return {"pkg": "ohlcv", "version": __version__, "env": env}


# =============================
# Запись idempotent
# =============================

def write_idempotent(
    root: Path | str,
    symbol: str,
    tf: str,
    df_new: pd.DataFrame,
    *,
    compression: str | None = "zstd",
    zstd_level: int = 7,
    row_group_size: int = 256_000,
    source: str = "bybit",
) -> Path:
    """Идемпотентно записывает данные в store, мерж по ts, возвращает путь к файлу.

    - Если файл существует: конкатенация и дедуп по ts (последний выигрывает).
    - Метаданные footer: ключ `c1.meta` с JSON {source,symbol,tf,rows,data_hash,build_signature}.
    - Атомарная запись через временный файл.
    """
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)

    new_norm = _normalize_df(df_new)

    # Загрузка существующего
    if path.exists():
        try:
            df_old = pd.read_parquet(path)
            old_norm = _normalize_df(df_old)
            merged = pd.concat([old_norm, new_norm], ignore_index=True)
            merged = merged.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
        except Exception:
            # fallback: писать только новые, если старый файл повреждён
            merged = new_norm
    else:
        merged = new_norm

    # Итоговый порядок колонок по канону
    cols = [c for c in _DEF_ORDER if c in merged.columns]
    merged = merged[cols]

    # Метаданные
    meta = {
        "source": source,
        "symbol": symbol,
        "tf": tf,
        "rows": int(len(merged)),
        "data_hash": _hash_df(merged),
        "build_signature": _build_signature(),
    }

    # Атомарная запись
    tmp = path.with_suffix(path.suffix + ".tmp")

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

        table = pa.Table.from_pandas(merged, preserve_index=False)
        # Добавить/сохранить метаданные схемы
        md = dict(table.schema.metadata or {})
        md[b"c1.meta"] = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        table = table.replace_schema_metadata(md)

        # Параметры компрессии
        if compression == "zstd":
            comp = pq.ParquetWriter.write_table
        # Запись
        pq.write_table(
            table,
            tmp,
            compression=(compression or None),
            version="2.6",
            use_dictionary=True,
            data_page_size=1 << 20,
            write_statistics=True,
            coerce_timestamps="ms",
            allow_truncated_timestamps=True,
            row_group_size=row_group_size,
        )
    except Exception:
        # Фоллбек без явной меты
        if compression in (None, "none"):
            merged.to_parquet(tmp)
        else:
            merged.to_parquet(tmp, compression=compression)

    os.replace(tmp, path)
    return path
