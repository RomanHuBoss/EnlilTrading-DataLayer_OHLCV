# ohlcv/io/parquet_store.py
from __future__ import annotations

import json
import os
from hashlib import blake2b
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Версия пакета: устойчивый импорт с резервом
try:  # ohlcv/utils/version.py
    from ..utils.version import __version__
except Exception:
    try:  # ohlcv/version.py — резерв на случай иной структуры
        from ..version import __version__  # type: ignore
    except Exception:  # финальный fallback
        __version__ = "0.0.0"  # type: ignore

__all__ = ["parquet_path", "write_idempotent"]

# ---------------- Path ----------------

def parquet_path(root: Path | str, symbol: str, tf: str) -> Path:
    return Path(root) / str(symbol) / f"{tf}.parquet"

# -------------- Normalize --------------

# Жёсткий порядок поддерживаемых колонок
_DEF_ORDER = ["ts", "o", "h", "l", "c", "v", "t", "ver", "is_gap"]


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приведение к контракту C1:
    - ts:int64 (ms, UTC)
    - o/h/l/c/v:float64
    - t:float64 (опц.)
    - ver:int32 (опц.)
    - is_gap:bool (опц.)
    - сортировка и дедуп по ts (последний выигрывает)
    """
    out = df.copy()

    # DatetimeIndex → ts(ms, UTC)
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            idx: pd.DatetimeIndex = pd.DatetimeIndex(out.index)
            idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            out["ts"] = (idx.asi8 // 1_000_000).astype("int64")
        else:
            raise ValueError("ожидается колонка 'ts' (ms) или DatetimeIndex")

    # Типы
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ("o", "h", "l", "c", "v"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "t" in out.columns:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("float64")
    if "ver" in out.columns:
        out["ver"] = pd.to_numeric(out["ver"], errors="coerce").fillna(0).astype("int32")
    if "is_gap" in out.columns:
        out["is_gap"] = out["is_gap"].astype("bool")

    # Сортировка и дедуп по ts (последний выигрывает)
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # Фиксированный порядок колонок
    cols = [c for c in _DEF_ORDER if c in out.columns]
    out = out[cols]
    return out

# -------------- Hash/Meta --------------

def _hash_df(df: pd.DataFrame) -> str:
    """
    Детерминированный хэш набора строк по _DEF_ORDER∩columns.
    NaN канонизируются, bool → uint8. Независим от перестановки колонок.
    """
    cols = [c for c in _DEF_ORDER if c in df.columns]
    h = blake2b(digest_size=20)
    for c in cols:
        v = df[c].to_numpy()
        if v.dtype == np.float64:
            arr = np.nan_to_num(v, copy=False, nan=np.inf, posinf=np.inf, neginf=-np.inf).astype(np.float64)
            h.update(arr.tobytes())
        elif v.dtype == np.int64:
            h.update(v.astype(np.int64).tobytes())
        elif v.dtype == np.bool_:
            h.update(v.astype(np.uint8).tobytes())
        elif v.dtype == np.int32:
            h.update(v.astype(np.int32).tobytes())
        else:
            # Объектные/строковые — через стабильное строковое представление
            seq = ["" if x is None else str(x) for x in v.tolist()]
            b = ("".join(seq)).encode("utf-8")
            h.update(b)
    h.update(str(len(df)).encode("ascii"))
    h.update(("|" + "|".join(cols)).encode("utf-8"))
    return h.hexdigest()


def _build_signature() -> dict:
    keys = ["GIT_COMMIT", "CI_COMMIT_SHA", "SOURCE_VERSION", "PYPROJECT_SHA", "GIT_DIRTY"]
    env = {k: os.environ.get(k) for k in keys if os.environ.get(k)}
    return {"pkg": "ohlcv", "version": __version__, "env": env}

# -------------- Write (idempotent) --------------

def write_idempotent(
    root: Path | str,
    symbol: str,
    tf: str,
    df_new: pd.DataFrame,
    *,
    compression: Literal["snappy", "gzip", "brotli", "lz4", "zstd"] | None = "zstd",
    zstd_level: int = 7,
    row_group_size: int = 256_000,
    source: str = "bybit",
) -> Path:
    """
    Идемпотентная запись OHLCV в соответствии с C1:
      — merge со старым, дедуп по ts (последний выигрывает), сортировка.
      — отброс незакрытого текущего бара: ts ≤ floor(now,1m) − 60_000.
      — инкремент ver при перезаписи минутных баров.
      — footer-метаданные (ключ b"c1.meta") с {source,symbol,tf,rows,min_ts,max_ts,data_hash,build_signature}.
      — атомарная запись через временный файл + os.replace().
      — Parquet: pyarrow обязателен.
    """
    # Пути
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Нормализация нового блока
    new_norm = _normalize_df(df_new)

    # Отброс незакрытого бара (UTC, граница — начало текущей минуты минус 60s)
    cutoff_ms = int(pd.Timestamp.utcnow().floor("min").value // 1_000_000) - 60_000
    if len(new_norm):
        new_norm = new_norm[new_norm["ts"] <= cutoff_ms]

    # Загрузка и нормализация существующего
    if path.exists():
        df_old = pd.read_parquet(path)
        old_norm = _normalize_df(df_old)
    else:
        old_norm = pd.DataFrame(columns=[c for c in _DEF_ORDER if c != "is_gap"])
        dtypes = {
            "ts": "int64",
            "o": "float64",
            "h": "float64",
            "l": "float64",
            "c": "float64",
            "v": "float64",
            "t": "float64",
            "ver": "int32",
        }
        old_norm = old_norm.astype({k: v for k, v in dtypes.items() if k in old_norm.columns})

    # Гарантируем наличие 'ver'
    if "ver" not in old_norm.columns and len(old_norm):
        old_norm = old_norm.assign(ver=np.int32(0))
    if "ver" not in new_norm.columns and len(new_norm):
        new_norm = new_norm.assign(ver=np.int32(0))

    # Инкремент 'ver' для пересекающихся ts
    if len(new_norm) and len(old_norm):
        old_ver_map = old_norm.set_index("ts")["ver"] if "ver" in old_norm.columns else pd.Series(dtype="int32")
        common_mask = new_norm["ts"].isin(old_ver_map.index)
        if common_mask.any():
            new_norm.loc[common_mask, "ver"] = (
                old_ver_map.reindex(new_norm.loc[common_mask, "ts"]).to_numpy(dtype=np.int32) + 1
            )

    # Merge + дедуп в пользу последних записей
    merged = pd.concat([old_norm, new_norm], ignore_index=True)
    merged = merged.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # Фиксированный порядок колонок
    cols = [c for c in _DEF_ORDER if c in merged.columns]
    merged = merged[cols]

    # Если не изменилось — выходим без перезаписи
    if path.exists():
        try:
            old_aligned = old_norm.reindex(columns=cols)
            if old_aligned.equals(merged):
                return path
        except Exception:
            pass

    # Метаданные
    min_ts = int(merged["ts"].iloc[0]) if len(merged) else 0
    max_ts = int(merged["ts"].iloc[-1]) if len(merged) else 0
    meta = {
        "source": source,
        "symbol": symbol,
        "tf": tf,
        "rows": int(len(merged)),
        "min_ts": min_ts,
        "max_ts": max_ts,
        "data_hash": _hash_df(merged),
        "build_signature": _build_signature(),
    }

    # Путь временного файла
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Запись строго через pyarrow с встройкой footer-метаданных
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow обязателен для записи Parquet с footer-метаданными C1") from e

    table = pa.Table.from_pandas(merged, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"c1.meta"] = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    table = table.replace_schema_metadata(md)

    pq.write_table(
        table,
        str(tmp),
        compression=(compression or None),
        compression_level=(zstd_level if compression == "zstd" else None),
        version="2.0",
        use_dictionary=True,
        data_page_size=1 << 20,
        write_statistics=True,
        coerce_timestamps="ms",
        allow_truncated_timestamps=True,
        row_group_size=row_group_size,
    )

    os.replace(str(tmp), str(path))
    return path
