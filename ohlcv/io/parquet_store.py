"""Parquet Store: идемпотентная запись и слияние по минутным барам.

Изменения:
- ZSTD level = 7, row_group_size = 256_000, use_dictionary = True.
- Стабильные метаданные в footer под ключом "c1.meta":
  {
    symbol, tf, rows, min_ts, max_ts, generated_at,
    data_hash, content_sha1(back-compat), build_signature,
    schema, schema_version, zstd_level, row_group_size
  }
- Идемпотентный merge по 'ts' с приоритетом новых строк.
- Жёстная нормализация типов и сортировка по времени.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 7
PARQUET_ROW_GROUP_SIZE = 256_000
SCHEMA_VERSION = 1


def parquet_path(root: Path | str, symbol: str, tf: str) -> Path:
    """Канонический путь: <root>/<symbol>/<tf>.parquet."""
    root_p = Path(root).expanduser().resolve()
    return root_p / symbol / f"{tf}.parquet"


def _reset_index_to_ts(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise TypeError("df must have tz-aware DatetimeIndex (UTC)")
    out = df.copy()
    out = out.sort_index()
    out = out.reset_index(names="ts")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def _ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["o", "h", "l", "c", "v"]:
        if c not in df.columns:
            df[c] = 0.0
    if "t" in df.columns:
        df["t"] = pd.to_numeric(df["t"], errors="coerce").astype("float64")
    if "is_gap" in df.columns:
        df["is_gap"] = df["is_gap"].astype(bool)
    return df


def _order_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = ["ts", "o", "h", "l", "c", "v"]
    if "t" in df.columns:
        cols.append("t")
    if "is_gap" in df.columns:
        cols.append("is_gap")
    tail = [c for c in df.columns if c not in cols]
    return cols + tail


def _canonical_content_hash(df: pd.DataFrame) -> str:
    # Стабильный хэш содержимого: сортировка по ts, NaN→0 для детерминизма.
    df_sorted = df.sort_values("ts").reset_index(drop=True)
    h = pd.util.hash_pandas_object(df_sorted.fillna(0), index=True).values.tobytes()
    return hashlib.sha1(h).hexdigest()


def _build_signature() -> str:
    # Источник: версия пакета + внешняя переменная окружения с git-commit (если есть)
    try:
        from ohlcv import __version__  # lazy-import, чтобы избежать циклов
    except Exception:
        __version__ = "0.0.0"  # type: ignore
    git_sha = os.getenv("GIT_COMMIT") or os.getenv("CI_COMMIT_SHA") or os.getenv("SOURCE_VERSION")
    if git_sha:
        git_tag = git_sha[:8]
        return f"C1-{__version__}+g{git_tag}"
    return f"C1-{__version__}"


def _meta_blob(symbol: str, tf: str, df: pd.DataFrame) -> bytes:
    ts = pd.to_datetime(df["ts"], utc=True)
    payload = {
        "symbol": symbol,
        "tf": tf,
        "rows": int(len(df)),
        "min_ts": (ts.min().isoformat() if not df.empty else None),
        "max_ts": (ts.max().isoformat() if not df.empty else None),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_hash": _canonical_content_hash(df),
        "content_sha1": _canonical_content_hash(df),  # back-compat с прежним ключом
        "build_signature": _build_signature(),
        "schema": [str(c) for c in df.columns],
        "schema_version": SCHEMA_VERSION,
        "zstd_level": PARQUET_COMPRESSION_LEVEL,
        "row_group_size": PARQUET_ROW_GROUP_SIZE,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _to_pa_table(df: pd.DataFrame, symbol: str, tf: str) -> pa.Table:
    arrays: List[Tuple[str, pa.Array]] = []
    ts = pa.array(pd.to_datetime(df["ts"], utc=True), type=pa.timestamp("ns", tz="UTC"))
    arrays.append(("ts", ts))
    for c in ["o", "h", "l", "c", "v"]:
        arrays.append((c, pa.array(df[c].astype("float64"), type=pa.float64())))
    if "t" in df.columns:
        arrays.append(("t", pa.array(df["t"].astype("float64"), type=pa.float64())))
    if "is_gap" in df.columns:
        arrays.append(("is_gap", pa.array(df["is_gap"].astype("bool"), type=pa.bool_())))
    for c in df.columns:
        if c in {"ts", "o", "h", "l", "c", "v", "t", "is_gap"}:
            continue
        arrays.append((c, pa.array(df[c])))

    tbl = pa.Table.from_arrays([arr for _, arr in arrays], names=[name for name, _ in arrays])
    md = {"c1.meta": _meta_blob(symbol, tf, df)}
    tbl = tbl.replace_schema_metadata(md)
    return tbl


def write_idempotent(root: Path | str, symbol: str, tf: str, df: pd.DataFrame) -> Path:
    """Идемпотентная запись:
    - создаёт каталоги
    - сливает с существующим файлом по 'ts' (новые строки побеждают)
    - пишет Parquet с обновлёнными метаданными
    """
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = _reset_index_to_ts(df)
    _ensure_numeric(df, ["o", "h", "l", "c", "v"])
    df = _ensure_optional_columns(df)
    df = df.set_index("ts").sort_index()

    if path.exists():
        old = pd.read_parquet(path)
        old["ts"] = pd.to_datetime(old["ts"], utc=True)
        old = old.set_index("ts").sort_index()
        all_cols = sorted(set(df.columns).union(old.columns))
        df = df.reindex(columns=all_cols)
        old = old.reindex(columns=all_cols)
        merged = old.combine_first(df)
        merged.update(df)  # новые значения побеждают
        out = merged.sort_index()
    else:
        out = df

    out = out.reset_index()
    cols = _order_columns(out)
    out = out[cols]

    table = _to_pa_table(out, symbol, tf)
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
        use_dictionary=True,
    )
    return path
