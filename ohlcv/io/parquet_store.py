"""Parquet Store: idempotent append/merge with metadata.

- Columnar Parquet.
- Idempotent merge on "ts" (UTC minute right edges). New rows win on collision.
- Stable file metadata: minimal JSON blob under the "c1.meta" key.
- ZSTD compression, reasonable row group size.
- Note: "ts" is stored as a column (not as index). Input df index must be tz-aware UTC.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 6
PARQUET_ROW_GROUP_SIZE = 64_000


def parquet_path(root: Path | str, symbol: str, tf: str) -> Path:
    """Return canonical path: <root>/<symbol>/<tf>.parquet."""
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


def _meta_blob(symbol: str, tf: str, df: pd.DataFrame) -> bytes:
    payload = {
        "symbol": symbol,
        "tf": tf,
        "rows": int(len(df)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_sha1": hashlib.sha1(
            pd.util.hash_pandas_object(df.fillna(0), index=True).values.tobytes()
        ).hexdigest(),
        "schema": [str(c) for c in df.columns],
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
    schema = pa.schema([pa.field(name, arr.type) for name, arr in arrays])
    tbl = pa.Table.from_arrays(
        [arr for _, arr in arrays],
        names=[name for name, _ in arrays],
        schema=schema,
    )
    md = {"c1.meta": _meta_blob(symbol, tf, df)}
    tbl = tbl.replace_schema_metadata(md)
    return tbl


def write_idempotent(root: Path | str, symbol: str, tf: str, df: pd.DataFrame) -> Path:
    """Idempotent write:
    - create parent directories
    - merge with existing file on 'ts', prioritising new rows
    - write Parquet with "c1.meta" metadata
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
        # keep old where new is NaN; then update with new to make new rows win
        merged.update(df)
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
