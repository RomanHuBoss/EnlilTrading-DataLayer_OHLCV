# Хранилище Parquet с идемпотентной догрузкой.
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

META_NS = b"c1.meta"  # пространство имён для метаданных

def _to_pa_table(df: pd.DataFrame, symbol: str, tf: str) -> pa.Table:
    tbl = pa.Table.from_pandas(df.reset_index().rename(columns={"index": "ts"}), preserve_index=False)
    # Метаданные в файле
    md = tbl.schema.metadata or {}
    md = dict(md)
    md[META_NS] = str({
        "symbol": symbol,
        "tf": tf,
        "schema": ["ts","o","h","l","c","v","t?"],
        "generated_at": datetime.now(timezone.utc).isoformat()
    }).encode("utf-8")
    return tbl.replace_schema_metadata(md)

def parquet_path(root: Path, symbol: str, tf: str) -> Path:
    return root / symbol / f"{tf}.parquet"

def write_idempotent(root: Path, symbol: str, tf: str, df_new: pd.DataFrame) -> Path:
    """
    Идемпотентная запись:
      - читаем существующий файл, если есть
      - объединяем по индексу (ts), при коллизии берём df_new (свежее)
      - сортируем, гарантируем уникальность ts
      - пишем заново единый Parquet
    """
    root.mkdir(parents=True, exist_ok=True)
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock")
    with lock:
        if path.exists():
            old = pq.read_table(path).to_pandas()
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            old = old.set_index("ts").sort_index()
        else:
            old = None
        df = df_new.copy()
        df = df[["o","h","l","c","v"] + ([ "t" ] if "t" in df_new.columns else [])]
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Ожидается DatetimeIndex в качестве индекса")
        df = df.sort_index()
        if old is not None and not old.empty:
            df = pd.concat([old, df])
        # Исправление после concat: удалить дубли, отсортировать
        df = df[~df.index.duplicated(keep="last")].sort_index()
        tbl = _to_pa_table(df, symbol, tf)
        pq.write_table(tbl, path)
    return path
