# ohlcv/io/parquet_store.py
# Хранилище Parquet с идемпотентной догрузкой и расширенными метаданными.
# Требования:
# - Колоночный формат Parquet.
# - Идемпотентный догруз: при коллизиях по ts побеждают новые данные.
# - Стабильные метаданные файла: источник, схема, хэш данных, сигнатура сборки.
# - Компрессия ZSTD, разумный размер row group.
# Примечание: колонка ts хранится как столбец (а не индекс). Индекс в оперативном df — tz-aware UTC DatetimeIndex.

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import List
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

# Пространство имён метаданных
META_NS = b"c1.meta"

# Компрессия/разбиение по группам строк
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 7
PARQUET_ROW_GROUP_SIZE = 256_000  # ориентир, может быть скорректирован по профилю


# ---- Вспомогательные утилиты -------------------------------------------------


def _optional_cols(df: pd.DataFrame) -> List[str]:
    """
    Разрешённые опциональные столбцы:
      - t          : turnover
      - is_gap     : флаг синтетического бара (календаризация)
      - dq_flags   : битовая маска правил DataQuality (C2)
      - dq_notes   : текстовые пометки DataQuality (C2)
    """
    cols: List[str] = []
    for c in ("t", "is_gap", "dq_flags", "dq_notes"):
        if c in df.columns:
            cols.append(c)
    return cols


def _stable_hash(df: pd.DataFrame) -> str:
    """
    Стабильный хэш содержимого данных (индекс+обязательные+опциональные столбцы).
    Хэш считается по байтовому представлению:
      index (ns, int64), o,h,l,c,v (float64), затем по мере наличия t,is_gap,dq_flags,dq_notes.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex в качестве индекса")
    # Порядок столбцов
    cols = ["o", "h", "l", "c", "v"] + _optional_cols(df)
    work = df[cols].copy()

    # Приведение типов к стабильным для хэша
    idx_ns = df.index.view("i8")  # ns int64
    buf_parts: List[bytes] = [idx_ns.tobytes()]

    # Числовые
    for c in ["o", "h", "l", "c", "v"]:
        buf_parts.append(np.asarray(work[c].astype("float64").values).tobytes())

    # Опциональные
    if "t" in work.columns:
        buf_parts.append(np.asarray(work["t"].astype("float64").values).tobytes())
    if "is_gap" in work.columns:
        # приведение к uint8
        buf_parts.append(np.asarray(work["is_gap"].astype("uint8").values).tobytes())
    if "dq_flags" in work.columns:
        buf_parts.append(np.asarray(work["dq_flags"].astype("int32").values).tobytes())
    if "dq_notes" in work.columns:
        # Преобразуем к bytes построчно с разделителем \0 (детерминированно)
        notes_bytes = b"\0".join(
            ("" if pd.isna(x) else str(x)).encode("utf-8") for x in work["dq_notes"].values
        )
        buf_parts.append(notes_bytes)

    h = hashlib.sha256()
    for part in buf_parts:
        h.update(part)
    return h.hexdigest()


def _to_pa_table(df: pd.DataFrame, symbol: str, tf: str) -> pa.Table:
    """
    Преобразование в pyarrow.Table с нужной схемой и метаданными.
    ts переносим в колонку; ожидается tz-aware UTC.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Ожидается tz-aware UTC DatetimeIndex")

    # Базовые и опциональные колонки
    cols = ["o", "h", "l", "c", "v"] + _optional_cols(df)

    # Сброс индекса в колонку ts (ISO8401 сохраняется через pandas→arrow как timestamp[ns, UTC])
    tbl = pa.Table.from_pandas(
        df.reset_index().rename(columns={"index": "ts"})[["ts"] + cols],
        preserve_index=False,
    )

    # Метаданные
    md = dict(tbl.schema.metadata or {})
    meta = {
        "symbol": symbol,
        "tf": tf,
        "schema": ["ts", "o", "h", "l", "c", "v"] + [c + "?" for c in _optional_cols(df)],
        "source": "bybit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "build_signature": os.environ.get("C1_BUILD_SIGNATURE", "unknown"),
        "data_hash": _stable_hash(df),
    }
    md[META_NS] = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    return tbl.replace_schema_metadata(md)


def parquet_path(root: Path, symbol: str, tf: str) -> Path:
    return root / symbol / f"{tf}.parquet"


# ---- Основная запись ---------------------------------------------------------


def write_idempotent(root: Path, symbol: str, tf: str, df_new: pd.DataFrame) -> Path:
    """
    Идемпотентная запись Parquet:
      1) Чтение существующего файла (если есть).
      2) Объединение по индексу ts; при дубликатах побеждают новые значения.
      3) Сортировка по времени; удаление дублей.
      4) Запись единого Parquet с ZSTD и метаданными.

    Предполагаемый индекс df_new: tz-aware UTC DatetimeIndex, правые границы бара.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Лок на время MERGE+WRITE
    lock = FileLock(str(path) + ".lock")
    with lock:
        # Загрузка старых данных (если файл существует)
        if path.exists():
            old_tbl = pq.read_table(path)
            old = old_tbl.to_pandas()
            # Приведение ts к UTC Timestamp
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            old = old.set_index("ts").sort_index()
        else:
            old = None

        # Приведение новых данных к унифицированной форме и сортировка
        keep_cols = ["o", "h", "l", "c", "v"] + _optional_cols(df_new)
        df = df_new.copy()[keep_cols]
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Ожидается DatetimeIndex в качестве индекса")
        if df.index.tz is None:
            # Защита от случайного наивного индекса
            df.index = df.index.tz_localize("UTC")
        df = df.sort_index()

        # Объединение
        if old is not None and not old.empty:
            df = pd.concat([old, df])

        # Удаление дублей и окончательная сортировка
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Формирование и запись Parquet
        tbl = _to_pa_table(df, symbol, tf)
        pq.write_table(
            tbl,
            path,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_COMPRESSION_LEVEL,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            use_dictionary=True,
        )

    return path
