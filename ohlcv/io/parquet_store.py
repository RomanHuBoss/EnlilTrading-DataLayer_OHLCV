# ohlcv/io/parquet_store.py
# Хранилище Parquet с идемпотентной догрузкой и расширенными метаданными.
# Требования:
# - Колоночный формат Parquet.
# - Идемпотентный догруз: при коллизиях по ts побеждают новые данные.
# - Стабильные метаданные файла: источник, схема, хэш данных, сигнатура сборки.
# - Компрессия ZSTD, разумный размер row group.
# Примечание: колонка ts хранится как столбец (а не индекс). Индекс в оперативном df — tz-aware UTC DatetimeIndex.

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
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

OPTIONAL_ALLOWED = ("t", "is_gap", "dq_flags", "dq_notes")


def _optional_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in OPTIONAL_ALLOWED:
        if c in df.columns:
            cols.append(c)
    return cols


def _ensure_optional_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Гарантированно добавить опциональные колонки с детерминированными значениями и типами."""
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            if c == "t":
                out[c] = 0.0
            elif c == "is_gap":
                out[c] = False
            elif c == "dq_flags":
                out[c] = np.int32(0)
            elif c == "dq_notes":
                out[c] = ""
    # Приведение типов — важно для стабильного хэша и записи
    if "t" in out.columns:
        out["t"] = out["t"].astype("float64")
    if "is_gap" in out.columns:
        # допускаем NaN → False
        out["is_gap"] = out["is_gap"].fillna(False).astype("bool")
    if "dq_flags" in out.columns:
        out["dq_flags"] = out["dq_flags"].fillna(0).astype("int32")
    if "dq_notes" in out.columns:
        out["dq_notes"] = out["dq_notes"].fillna("").astype("string")
    return out


def _stable_hash(df: pd.DataFrame) -> str:
    """
    Стабильный хэш содержимого данных (индекс+обязательные+опциональные столбцы).
    Хэш считается по байтовому представлению:
      index (ns, int64), o,h,l,c,v (float64), затем по мере наличия t,is_gap,dq_flags,dq_notes.
    Допуски: отсутствующие опциональные колонки считаются как нули/пусто.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Ожидается DatetimeIndex в качестве индекса")

    # Полный набор опциональных колонок, присутствующих в df
    opt_cols = _optional_cols(df)
    work = df[["o", "h", "l", "c", "v"] + opt_cols].copy()

    # Приведение типов к стабильным для хэша и заполнение пропусков
    idx_ns = df.index.view("i8")  # ns int64
    buf_parts: List[bytes] = [idx_ns.tobytes()]

    for c in ["o", "h", "l", "c", "v"]:
        buf_parts.append(np.asarray(work[c].astype("float64").values).tobytes())

    if "t" in work.columns:
        buf_parts.append(np.asarray(work["t"].fillna(0.0).astype("float64").values).tobytes())
    if "is_gap" in work.columns:
        # NaN → False
        buf_parts.append(np.asarray(work["is_gap"].fillna(False).astype("uint8").values).tobytes())
    if "dq_flags" in work.columns:
        buf_parts.append(np.asarray(work["dq_flags"].fillna(0).astype("int32").values).tobytes())
    if "dq_notes" in work.columns:
        notes_bytes = b"\0".join(
            ("" if pd.isna(x) else str(x)).encode("utf-8") for x in work["dq_notes"].values
        )
        buf_parts.append(notes_bytes)

    h = hashlib.sha256()
    for part in buf_parts:
        h.update(part)
    return h.hexdigest()


def _to_pa_table(df: pd.DataFrame, symbol: str, tf: str) -> pa.Table:
    """Преобразование в pyarrow.Table с нужной схемой и метаданными."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("Ожидается tz-aware UTC DatetimeIndex")

    cols = ["o", "h", "l", "c", "v"] + _optional_cols(df)

    tbl = pa.Table.from_pandas(
        df.reset_index().rename(columns={"index": "ts"})[["ts"] + cols],
        preserve_index=False,
    )

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
      2) Выравнивание состава колонок между старым и новым фреймом.
      3) Объединение по индексу ts; при дубликатах побеждают новые значения.
      4) Сортировка по времени; удаление дублей.
      5) Запись Parquet с ZSTD и метаданными.

    Предполагаемый индекс df_new: tz-aware UTC DatetimeIndex, правые границы бара.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = parquet_path(root, symbol, tf)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = FileLock(str(path) + ".lock")
    with lock:
        # Загрузка старых данных (если есть)
        if path.exists():
            old_tbl = pq.read_table(path)
            old = old_tbl.to_pandas()
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            old = old.set_index("ts").sort_index()
        else:
            old = None

        # Подготовка новых данных
        if not isinstance(df_new.index, pd.DatetimeIndex):
            raise ValueError("Ожидается DatetimeIndex в качестве индекса")
        if df_new.index.tz is None:
            df_new = df_new.copy()
            df_new.index = df_new.index.tz_localize("UTC")
        df_new = df_new.sort_index()

        # Базовые и опциональные колонки
        base_cols = ["o", "h", "l", "c", "v"]
        new_opt = set(_optional_cols(df_new))
        old_opt = set(_optional_cols(old)) if old is not None else set()
        union_opt = sorted(
            (new_opt | old_opt),
            key=lambda x: OPTIONAL_ALLOWED.index(x) if x in OPTIONAL_ALLOWED else 999,
        )

        # Приведение состава и типов
        df_new2 = _ensure_optional_columns(df_new[base_cols + list(new_opt)], union_opt)
        if old is not None:
            old2 = _ensure_optional_columns(old[base_cols + list(old_opt)], union_opt)
            df = pd.concat([old2, df_new2], axis=0)
        else:
            df = df_new2

        # Удаление дублей и окончательная сортировка: новые побеждают
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Защита от NaN в опциональных после конкатенации
        df = _ensure_optional_columns(df, union_opt)

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
