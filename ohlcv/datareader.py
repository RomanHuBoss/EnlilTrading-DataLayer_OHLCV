# ohlcv/datareader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io.parquet_store import parquet_path

# =============================
# Константы времени
# =============================
MINUTE_MS = 60_000


# =============================
# Статистика чтения
# =============================
@dataclass(frozen=True)
class ReadStats:
    rows: int
    min_ts: Optional[int]
    max_ts: Optional[int]


# =============================
# Вспомогательные
# =============================

def _as_utc_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is None:
            return idx.tz_localize("UTC")
        return idx.tz_convert("UTC")
    raise ValueError("ожидается DatetimeIndex")


def _ensure_is_gap(df: pd.DataFrame) -> pd.DataFrame:
    if "is_gap" not in df.columns:
        df = df.copy()
        df["is_gap"] = False
    return df


# =============================
# DataReader
# =============================
@dataclass
class DataReader:
    root: Path | str

    # -------------
    # Основное API
    # -------------
    def read_range(
        self,
        symbol: str,
        tf: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, ReadStats]:
        """Читает диапазон [start_ms, end_ms) для (symbol, tf) из store.

        Возвращает (df, ReadStats). Гарантируется сортировка по ts и отсутствие дубликатов.
        Если указаны columns — возвращаются только эти колонки, если их нет в файле — создаются пустые по канону (например, is_gap=False).
        """
        p = parquet_path(self.root, symbol, tf)
        if not Path(p).exists():
            # Пустой результат
            empty = pd.DataFrame(columns=["ts"]).astype({"ts": "int64"})
            st = ReadStats(rows=0, min_ts=None, max_ts=None)
            return empty, st

        df = pd.read_parquet(p)

        # Нормализация ts
        if "ts" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
            df["ts"] = (pd.to_datetime(df["ts"], utc=True).view("int64") // 1_000_000).astype("int64")
        elif "ts" not in df.columns:
            raise ValueError("файл без колонки ts")

        df = (
            df.copy()
            .assign(ts=pd.to_numeric(df["ts"], errors="coerce").astype("int64"))
            .sort_values("ts")
            .drop_duplicates(subset=["ts"], keep="last")
            .reset_index(drop=True)
        )

        # Фильтр по времени
        if start_ms is not None:
            df = df[df["ts"] >= int(start_ms)]
        if end_ms is not None:
            df = df[df["ts"] < int(end_ms)]

        # Подготовка колонок по запросу
        df = _ensure_is_gap(df)
        if columns is not None:
            cols_req = list(columns)
            # Всегда сохраняем ts
            cols_final: list[str] = ["ts"] + [c for c in cols_req if c != "ts"]
            # Создаём недостающие
            for c in cols_final:
                if c not in df.columns:
                    if c in ("o", "h", "l", "c", "v", "t"):
                        df[c] = np.nan
                    elif c == "is_gap":
                        df[c] = False
                    else:
                        df[c] = np.nan
            df = df[cols_final]
        else:
            # Канонический порядок при полном чтении
            order = [c for c in ["ts", "o", "h", "l", "c", "v", "t", "is_gap"] if c in df.columns]
            df = df[order]

        # Статистика
        st = ReadStats(
            rows=int(len(df)),
            min_ts=int(df["ts"].min()) if len(df) else None,
            max_ts=int(df["ts"].max()) if len(df) else None,
        )
        return df, st

    # -------------
    # Утилиты по дням
    # -------------
    def read_day(
        self,
        symbol: str,
        tf: str,
        day_iso: str,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, ReadStats]:
        day = pd.Timestamp(day_iso)
        day = day.tz_localize("UTC") if day.tz is None else day.tz_convert("UTC")
        start_ms = int(day.value // 1_000_000)
        end_ms = int((day + pd.Timedelta(days=1)).value // 1_000_000)
        return self.read_range(symbol, tf, start_ms=start_ms, end_ms=end_ms, columns=columns)

    def iter_days(
        self,
        symbol: str,
        tf: str,
        start_day_iso: str,
        end_day_iso: str,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> Iterable[Tuple[str, pd.DataFrame]]:
        start = pd.Timestamp(start_day_iso)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        end = pd.Timestamp(end_day_iso)
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
        cur = start
        while cur < end:
            nxt = cur + pd.Timedelta(days=1)
            df, _ = self.read_range(
                symbol,
                tf,
                start_ms=int(cur.value // 1_000_000),
                end_ms=int(nxt.value // 1_000_000),
                columns=columns,
            )
            yield cur.date().isoformat(), df
            cur = nxt
