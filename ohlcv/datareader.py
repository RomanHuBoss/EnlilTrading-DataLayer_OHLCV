# ohlcv/datareader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .core.validate import align_and_flag_gaps
from .io.parquet_store import parquet_path
from .io import tail_cache as tail
from .utils.timeframes import MINUTE_MS, parse_tf, to_pandas_freq


_EXPECTED_ORDER = ("o", "h", "l", "c", "v", "t", "is_gap")


@dataclass(frozen=True)
class ReadStats:
    symbol: str
    tf: str
    rows: int
    start_ms: Optional[int]
    end_ms: Optional[int]
    used_filters: bool


class DataReader:
    """Чтение OHLCV/агрегатов из Parquet‑хранилища по (symbol, tf) и диапазонам времени.

    Особенности:
      - Файловая фильтрация по времени через Parquet filters (pyarrow) для ускорения.
      - Вариант выравнивания 1m к минутному календарю с генерацией is_gap.
      - Sidecar‑кэш (latest.json) для быстрого получения хвоста.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    # -------------------------
    # Хвост/метаданные
    # -------------------------
    def latest_ts(self, symbol: str, tf: str) -> Optional[int]:
        info = tail.read(self.root, symbol, tf)
        if info.latest_ts_ms is not None:
            return int(info.latest_ts_ms)
        # Fallback — быстрый проход по Parquet только для столбца ts
        p = parquet_path(self.root, symbol, tf)
        if not Path(p).exists():
            return None
        df = pd.read_parquet(p, columns=["ts"])  # type: ignore[arg-type]
        if len(df) == 0:
            return None
        ts = pd.to_datetime(df["ts"], utc=True)
        return int(ts.iloc[-1].value // 1_000_000)

    # -------------------------
    # Чтение диапазона
    # -------------------------
    def _filters(self, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[tuple]]:
        if start_ms is None and end_ms is None:
            return None
        flt: List[tuple] = []
        if start_ms is not None:
            flt.append(("ts", ">=", pd.to_datetime(int(start_ms), unit="ms", utc=True)))
        if end_ms is not None:
            flt.append(("ts", "<", pd.to_datetime(int(end_ms), unit="ms", utc=True)))
        return flt

    def _select_columns(self, cols: Optional[Sequence[str]], available: Iterable[str]) -> Optional[List[str]]:
        if cols is None:
            return None
        keep = [c for c in cols if c in available]
        return keep or None

    def read_range(
        self,
        symbol: str,
        tf: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        align_1m: bool = False,
        strict: bool = False,
    ) -> tuple[pd.DataFrame, ReadStats]:
        """Читает [start_ms, end_ms) для (symbol, tf).

        columns — опциональный список столбцов; порядок приведён к канону.
        align_1m=True — для tf=="1m" вернёт полный минутный ряд с is_gap.
        """
        path = parquet_path(self.root, symbol, tf)
        p = Path(path)
        if not p.exists():
            empty = pd.DataFrame(index=pd.DatetimeIndex([], name="ts", tz="UTC"))
            return empty, ReadStats(symbol, tf, 0, start_ms, end_ms, False)

        flt = self._filters(start_ms, end_ms)
        use_filters = flt is not None

        cols = None  # читаем минимум ts+запрошенные
        if columns is not None:
            req = list(dict.fromkeys(["ts", *columns]))
            cols = req

        df = pd.read_parquet(p, filters=flt, columns=cols)  # type: ignore[arg-type]
        if len(df) == 0:
            return df.set_index(pd.DatetimeIndex([], name="ts", tz="UTC")), ReadStats(symbol, tf, 0, start_ms, end_ms, use_filters)

        # Индекс и порядок столбцов
        if "ts" not in df.columns:
            raise ValueError("ожидается столбец 'ts' в parquet")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()

        # Восстановить ожидаемый порядок колонок
        ordered = [c for c in _EXPECTED_ORDER if c in df.columns]
        tail_cols = [c for c in df.columns if c not in ordered]
        df = df[ordered + tail_cols]

        # Выравнивание минутного ряда
        if align_1m:
            if tf != "1m":
                raise ValueError("align_1m поддерживается только для tf='1m'")
            df, _ = align_and_flag_gaps(df, start_ms=start_ms, end_ms=end_ms, strict=strict)

        # Финальная выборка колонок
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]

        stats = ReadStats(symbol=symbol, tf=tf, rows=int(len(df)), start_ms=start_ms, end_ms=end_ms, used_filters=use_filters)
        return df, stats

    # -------------------------
    # Утилиты высокого уровня
    # -------------------------
    def read_day_utc(self, symbol: str, tf: str, day_iso: str, *, columns: Optional[Sequence[str]] = None) -> tuple[pd.DataFrame, ReadStats]:
        day = pd.Timestamp(day_iso).tz_localize("UTC") if pd.Timestamp(day_iso).tz is None else pd.Timestamp(day_iso).tz_convert("UTC")
        start_ms = int(day.value // 1_000_000)
        end_ms = int((day + pd.Timedelta(days=1)).value // 1_000_000)
        return self.read_range(symbol, tf, start_ms=start_ms, end_ms=end_ms, columns=columns)

    def iter_days(self, symbol: str, tf: str, start_day_iso: str, end_day_iso: str, *, columns: Optional[Sequence[str]] = None):
        start = pd.Timestamp(start_day_iso).tz_localize("UTC") if pd.Timestamp(start_day_iso).tz is None else pd.Timestamp(start_day_iso).tz_convert("UTC")
        end = pd.Timestamp(end_day_iso).tz_localize("UTC") if pd.Timestamp(end_day_iso).tz is None else pd.Timestamp(end_day_iso).tz_convert("UTC")
        cur = start
        while cur < end:
            nxt = cur + pd.Timedelta(days=1)
            df, _ = self.read_range(symbol, tf, start_ms=int(cur.value // 1_000_000), end_ms=int(nxt.value // 1_000_000), columns=columns)
            yield cur.date().isoformat(), df
            cur = nxt
