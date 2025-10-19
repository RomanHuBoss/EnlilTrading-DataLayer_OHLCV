# ohlcv/core/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MINUTE_MS",
    "ValidateStats",
    "normalize_ohlcv_1m",
    "align_and_flag_gaps",
]

# =============================
# Константы
# =============================
MINUTE_MS = 60_000


# =============================
# Статистика
# =============================
@dataclass(frozen=True)
class ValidateStats:
    rows_in: int
    rows_out: int
    minutes_expected: int
    minutes_present: int
    minutes_missing: int
    missing_pct: float
    first_ts_ms: Optional[int]
    last_ts_ms: Optional[int]


# =============================
# Утилиты
# =============================

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # ts → DatetimeIndex если нужно
    if not isinstance(out.index, pd.DatetimeIndex):
        # предпочитаем колонку ts (ms)
        if "ts" in out.columns:
            idx = pd.to_datetime(out["ts"], unit="ms", utc=True)
            out = out.drop(columns=["ts"])  # ts → индекс
        else:
            raise ValueError("ожидалась колонка 'ts' (ms) или DatetimeIndex")
    else:
        idx = out.index
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    out.index = pd.DatetimeIndex(idx, tz="UTC").sort_values()
    out.index.name = "ts"
    return out


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Привести типы числовых колонок, создать отсутствующие
    for c in ["o", "h", "l", "c", "v", "t"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "is_gap" not in out.columns:
        out["is_gap"] = False
    else:
        out["is_gap"] = out["is_gap"].astype(bool)
    # Снять дубли по индексу
    out = out[~out.index.duplicated(keep="last")]
    return out[["o", "h", "l", "c", "v", "t", "is_gap"]]


# =============================
# Нормализация 1m
# =============================

def normalize_ohlcv_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует сырые 1m OHLCV в канонический вид.

    Выход: DatetimeIndex UTC name='ts', столбцы: o,h,l,c,v,[t],is_gap.
    Дубликаты минут по индексу — удаляются (последняя запись выигрывает).
    """
    out = _to_utc_index(df)
    out = _ensure_columns(out)
    return out


# =============================
# Выравнивание и пометка пропусков
# =============================

def align_and_flag_gaps(
    df_1m: pd.DataFrame,
    *,
    strict: bool = True,
) -> Tuple[pd.DataFrame, ValidateStats]:
    """Выравнивает 1m данные на полную минутную сетку и помечает пропуски.

    Правила C1:
    - индексация UTC по минутам, label — правая граница минуты;
    - если минута отсутствует: синтетический бар с `o=h=l=c=prev_close`, `v=0`, `t=0` (если есть), `is_gap=true`;
    - если присутствует бар, но с NaN в OHLC: заменить на prev_close и `is_gap=true`;
    - первый бар, если нет prev_close — оставить фактическим (без синтетики), NaN не затираем;
    - статистика: ожидаемые/присутствующие/пропущенные минуты и доля пропусков.
    """
    if df_1m.empty:
        empty = pd.DataFrame(columns=["o", "h", "l", "c", "v", "t", "is_gap"], index=pd.DatetimeIndex([], tz="UTC", name="ts"))
        stats = ValidateStats(
            rows_in=0,
            rows_out=0,
            minutes_expected=0,
            minutes_present=0,
            minutes_missing=0,
            missing_pct=0.0,
            first_ts_ms=None,
            last_ts_ms=None,
        )
        return empty, stats

    src = normalize_ohlcv_1m(df_1m)

    # Минутная сетка от первой до последней минуты включительно
    start = src.index.min().floor("T")
    end = src.index.max().floor("T")
    full_idx = pd.date_range(start, end, freq="T", tz="UTC", name="ts")

    # Подсчёт статистики до выравнивания
    minutes_expected = int(len(full_idx))
    minutes_present = int(len(src))
    minutes_missing = max(0, minutes_expected - minutes_present)
    missing_pct = float(minutes_missing / minutes_expected) if minutes_expected else 0.0

    # Реиндексация: где пропуски — NaN
    out = src.reindex(full_idx)

    # prev_close для синтетики
    prev_close = out["c"].ffill()

    # 1) Синтетика для полностью отсутствующих минут
    miss_mask = out["o"].isna() & out["h"].isna() & out["l"].isna() & out["c"].isna()
    if miss_mask.any():
        out.loc[miss_mask, ["o", "h", "l", "c"]] = prev_close.loc[miss_mask].values.reshape(-1, 1)
        out.loc[miss_mask, "v"] = 0.0
        if "t" in out.columns:
            out.loc[miss_mask, "t"] = 0.0
        out.loc[miss_mask, "is_gap"] = True

    # 2) На месте, где OHLC частично NaN — добиваем prev_close и помечаем gap
    partial_nan = out[["o", "h", "l", "c"]].isna().any(axis=1) & (~miss_mask)
    if partial_nan.any():
        for col in ["o", "h", "l", "c"]:
            out.loc[partial_nan & out[col].isna(), col] = prev_close.loc[partial_nan & out[col].isna()]
        out.loc[partial_nan, "is_gap"] = True

    # Первый бар без prev_close оставляем как есть (если вся строка осталась NaN — не трогаем)
    # Уже реализовано ffill'ом prev_close и условием

    # Итоговые типы и порядок колонок
    out[["o", "h", "l", "c", "v"]] = out[["o", "h", "l", "c", "v"]].astype("float64")
    if "t" in out.columns:
        out["t"] = out["t"].astype("float64")
    out["is_gap"] = out["is_gap"].astype(bool)

    rows_in = int(len(df_1m))
    rows_out = int(len(out))

    stats = ValidateStats(
        rows_in=rows_in,
        rows_out=rows_out,
        minutes_expected=minutes_expected,
        minutes_present=minutes_present,
        minutes_missing=minutes_missing,
        missing_pct=missing_pct,
        first_ts_ms=int(full_idx[0].value // 1_000_000) if len(full_idx) else None,
        last_ts_ms=int(full_idx[-1].value // 1_000_000) if len(full_idx) else None,
    )

    # Режим strict=False — не генерировать синтетику, только помечать статистику
    if not strict:
        # Вернём просто src, но со стабильным индексом и колонками
        src[["o", "h", "l", "c", "v"]] = src[["o", "h", "l", "c", "v"]].astype("float64")
        if "t" in src.columns:
            src["t"] = src["t"].astype("float64")
        if "is_gap" not in src.columns:
            src["is_gap"] = False
        src["is_gap"] = src["is_gap"].astype(bool)
        return src, stats

    return out, stats
