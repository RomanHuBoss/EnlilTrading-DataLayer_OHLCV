# ohlcv/core/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

MINUTE_MS = 60_000


@dataclass(frozen=True)
class ValidateStats:
    rows_in: int
    rows_out: int
    gaps: int
    repaired_hl: int
    clipped_oc: int
    dropped_nan: int


def _to_utc_index_from_ms(ms: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(ms.astype("int64"), unit="ms", utc=True)
    return pd.DatetimeIndex(ts, name="ts")


def _derive_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp_ms" in df.columns:
        idx = _to_utc_index_from_ms(df["timestamp_ms"])  # type: ignore[arg-type]
        out = df.copy()
        out.index = idx
        out.index.name = "ts"
        return out
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out.index = (df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC"))
        out.index.name = "ts"
        return out
    raise KeyError("нужен 'timestamp_ms' столбец или DatetimeIndex")


def _coerce_and_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Канонические имена: open/high/low/close/volume → o/h/l/c/v
    rename = {}
    for a, b in {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}.items():
        if a in out.columns:
            rename[a] = b
    out = out.rename(columns=rename)

    for col in ["o", "h", "l", "c", "v"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if "t" in out.columns:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("float64")

    # Удаление явных дубликатов по времени, сортировка
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # Канонический порядок колонок
    cols = ["o", "h", "l", "c", "v"] + (["t"] if "t" in out.columns else [])
    others = [c for c in out.columns if c not in cols]
    return out[cols + others]


def _repair_or_raise_ohlc(out: pd.DataFrame, strict: bool) -> tuple[int, int]:
    repaired_hl = 0
    clipped_oc = 0

    # high < low → обмен значениями
    mask_swap = out["h"] < out["l"]
    if mask_swap.any():
        if strict:
            raise ValueError("нарушение инварианта: high < low")
        h = out.loc[mask_swap, "l"].values
        l = out.loc[mask_swap, "h"].values
        out.loc[mask_swap, "h"] = h
        out.loc[mask_swap, "l"] = l
        repaired_hl = int(mask_swap.sum())

    # open/close вне диапазона → клип внутрь [low, high]
    oc_low = (out["o"] < out["l"]) | (out["c"] < out["l"]) | out["o"].isna() | out["c"].isna()
    oc_high = (out["o"] > out["h"]) | (out["c"] > out["h"]) | out["o"].isna() | out["c"].isna()
    if (oc_low | oc_high).any():
        if strict:
            raise ValueError("нарушение инварианта: open/close вне [low, high]")
        before = (oc_low | oc_high).sum()
        out["o"] = out["o"].clip(lower=out["l"], upper=out["h"]).astype("float64")
        out["c"] = out["c"].clip(lower=out["l"], upper=out["h"]).astype("float64")
        clipped_oc = int(before)

    return repaired_hl, clipped_oc


def normalize_ohlcv_1m(df: pd.DataFrame, strict: bool = False) -> tuple[pd.DataFrame, ValidateStats]:
    """Нормализация минутного OHLCV к канону: tz-aware UTC индекс, столбцы o/h/l/c/v (+t).

    Поведение:
      - приводим типы, переименовываем, сортируем и дедуплицируем по времени;
      - +/-inf → NaN; при strict=True наличие NaN в o/h/l/c → ошибка; иначе строки с NaN удаляются;
      - volume<0 → 0;
      - ремонт/валидация инвариантов high>=low и open/close∈[low,high].
    """
    inp_rows = int(len(df))

    out = _derive_ts_index(df)
    out = _coerce_and_order(out)
    out = out.replace([np.inf, -np.inf], np.nan)

    # удаляем строки с NaN в ключевых колонках
    req = ["o", "h", "l", "c"]
    nan_mask = out[req].isna().any(axis=1)
    if nan_mask.any():
        if strict:
            raise ValueError("NaN в обязательных колонках o/h/l/c при strict=True")
        out = out.loc[~nan_mask].copy()
        dropped_nan = int(nan_mask.sum())
    else:
        dropped_nan = 0

    # volume и t
    out["v"] = out["v"].fillna(0.0).clip(lower=0.0).astype("float64")
    if "t" in out.columns:
        out["t"] = out["t"].fillna(0.0).clip(lower=0.0).astype("float64")

    repaired_hl, clipped_oc = _repair_or_raise_ohlc(out, strict=strict)

    stats = ValidateStats(
        rows_in=inp_rows,
        rows_out=int(len(out)),
        gaps=0,
        repaired_hl=repaired_hl,
        clipped_oc=clipped_oc,
        dropped_nan=dropped_nan,
    )
    return out, stats


def build_minute_calendar(start_ms: int, end_ms: int) -> pd.DatetimeIndex:
    """Минутная шкала [start, end) в UTC (левая включительно, правая исключена).

    start и end округляются к минутным границам вниз/вверх соответственно.
    """
    if end_ms <= start_ms:
        return pd.DatetimeIndex([], name="ts", tz="UTC")
    start_ms = (start_ms // MINUTE_MS) * MINUTE_MS
    # делаем правую границу эксклюзивной, но строим range включительно и потом отбрасываем последний шаг
    end_ms = ((end_ms + MINUTE_MS - 1) // MINUTE_MS) * MINUTE_MS
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)
    # включительно → добавит последний штрих; отрежем его ниже через[:-1]
    idx = pd.date_range(start, end, freq="1T", inclusive="both", tz="UTC")
    if len(idx) == 0:
        return idx
    return pd.DatetimeIndex(idx[:-1], name="ts")


def align_and_flag_gaps(df1m: pd.DataFrame, start_ms: Optional[int] = None, end_ms: Optional[int] = None, *, strict: bool = False) -> tuple[pd.DataFrame, ValidateStats]:
    """Выравнивает 1m OHLCV по минутному календарю и проставляет is_gap.

    Вход: DataFrame с индексом ts (UTC) и колонками o/h/l/c/v (+t). Если индекс/колонки иные —
    используйте normalize_ohlcv_1m сначала.

    Результат: полный минутный ряд [start, end) c колонками o/h/l/c/v (+t, если была) и булевым is_gap.
    На отсутствующих барах значения NaN, is_gap=True.
    """
    assert isinstance(df1m.index, pd.DatetimeIndex) and df1m.index.tz is not None, "ожидается tz-aware DatetimeIndex"

    if start_ms is None:
        start_ms = int(df1m.index[0].value // 1_000_000)
    if end_ms is None:
        end_ms = int(df1m.index[-1].value // 1_000_000 + MINUTE_MS)

    cal = build_minute_calendar(start_ms, end_ms)

    # левый join календаря к данным
    base_cols = ["o", "h", "l", "c", "v"] + (["t"] if "t" in df1m.columns else [])
    df = df1m[base_cols].copy()
    df = df[~df.index.duplicated(keep="last")]  # страховка

    aligned = pd.DataFrame(index=cal)
    for c in base_cols:
        if c in df.columns:
            aligned[c] = df[c]
        else:
            aligned[c] = np.nan

    # флаг пропуска
    aligned["is_gap"] = aligned["o"].isna() & aligned["h"].isna() & aligned["l"].isna() & aligned["c"].isna()

    # Статистика
    gaps = int(aligned["is_gap"].sum())

    # Проверка монотонности
    if not (np.all(aligned.index[1:].values > aligned.index[:-1].values)):
        raise AssertionError("не монотонный индекс после выравнивания")

    # При strict=True запрещаем NaN внутри присутствующих баров
    if strict:
        bad = (~aligned["is_gap"]) & aligned[["o", "h", "l", "c"]].isna().any(axis=1)
        if bad.any():
            raise ValueError("NaN в существующих барах при strict=True")

    stats = ValidateStats(
        rows_in=int(len(df1m)),
        rows_out=int(len(aligned)),
        gaps=gaps,
        repaired_hl=0,
        clipped_oc=0,
        dropped_nan=0,
    )
    return aligned, stats
