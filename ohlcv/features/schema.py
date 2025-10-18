# ohlcv/features/schema.py
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

# Каноническая схема входа для C3
EXPECTED_DTYPES: Dict[str, str] = {
    "timestamp_ms": "int64",
    "start_time_iso": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

REQUIRED_COLUMNS: List[str] = [
    "timestamp_ms",
    "open",
    "high",
    "low",
    "close",
]

OPTIONAL_COLUMNS: List[str] = [
    "start_time_iso",
    "volume",
    # допускаем «сквозной» проход дополнительных колонок (например, is_gap)
]


def _derive_timestamp_ms_from_index(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        # допускаем tz-naive → считаем, что это UTC
        if idx.tz is None:
            ts = idx.tz_localize("UTC")
        else:
            ts = idx.tz_convert("UTC")
        return (ts.view("int64") // 1_000_000).astype("int64")
    raise KeyError("timestamp_ms отсутствует и индекс не DatetimeIndex")


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp_ms" not in out.columns:
        out["timestamp_ms"] = _derive_timestamp_ms_from_index(out)
    for c in ["open", "high", "low", "close"]:
        if c not in out.columns:
            raise KeyError(f"Отсутствует обязательная колонка: {c}")
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "start_time_iso" not in out.columns:
        ts = pd.to_datetime(out["timestamp_ms"], unit="ms", utc=True)
        out["start_time_iso"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return out


def _cast_to_expected(out: pd.DataFrame) -> pd.DataFrame:
    # Приводим только известные столбцы; неизвестные не трогаем
    for c, dt in EXPECTED_DTYPES.items():
        if c in out.columns:
            try:
                out[c] = out[c].astype(dt)
            except Exception:
                # последняя попытка — через to_numeric для чисел
                if dt.startswith("float") or dt.startswith("int"):
                    out[c] = pd.to_numeric(out[c], errors="coerce").astype(dt)
                else:
                    out[c] = out[c].astype("string")
    return out


def _repair_ohlc(out: pd.DataFrame) -> None:
    # high/low перестановка
    swap_mask = out["high"] < out["low"]
    if swap_mask.any():
        h = out.loc[swap_mask, "low"].values
        l = out.loc[swap_mask, "high"].values
        out.loc[swap_mask, "high"] = h
        out.loc[swap_mask, "low"] = l
    # клип открытий/закрытий внутрь размаха
    out["open"] = out["open"].clip(lower=out["low"], upper=out["high"]).astype("float64")
    out["close"] = out["close"].clip(lower=out["low"], upper=out["high"]).astype("float64")


def _assert_monotonic_timestamps(out: pd.DataFrame) -> None:
    ts = out["timestamp_ms"].values
    if not (np.all(ts[1:] >= ts[:-1])):
        raise ValueError("timestamp_ms не монотонно неубывающий после нормализации")


def normalize_and_validate(df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Нормализует входной OHLCV к канону C3 и проверяет базовые инварианты.

    Поведение:
      - обеспечивает наличие timestamp_ms/start_time_iso/volume;
      - приводит типы к EXPECTED_DTYPES;
      - удаляет дубликаты по timestamp_ms (keep=last) и сортирует по времени;
      - заменяет +/-inf на NaN;
      - strict=False: исправляет high<low и клипует open/close внутрь [low, high];
      - strict=True: при нарушениях инвариантов и NaN в обязательных столбцах — исключение.
    """

    out = _ensure_required_columns(df)

    # типы и базовая чистка
    out = _cast_to_expected(out)
    out = out.replace([np.inf, -np.inf], np.nan)

    # удаление дубликатов и сортировка
    out = (
        out.drop_duplicates(subset=["timestamp_ms"], keep="last")
        .sort_values("timestamp_ms")
        .reset_index(drop=True)
    )

    # проверки NaN в обязательных колонках
    req = ["timestamp_ms", "open", "high", "low", "close"]
    nan_bad = out[req].isna().any(axis=1)
    if nan_bad.any():
        if strict:
            n = int(nan_bad.sum())
            raise ValueError(f"Обнаружены NaN/NaT в обязательных колонках: {n} строк")
        out = out.loc[~nan_bad].reset_index(drop=True)

    # инварианты OHLC
    bad_hl = (out["high"] < out["low"]) | out["high"].isna() | out["low"].isna()
    if bad_hl.any():
        if strict:
            raise ValueError("Нарушение инварианта high>=low")
        _repair_ohlc(out)

    # open/close в пределах [low, high]
    oc_low = (
        (out["open"] < out["low"])
        | (out["close"] < out["low"])
        | out["open"].isna()
        | out["close"].isna()
    )
    oc_high = (
        (out["open"] > out["high"])
        | (out["close"] > out["high"])
        | out["open"].isna()
        | out["close"].isna()
    )
    if (oc_low | oc_high).any():
        if strict:
            raise ValueError("Нарушение инварианта open/close ∈ [low, high]")
        _repair_ohlc(out)

    # volume: отрицательные → 0, NaN → 0 при non-strict
    if strict and out["volume"].isna().any():
        raise ValueError("NaN в volume при strict=True")
    out["volume"] = out["volume"].fillna(0.0)
    out["volume"] = out["volume"].clip(lower=0.0).astype("float64")

    _assert_monotonic_timestamps(out)

    return out
