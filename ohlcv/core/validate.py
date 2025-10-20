# ohlcv/core/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MINUTE_MS",
    "ValidateStats",
    "normalize_ohlcv_1m",
    "align_and_flag_gaps",
]

# ---------------- constants ----------------
MINUTE_MS: int = 60_000


# ---------------- schema & typing ----------------

def _lower_dict(keys: pd.Index) -> Dict[str, str]:
    """Построить сопоставление реальных имён столбцов к нижнему регистру."""
    return {k: k.lower() for k in map(str, keys)}


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Коэрсия входа к схеме C1: ts,o,h,l,c,v[,t].
    Допускаются синонимы: timestamp_ms|timestamp|time→ts, open→o, high→h, low→l,
    close→c, volume→v, turnover→t. Лишние столбцы отбрасываются.
    """
    if df is None or df.empty:
        # создаём пустой каркас нужной схемы
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
    # нормализуем регистр
    lower_map = _lower_dict(df.columns)
    dfl = df.rename(columns=lower_map)
    # гибкое сопоставление синонимов
    aliases = {
        "timestamp_ms": "ts",
        "timestamp": "ts",
        "time": "ts",
        "ts": "ts",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
        "turnover": "t",
        "amount": "t",
        "quote_volume": "t",
        "o": "o",
        "h": "h",
        "l": "l",
        "c": "c",
        "v": "v",
        "t": "t",
    }
    mapped_cols: Dict[str, str] = {}
    for col in dfl.columns:
        tgt = aliases.get(col)
        if tgt:
            mapped_cols[col] = tgt
    dfl = dfl.rename(columns=mapped_cols)
    required = {"ts", "o", "h", "l", "c", "v"}
    missing = required - set(dfl.columns)
    if missing:
        raise ValueError(f"Missing required columns after coercion: {sorted(missing)}")
    # сохраняем t если есть
    keep = ["ts", "o", "h", "l", "c", "v"] + (["t"] if "t" in dfl.columns else [])
    return dfl[keep]


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        # сформируем корректные типы даже для пустых
        for col, typ in {"ts": "int64", "o": "float64", "h": "float64", "l": "float64", "c": "float64", "v": "float64"}.items():
            if col not in out.columns:
                out[col] = pd.Series(dtype=typ)
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(typ)
        if "t" in out.columns:
            out["t"] = pd.to_numeric(out.get("t"), errors="coerce").astype("float64")
        return out
    out = df.copy()
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for col in ("o", "h", "l", "c", "v"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    if "t" in out.columns:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("float64")
    return out


def _floor_to_minute(ts_ms: np.ndarray) -> np.ndarray:
    return (ts_ms // MINUTE_MS) * MINUTE_MS


# ---------------- alignment ----------------

def align_and_flag_gaps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Выровнять по минутной сетке UTC с контрактом C1.
    Вставлять gap-бары ТОЛЬКО там, где в исходнике отсутствовала минута.
    Правило заполнения для gap: o=h=l=c=prev_close, v=0, и если есть t — t=0. is_gap=True.
    Возвращает (df_aligned, gap_count). Предполагается уникальность ts; при дублях берётся последняя запись.
    """
    if df is None or df.empty:
        cols = ["ts", "o", "h", "l", "c", "v", "is_gap"]
        out = pd.DataFrame(columns=cols)
        out = out.astype({"ts": "int64", "o": "float64", "h": "float64", "l": "float64", "c": "float64", "v": "float64", "is_gap": "bool"})
        return out, 0

    # гарантируем уникальные ts
    base = df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    start = int(base["ts"].iloc[0])
    end = int(base["ts"].iloc[-1])
    full_index = np.arange(start, end + MINUTE_MS, MINUTE_MS, dtype=np.int64)

    # отметим исходные метки времени
    orig_ts = base["ts"].to_numpy(dtype=np.int64)
    is_missing_mask = ~np.isin(full_index, orig_ts)

    # каркас и слияние
    scaffold = pd.DataFrame({"ts": full_index})
    aligned = scaffold.merge(base, on="ts", how="left", suffixes=("", ""))

    # prev close для заполнения gap
    prev_close = aligned["c"].ffill()
    gap_idx = np.where(is_missing_mask)[0]
    if gap_idx.size:
        aligned.loc[gap_idx, "c"] = prev_close.loc[gap_idx]
        aligned.loc[gap_idx, ["o", "h", "l"]] = aligned.loc[gap_idx, ["c", "c", "c"]].values
        aligned.loc[gap_idx, "v"] = 0.0
        if "t" in aligned.columns:
            aligned.loc[gap_idx, "t"] = 0.0

    aligned["is_gap"] = False
    if gap_idx.size:
        aligned.loc[gap_idx, "is_gap"] = True

    # типы и порядок
    aligned = aligned.astype({
        "ts": "int64",
        "o": "float64",
        "h": "float64",
        "l": "float64",
        "c": "float64",
        "v": "float64",
        "is_gap": "bool",
    })
    if "t" in aligned.columns:
        aligned["t"] = pd.to_numeric(aligned["t"], errors="coerce").astype("float64")

    gaps = int(is_missing_mask.sum())

    # канонический порядок колонок
    order = ["ts", "o", "h", "l", "c", "v"] + (["t"] if "t" in aligned.columns else []) + ["is_gap"]
    return aligned[order], gaps


# ---------------- stats ----------------

@dataclass(frozen=True)
class ValidateStats:
    rows_in: int
    rows_out: int
    dup_removed: int
    misaligned_fixed: int
    gap_bars: int
    gap_pct: float
    min_ts: int
    max_ts: int
    span_minutes: int


def _calc_stats(before: pd.DataFrame | None, after: pd.DataFrame, dup_removed: int, misaligned_fixed: int, gaps: int) -> ValidateStats:
    rows_in = int(0 if before is None else len(before))
    rows_out = int(len(after))
    if rows_out:
        min_ts = int(after["ts"].iloc[0])
        max_ts = int(after["ts"].iloc[-1])
        span_minutes = int((max_ts - min_ts) // MINUTE_MS + 1)
    else:
        min_ts = max_ts = span_minutes = 0
    gap_pct = (gaps / max(1, span_minutes)) * 100.0
    return ValidateStats(
        rows_in=rows_in,
        rows_out=rows_out,
        dup_removed=int(dup_removed),
        misaligned_fixed=int(misaligned_fixed),
        gap_bars=int(gaps),
        gap_pct=float(gap_pct),
        min_ts=min_ts,
        max_ts=max_ts,
        span_minutes=span_minutes,
    )


# ---------------- public API ----------------

def normalize_ohlcv_1m(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidateStats]:
    """
    Контракт C1: возвращает (normalized_df, ValidateStats).
    Шаги:
      1) Коэрсия схемы к ts,o,h,l,c,v[,t] (case-insensitive с синонимами).
      2) Типы: ts=int64(ms), цены/объём/turnover=float64.
      3) Починка мисалайна ts → floor к началу минуты.
      4) Дедуп по ts (берётся последняя запись).
      5) Выравнивание минутной сеткой; вставка gap-только для отсутствовавших минут, is_gap=True.
    Гарантии:
      — ts монотонен с шагом 60_000;
      — gap-бары: o=h=l=c=prev_close, v=0, t=0(если колонка существует), is_gap=True.
    """
    raw = _coerce_schema(df)
    raw = _ensure_types(raw)

    # фиксация мисалайна
    ts_arr = raw["ts"].to_numpy(dtype=np.int64)
    floored = _floor_to_minute(ts_arr)
    misaligned_fixed = int(np.count_nonzero(floored != ts_arr))
    raw = raw.copy()
    raw["ts"] = floored

    # дедуп
    before_len = len(raw)
    dedup = raw.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    dup_removed = int(before_len - len(dedup))

    # выравнивание
    normalized, gaps = align_and_flag_gaps(dedup)

    stats = _calc_stats(df, normalized, dup_removed, misaligned_fixed, gaps)
    return normalized, stats
