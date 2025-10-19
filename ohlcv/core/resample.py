# ohlcv/core/resample.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .validate import normalize_ohlcv_1m, align_and_flag_gaps

__all__ = ["resample_1m_to"]


def _dst_freq(tf: str) -> Tuple[str, int]:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        n = int(tf[:-1])
        return f"{n}T", n
    if tf.endswith("h"):
        n = int(tf[:-1])
        return f"{n}H", 60 * n
    raise ValueError(f"неподдерживаемый целевой tf: {tf}")


def _to_right_label_ms(idx: pd.DatetimeIndex) -> pd.Series:
    """Преобразует DatetimeIndex групп к правой границе в миллисекундах."""
    return (idx.view("int64") // 1_000_000).astype("int64")


def resample_1m_to(df_1m: pd.DataFrame, dst_tf: str) -> pd.DataFrame:
    """Ресемплинг 1m → dst_tf (5m/15m/1h) по правой границе окна.

    Вход: DataFrame 1m со столбцами o,h,l,c,v[,t] и/или колонкой ts либо DatetimeIndex UTC.
    Алгоритм:
      1) normalize_ohlcv_1m → align_and_flag_gaps(strict=True) для полной минутной сетки и is_gap по минутам.
      2) Группировка по dst_tf (правые границы).
      3) Агрегаты: o=first, h=max, l=min, c=last, v=sum, t=sum; is_gap=any или недобор минут.
      4) Вывод: ts=int64 (мс, правая граница), канонический порядок столбцов.
    """
    if df_1m is None or len(df_1m) == 0:
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v", "t", "is_gap"]).astype({"ts": "int64"})

    # Шаг 1: нормализация и выравнивание минутной сетки
    norm = normalize_ohlcv_1m(df_1m)
    aligned, _stats = align_and_flag_gaps(norm, strict=True)

    freq, nmin = _dst_freq(dst_tf)

    # Группировка правыми границами
    g = aligned.groupby(pd.Grouper(freq=freq, label="right", closed="right"))

    out = pd.DataFrame(index=g.size().index)

    # Агрегаты OHLCV
    out["o"] = g["o"].first()
    out["h"] = g["h"].max()
    out["l"] = g["l"].min()
    out["c"] = g["c"].last()
    out["v"] = g["v"].sum()
    if "t" in aligned.columns:
        out["t"] = g["t"].sum()

    # is_gap окна: если любая минута внутри окна is_gap=True ИЛИ окно неполное по числу минут
    # (неполные окна возникают на краях диапазона или при дырках до выравнивания; последние уже помечены минутными is_gap)
    any_gap = g["is_gap"].any()

    # число минут в каждом окне
    cnt = g.size().rename("count").astype("int64")
    out["is_gap"] = any_gap.reindex(out.index).fillna(False).astype(bool)
    out.loc[cnt < nmin, "is_gap"] = True

    # Индекс → ts (правая граница окна)
    out_ts = _to_right_label_ms(out.index)
    out = out.reset_index(drop=True)
    out.insert(0, "ts", out_ts.values)

    # Типы
    for c in ["o", "h", "l", "c", "v"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    if "t" in out.columns:
        out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("float64")
    out["is_gap"] = out["is_gap"].astype(bool)
    out["ts"] = out["ts"].astype("int64")

    # Канонический порядок
    cols = ["ts", "o", "h", "l", "c", "v"]
    if "t" in out.columns:
        cols.append("t")
    cols.append("is_gap")
    out = out[cols]

    # Удалить полностью пустые окна (в теории не встречаются после strict=True, оставлено на всякий случай)
    empty_mask = out[[c for c in ["o", "h", "l", "c"]]].isna().all(axis=1)
    if empty_mask.any():
        # Синтетика: o=h=l=c=prev_close, v=0, is_gap=True
        prev_c = out["c"].ffill()
        for c in ["o", "h", "l", "c"]:
            out.loc[empty_mask, c] = prev_c.loc[empty_mask]
        out.loc[empty_mask, "v"] = 0.0
        if "t" in out.columns:
            out.loc[empty_mask, "t"] = 0.0
        out.loc[empty_mask, "is_gap"] = True

    # Финальная сортировка по ts, дедуп по ts
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)

    return out
