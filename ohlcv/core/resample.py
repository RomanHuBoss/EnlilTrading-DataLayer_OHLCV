# ohlcv/core/resample.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Разрешённые целевые таймфреймы
_RULES: Dict[str, Tuple[str, int]] = {
    "5m": ("5T", 5),
    "15m": ("15T", 15),
    "1h": ("1H", 60),
}


def _require_1m_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise TypeError("ожидается tz-aware DatetimeIndex в UTC")
    # Разрешаем пропуски, но шаг каждой строки должен быть ровно 1 минута относительно своего собственного таймстемпа
    # То есть индекс должен быть минутным (секунды/милисекунды = 0)
    if not (df.index.second == 0).all() or not (df.index.nanosecond == 0).all():
        raise ValueError("индекс должен быть выровнен по минутам (секунды/нс = 0)")


def _has_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def resample_1m_to(df1m: pd.DataFrame, dst_tf: str, *, allow_partial: bool = False) -> pd.DataFrame:
    """Агрегирует минутные бары в {5m, 15m, 1h} с якорем по UTC-эпохе.

    Вход: df1m с индексом ts (UTC) и колонками o/h/l/c/v (+t, +is_gap опционально).
    Выход: агрегированный DataFrame с теми же колонками и колонкой is_gap, отражающей
           наличие пропусков внутри окна (или поднятую из входного is_gap).

    Правила агрегирования: open=first, high=max, low=min, close=last, volume=sum, t=sum, is_gap=max|incomplete.
    Частичные окна в начале/хвосте по умолчанию удаляются (allow_partial=False).
    """
    if dst_tf not in _RULES:
        raise ValueError(f"неподдерживаемый целевой таймфрейм: {dst_tf}")
    rule, minutes = _RULES[dst_tf]

    _require_1m_index(df1m)

    cols = [c for c in ("o", "h", "l", "c", "v", "t", "is_gap") if _has_col(df1m, c)]
    df = df1m[cols].copy()

    # Булевы/числовые типы
    if _has_col(df, "is_gap"):
        df["is_gap"] = df["is_gap"].astype(bool)

    # Группировка с якорем по эпохе
    g = df.groupby(pd.Grouper(freq=rule, label="left", origin="epoch"))

    # Подсчёт наличия валидного бара внутри окна (все OHLC не NaN)
    present = (~df[[c for c in ["o", "h", "l", "c"] if c in df.columns]].isna().any(axis=1))
    present_count = present.groupby(pd.Grouper(freq=rule, label="left", origin="epoch")).sum(min_count=1)

    agg_map = {}
    if _has_col(df, "o"): agg_map["o"] = "first"
    if _has_col(df, "h"): agg_map["h"] = "max"
    if _has_col(df, "l"): agg_map["l"] = "min"
    if _has_col(df, "c"): agg_map["c"] = "last"
    if _has_col(df, "v"): agg_map["v"] = "sum"
    if _has_col(df, "t"): agg_map["t"] = "sum"
    if _has_col(df, "is_gap"): agg_map["is_gap"] = "max"

    out = g.agg(agg_map)

    # Компактная проверка пустоты
    if out.empty:
        # возвращаем пустой фрейм с ожидаемыми колонками в правильном порядке
        want = [c for c in ["o", "h", "l", "c", "v", "t", "is_gap"] if c in agg_map]
        return pd.DataFrame(columns=want, index=out.index).astype({c: out.dtypes.get(c, "float64") for c in want})

    # Вычисление флага неполного окна
    expected = minutes  # количество минут в окне
    incomplete = (present_count.fillna(0).astype(int) < expected)

    # Признак gap: максимум входных флагов или неполное окно
    if _has_col(out, "is_gap"):
        out["is_gap"] = out["is_gap"].astype(bool) | incomplete.reindex(out.index, fill_value=False).astype(bool)
    else:
        out["is_gap"] = incomplete.reindex(out.index, fill_value=False).astype(bool)

    # Удаление частичных окон на краях, если запрещено
    if not allow_partial and len(out) > 0:
        mask_full = ~out["is_gap"].astype(bool)
        # Крайние неполные окна часто только на границах; удаляем первую/последнюю, если они неполные
        if not mask_full.iloc[0]:
            out = out.iloc[1:]
            mask_full = mask_full.iloc[1:]
        if len(out) > 0 and not mask_full.iloc[-1]:
            out = out.iloc[:-1]

    # Приведение типов и имя индекса
    out.index = pd.DatetimeIndex(out.index, tz="UTC", name="ts")

    # Порядок столбцов
    order = [c for c in ["o", "h", "l", "c", "v", "t", "is_gap"] if c in out.columns]
    out = out[order]

    return out
