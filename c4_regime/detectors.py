from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS_D1 = {"f_adx14", "f_pdi14", "f_mdi14"}


def _require(df: pd.DataFrame, cols: set[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{missing}")


# --- D1: ADX тренд / флэт ---


def d1_adx(
    df: pd.DataFrame, T_adx: float, T_di: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(df, REQUIRED_COLS_D1)
    adx = df["f_adx14"].to_numpy()
    pdi = df["f_pdi14"].to_numpy()
    mdi = df["f_mdi14"].to_numpy()

    strong = adx >= T_adx
    di_gap = np.abs(pdi - mdi) >= T_di

    trend = (strong & di_gap).astype(np.int8)
    flat = (~(strong & di_gap)).astype(np.int8)

    T_max = max(T_adx + 18.0, T_adx + 1.0)  # мягкая верхняя отсечка
    s = (adx - T_adx) / (T_max - T_adx)
    s = np.clip(s, 0.0, 1.0)
    return trend, flat, s


# --- D2: Donchian ширина / импульс ---


def d2_donch(
    df: pd.DataFrame,
    window: int,
    quantile_window: int,
    q_low_pct: float,
    q_high_pct: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Берём готовую ширину из C3, либо fallback по (H-L)/C
    col = f"f_donch_width_pct_{window}"
    if col in df.columns:
        w = df[col].astype(float)
    else:
        # Fallback без зависимости от C3 Donchian — безопасно, но грубее
        for c in ("high", "low", "close"):
            if c not in df.columns:
                raise KeyError(f"['{col}' or high/low/close]")
        w = (df["high"] - df["low"]) / df["close"].replace(0.0, np.nan)
        w = w.fillna(0.0)

    # Квантили на скользящем окне
    Wq = max(int(quantile_window), 10)
    ql = w.rolling(Wq, min_periods=Wq).quantile(q_low_pct / 100.0)
    qh = w.rolling(Wq, min_periods=Wq).quantile(q_high_pct / 100.0)

    # Импульс по ширине — не даём тренду при сужении диапазона
    slope_nonneg = w.diff().fillna(0.0) >= 0.0

    trend = ((w >= qh) & slope_nonneg).astype(np.int8).to_numpy()
    flat = (w <= ql).astype(np.int8).to_numpy()

    denom = (qh - ql).replace(0.0, np.nan)
    s = ((w - ql) / denom).clip(0.0, 1.0).fillna(0.0).to_numpy()
    return trend, flat, s


# --- D4: HMM по дневной RV (через контекст или fallback) ---


def d4_hmm_rv(
    work_df: pd.DataFrame,
    context: Dict | pd.DataFrame | None,
    n_states: int,
    calm_threshold: float,
    allow_fallback_daily: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (high_rv_flag, s_hmm_rv, rv_daily_series_aligned).
    Если есть ctx_rv_daily_state/ctx_rv_daily_pct — используем их.
    Иначе позволяется резервная агрегация по рабочему ТФ (дневные лог-реты → std).
    """
    # Попытка взять готовый контекст C5
    if isinstance(context, pd.DataFrame):
        ctx = context
    elif isinstance(context, dict):
        ctx = pd.DataFrame(context)
    else:
        ctx = None

    if ctx is not None and "ctx_rv_daily_pct" in ctx.columns:
        pct = ctx["ctx_rv_daily_pct"].astype(float).to_numpy()
        state_col = ctx.get("ctx_rv_daily_state")
        if state_col is not None:
            state = state_col.astype(str).fillna("")
            high = ((state == "volatile") | (state == "extreme")).to_numpy()
        else:
            # эвристика по перцентилю
            high = (pct >= 0.90).astype(np.int8)
        s_calm = (1.0 - pct).clip(0.0, 1.0)
        return (
            high.astype(np.int8),
            s_calm.astype(float),
            ctx.get("ctx_rv_daily", pd.Series(index=ctx.index, dtype=float)).to_numpy(),
        )

    # Резервная агрегация
    if not allow_fallback_daily:
        # Без контекста и без права fallback — нет оценки
        n = len(work_df)
        return np.zeros(n, dtype=np.int8), np.zeros(n, dtype=float), np.zeros(n, dtype=float)

    # Рассчитываем дневную RV из рабочей серии (UTC дни)
    df = work_df[["start_time_iso", "close"]].copy()
    ts = pd.to_datetime(df["start_time_iso"], utc=True)
    # Берём последний close каждого UTC-дня
    day_close = (
        df.set_index(ts).resample("1D", label="right", closed="right")["close"].last().dropna()
    )
    logret_d = np.log(day_close / day_close.shift(1)).dropna()
    rv = logret_d.rolling(20, min_periods=20).std(ddof=1)
    # Раскладываем на рабочий индекс через asof
    rv_df = rv.to_frame("rv_daily").reset_index().rename(columns={"index": "day_end"})
    work_ts = pd.to_datetime(work_df["start_time_iso"], utc=True)
    # Присваиваем каждому бару последнюю закрытую дневную RV
    rv_aligned = (
        pd.merge_asof(
            pd.DataFrame({"t": work_ts}),
            rv_df,
            left_on="t",
            right_on="day_end",
            direction="backward",
            allow_exact_matches=False,
        )["rv_daily"]
        .fillna(method="ffill")
        .fillna(0.0)
    )
    # Перцентиль на окне 252 дней (если доступно)
    rv_series = rv_aligned.to_numpy()
    rv_s = pd.Series(rv_series)
    pct = (rv_s.rank(method="min") / rv_s.count()).fillna(0.0).to_numpy()
    high = (pct >= 0.90).astype(np.int8)
    s_calm = (1.0 - pct).clip(0.0, 1.0)
    return high, s_calm, rv_series
