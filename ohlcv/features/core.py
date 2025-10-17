# ohlcv/features/core.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .schema import normalize_and_validate

EPS = 1e-12

# Параметры по умолчанию для набора фич
DEFAULTS: Dict[str, Any] = {
    "rv_windows": [20, 60],
    "donch_window": 20,
    "z_windows": [20, 60],
    "ema_windows": [5, 12, 26, 60],
    "sma_windows": [5, 20, 60],
    "atr_window": 14,
    "adx_window": 14,
    "strict": False,
}


# ------------------------- базовые индикаторы -------------------------


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    alpha = 2.0 / (n + 1.0)
    return x.ewm(alpha=alpha, adjust=False).mean()


def rolling_std(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n, min_periods=1).std(ddof=0)


def true_range(high: pd.Series, low: pd.Series, close_prev: pd.Series) -> pd.Series:
    prev_close = close_prev.shift(1)
    a = (high - low).abs()
    b = (high - prev_close).abs()
    c = (low - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1).fillna(0.0)


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / float(n), adjust=False).mean()


def di_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0.0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0.0)
    atr = atr_wilder(high, low, close, n).replace(0.0, np.nan)
    pdi = 100.0 * ema(plus_dm, n) / atr
    mdi = 100.0 * ema(minus_dm, n) / atr
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi + EPS)).fillna(0.0)
    adx = ema(dx, n)
    return pdi.fillna(0.0), mdi.fillna(0.0), adx.fillna(0.0)


def zscore(x: pd.Series, n: int) -> pd.Series:
    mu = sma(x, n)
    sd = rolling_std(x, n)
    return (x - mu) / (sd.replace(0.0, np.nan))


def donchian(high: pd.Series, low: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    hh = high.rolling(n, min_periods=1).max()
    ll = low.rolling(n, min_periods=1).min()
    return hh, ll


def realized_variance(close: pd.Series, n: int) -> pd.Series:
    r = np.log(close.replace(0.0, np.nan)).diff().fillna(0.0)
    return r.pow(2).rolling(n, min_periods=1).sum()


# ------------------------- основной билдер фич -------------------------


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def compute_features(
    df: pd.DataFrame,
    symbol: str | None = None,
    tf: str | None = None,
    params: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(params or {})}

    # Нормализация входной схемы и строгая проверка при необходимости
    norm = normalize_and_validate(df.copy(), strict=bool(cfg.get("strict", False)))

    # Базовые серии
    high = pd.to_numeric(norm["high"], errors="coerce").astype(float)
    low = pd.to_numeric(norm["low"], errors="coerce").astype(float)
    close = pd.to_numeric(norm["close"], errors="coerce").astype(float)
    vol = pd.to_numeric(norm["volume"], errors="coerce").astype(float)

    out = pd.DataFrame(index=norm.index)

    # True range и ATR
    out["f_tr"] = true_range(high, low, close)
    n_atr = int(cfg["atr_window"])
    out[f"f_atr_{n_atr}"] = atr_wilder(high, low, close, n_atr)

    # DI/ADX
    n_adx = int(cfg["adx_window"])
    pdi, mdi, adx = di_adx(high, low, close, n_adx)
    out[f"f_pdi_{n_adx}"] = pdi
    out[f"f_mdi_{n_adx}"] = mdi
    out[f"f_adx_{n_adx}"] = adx

    # EMA/SMA
    for n in cfg["ema_windows"]:
        out[f"f_ema_close_{int(n)}"] = ema(close, int(n))
    for n in cfg["sma_windows"]:
        out[f"f_sma_close_{int(n)}"] = sma(close, int(n))
        out[f"f_sma_vol_{int(n)}"] = sma(vol, int(n))

    # Z-score
    for n in cfg["z_windows"]:
        out[f"f_z_close_{int(n)}"] = zscore(close, int(n))

    # Дончиан
    n_d = int(cfg["donch_window"])
    hh, ll = donchian(high, low, n_d)
    out[f"f_donch_hh_{n_d}"] = hh
    out[f"f_donch_ll_{n_d}"] = ll
    out[f"f_donch_width_{n_d}"] = hh - ll

    # Реализованная дисперсия (волатильность)
    for n in cfg["rv_windows"]:
        out[f"f_rv_{int(n)}"] = realized_variance(close, int(n))

    # Возвраты
    out["f_ret"] = close.pct_change().fillna(0.0)
    out["f_logret"] = np.log(close.replace(0.0, np.nan)).diff().fillna(0.0)

    # Служебные поля
    out["symbol"] = str(symbol) if symbol is not None else ""
    out["tf"] = str(tf) if tf is not None else ""

    # Первая валидная строка по всем f_*
    fcols = [c for c in out.columns if c.startswith("f_")]
    first_valid = 0
    if fcols:
        mask = out[fcols].notna().all(axis=1)
        first_valid = int(np.argmax(mask.values)) if mask.any() else len(out)
    out["f_valid_from"] = first_valid

    return out
