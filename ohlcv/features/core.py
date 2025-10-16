# ohlcv/features/core.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from .schema import normalize_and_validate

EPS = 1e-12

DEFAULTS = {
    "rv_windows": [20, 60],
    "donch_window": 20,
    "z_windows": [20, 60],
    "vwap_roll_window": 96,
    "rsi_period": 14,
    "adx_period": 14,
    "ema_windows": [20],
    "mom_windows": [20],
}


def _logret(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)


def realized_vol(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n, min_periods=n).std(ddof=1)


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(alpha=1.0 / float(n), adjust=False).mean()


def rsi_wilder(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = ema(up, n)
    avg_loss = ema(down, n)
    rs = avg_gain / (avg_loss + EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = true_range(high, low, close)
    return ema(tr, n)


def di_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)
    atr = atr_wilder(high, low, close, n)
    pdi = 100.0 * ema(plus_dm, n) / (atr + EPS)
    mdi = 100.0 * ema(minus_dm, n) / (atr + EPS)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    adx = ema(dx, n)
    return pdi, mdi, adx


def donchian(high: pd.Series, low: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    return hh, ll


def zscore(x: pd.Series, n: int) -> pd.Series:
    m = x.rolling(n, min_periods=n).mean()
    s = x.rolling(n, min_periods=n).std(ddof=1)
    return (x - m) / (s + EPS)


def vwap_rolling(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    s_pv = pv.rolling(n, min_periods=n).sum()
    s_v = df["volume"].rolling(n, min_periods=n).sum()
    return s_pv / (s_v + EPS)


def vwap_session(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["start_time_iso"], utc=True)
    day_key = ts.dt.strftime("%Y-%m-%d")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    s_pv = pv.groupby(day_key).cumsum()
    s_v = df["volume"].groupby(day_key).cumsum()
    return s_pv / (s_v + EPS)


def obv(close: pd.Series, vol: pd.Series) -> pd.Series:
    sign = (close.diff()).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))
    return (sign * vol).cumsum()


def compute_features(
    df: pd.DataFrame, symbol: str, tf: str, params: Dict[str, Any] | None = None
) -> pd.DataFrame:
    params = {**DEFAULTS, **(params or {})}
    strict = bool(params.get("strict", False))

    # Нормализация входа и проверка схемы
    df = normalize_and_validate(df, strict=strict)

    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    open_ = out["open"]
    vol = out["volume"]

    logret = _logret(close)

    # Доходности/волатильность
    out["f_ret1"] = close.pct_change()
    out["f_logret1"] = logret
    for n in params["rv_windows"]:
        out[f"f_rv_{n}"] = realized_vol(logret, n)
    out["f_range_pct"] = (high - low) / (close + EPS)
    out["f_body_pct"] = (close - open_).abs() / (close + EPS)
    out["f_wick_upper_pct"] = (high - np.maximum(close, open_)) / (close + EPS)
    out["f_wick_lower_pct"] = (np.minimum(close, open_) - low) / (close + EPS)

    # Тренд/моментум
    for n in params["ema_windows"]:
        out[f"f_ema_{n}"] = ema(close, n)
        out[f"f_ema_slope_{n}"] = out[f"f_ema_{n}"].diff()
    for n in params["mom_windows"]:
        out[f"f_mom_{n}"] = close - close.shift(n)
    n_rsi = int(params["rsi_period"])
    out[f"f_rsi{n_rsi}"] = rsi_wilder(close, n_rsi)
    n_adx = int(params["adx_period"])
    pdi, mdi, adx = di_adx(high, low, close, n_adx)
    out[f"f_pdi{n_adx}"] = pdi
    out[f"f_mdi{n_adx}"] = mdi
    out[f"f_adx{n_adx}"] = adx
    n_atr = int(params.get("atr_period", 14))
    out["f_tr"] = true_range(high, low, close)
    out[f"f_atr_{n_atr}"] = atr_wilder(high, low, close, n_atr)
    out[f"f_atr_pct_{n_atr}"] = out[f"f_atr_{n_atr}"] / (close + EPS)

    # Donchian
    n_d = int(params["donch_window"])
    hi, lo = donchian(high, low, n_d)
    out[f"f_donch_h_{n_d}"] = hi
    out[f"f_donch_l_{n_d}"] = lo
    prev_close = close.shift(1)
    brk = np.where(
        (close > hi.shift(1)) & (prev_close <= hi.shift(1)),
        1,
        np.where((close < lo.shift(1)) & (prev_close >= lo.shift(1)), -1, 0),
    )
    out[f"f_donch_break_dir_{n_d}"] = brk
    out[f"f_donch_width_pct_{n_d}"] = (hi - lo) / (close + EPS)

    # Z‑scores
    out["f_range"] = high - low
    for n in params["z_windows"]:
        out[f"f_close_z_{n}"] = zscore(close, n)
        out[f"f_range_z_{n}"] = zscore(out["f_range"], n)
        out[f"f_vol_z_{n}"] = zscore(vol, n)

    # Объёмы
    up_mask = close > close.shift(1)
    down_mask = close < close.shift(1)
    for n in params.get("vol_windows", [20]):
        upv = (vol * up_mask.astype(int)).rolling(n, min_periods=n).sum()
        dnv = (vol * down_mask.astype(int)).rolling(n, min_periods=n).sum()
        out[f"f_upvol_{n}"] = upv
        out[f"f_downvol_{n}"] = dnv
        out[f"f_vol_balance_{n}"] = (upv - dnv) / (upv + dnv + EPS)
    out["f_obv"] = obv(close, vol)

    # VWAP
    n_vroll = int(params["vwap_roll_window"])
    out[f"f_vwap_roll_{n_vroll}"] = vwap_rolling(out, n_vroll)
    out[f"f_vwap_dev_pct_{n_vroll}"] = (close - out[f"f_vwap_roll_{n_vroll}"]) / (
        out[f"f_vwap_roll_{n_vroll}"] + EPS
    )
    out["f_vwap_session"] = vwap_session(out)
    out["f_vwap_session_dev_pct"] = (close - out["f_vwap_session"]) / (out["f_vwap_session"] + EPS)

    # Служебные поля
    out["symbol"] = str(symbol)
    out["tf"] = str(tf)

    # Первая валидная строка по всем f_*
    fcols = [c for c in out.columns if c.startswith("f_")]
    first_valid = 0
    if fcols:
        mask = out[fcols].notna().all(axis=1)
        first_valid = int(np.argmax(mask.values)) if mask.any() else len(out)
    out["f_valid_from"] = first_valid

    return out
