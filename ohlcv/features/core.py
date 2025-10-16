# ohlcv/features/core.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

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
    "kama_period": None,
}


def _maybe_map_internal_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Поддержка внутренней схемы C1/C2: o,h,l,c,v,(t) -> open,high,low,close,volume,(turnover)
    cols = set(df.columns)
    if {"o","h","l","c","v"}.issubset(cols):
        mapping = {"o":"open","h":"high","l":"low","c":"close","v":"volume"}
        if "t" in cols:
            mapping["t"] = "turnover"
        df = df.rename(columns=mapping)
    return df


def _ensure_cols(df: pd.DataFrame) -> None:
    req = {"timestamp_ms", "start_time_iso", "open", "high", "low", "close", "volume"}
    miss = sorted(list(req - set(df.columns)))
    if miss:
        raise ValueError(f"Отсутствуют обязательные колонки: {miss}")
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")

def _logret(c: pd.Series) -> pd.Series:
    return np.log(c / c.shift(1)).replace([np.inf, -np.inf], np.nan)

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

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def atr_wilder(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    tr = true_range(h, l, c)
    return ema(tr, n)

def di_adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> Tuple[pd.Series,pd.Series,pd.Series]:
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0.0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0.0)
    atr = atr_wilder(h, l, c, n)
    pdi = 100.0 * ema(plus_dm, n) / (atr + EPS)
    mdi = 100.0 * ema(minus_dm, n) / (atr + EPS)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    adx = ema(dx, n)
    return pdi, mdi, adx

def donchian(h: pd.Series, l: pd.Series, n: int) -> Tuple[pd.Series,pd.Series]:
    hh = h.rolling(n, min_periods=n).max()
    ll = l.rolling(n, min_periods=n).min()
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

def compute_features(df: pd.DataFrame, symbol: str, tf: str, params: Dict[str, Any] | None = None) -> pd.DataFrame:
    params = {**DEFAULTS, **(params or {})}
    _ensure_cols(df)
    df = _maybe_map_internal_cols(df)
    out = df.copy()

    c, h, l, o, v = out["close"], out["high"], out["low"], out["open"], out["volume"]
    logret = _logret(c)

    out["f_ret1"] = c.pct_change()
    out["f_logret1"] = logret
    for n in params["rv_windows"]:
        out[f"f_rv_{n}"] = realized_vol(logret, n)
    out["f_range_pct"] = (h - l) / (c + EPS)
    out["f_body_pct"] = (c - o).abs() / (c + EPS)
    out["f_wick_upper_pct"] = (h - np.maximum(c, o)) / (c + EPS)
    out["f_wick_lower_pct"] = (np.minimum(c, o) - l) / (c + EPS)

    for n in params["ema_windows"]:
        out[f"f_ema_{n}"] = ema(c, n)
        out[f"f_ema_slope_{n}"] = out[f"f_ema_{n}"].diff()
    for n in params["mom_windows"]:
        out[f"f_mom_{n}"] = c - c.shift(n)
    n_rsi = int(params["rsi_period"])
    out[f"f_rsi{n_rsi}"] = rsi_wilder(c, n_rsi)
    n_adx = int(params["adx_period"])
    pdi, mdi, adx = di_adx(h, l, c, n_adx)
    out[f"f_pdi{n_adx}"] = pdi
    out[f"f_mdi{n_adx}"] = mdi
    out[f"f_adx{n_adx}"] = adx
    n_atr = int(params.get("atr_period", 14))
    out[f"f_tr"] = true_range(h, l, c)
    out[f"f_atr_{n_atr}"] = atr_wilder(h, l, c, n_atr)
    out[f"f_atr_pct_{n_atr}"] = out[f"f_atr_{n_atr}"] / (c + EPS)

    n_d = int(params["donch_window"])
    hi, lo = donchian(h, l, n_d)
    out[f"f_donch_h_{n_d}"] = hi
    out[f"f_donch_l_{n_d}"] = lo
    prev_c = c.shift(1)
    brk = np.where((c > hi.shift(1)) & (prev_c <= hi.shift(1)), 1,
                   np.where((c < lo.shift(1)) & (prev_c >= lo.shift(1)), -1, 0))
    out[f"f_donch_break_dir_{n_d}"] = brk
    out[f"f_donch_width_pct_{n_d}"] = (hi - lo) / (c + EPS)

    out["f_range"] = (h - l)
    for n in params["z_windows"]:
        out[f"f_close_z_{n}"] = zscore(c, n)
        out[f"f_range_z_{n}"] = zscore(out["f_range"], n)
        out[f"f_vol_z_{n}"] = zscore(v, n)

    up_mask = c > c.shift(1)
    down_mask = c < c.shift(1)
    for n in params.get("vol_windows", [20]):
        upv = (v * up_mask.astype(int)).rolling(n, min_periods=n).sum()
        dnv = (v * down_mask.astype(int)).rolling(n, min_periods=n).sum()
        out[f"f_upvol_{n}"] = upv
        out[f"f_downvol_{n}"] = dnv
        out[f"f_vol_balance_{n}"] = (upv - dnv) / (upv + dnv + EPS)
    out["f_obv"] = obv(c, v)

    n_vroll = int(params["vwap_roll_window"])
    out[f"f_vwap_roll_{n_vroll}"] = vwap_rolling(out, n_vroll)
    out[f"f_vwap_dev_pct_{n_vroll}"] = (c - out[f"f_vwap_roll_{n_vroll}"]) / (out[f"f_vwap_roll_{n_vroll}"] + EPS)
    out["f_vwap_session"] = vwap_session(out)
    out["f_vwap_session_dev_pct"] = (c - out["f_vwap_session"]) / (out["f_vwap_session"] + EPS)

    out["symbol"] = str(symbol)
    out["tf"] = str(tf)

    fcols = [c for c in out.columns if c.startswith("f_")]
    first_valid = 0
    if fcols:
        mask = out[fcols].notna().all(axis=1)
        first_valid = int(np.argmax(mask.values)) if mask.any() else len(out)
    out["f_valid_from"] = first_valid

    return out
