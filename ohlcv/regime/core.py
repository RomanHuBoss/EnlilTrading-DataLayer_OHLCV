from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["RegimeConfig", "infer_regime"]


# =============================
# Конфиг
# =============================
@dataclass(frozen=True)
class RegimeConfig:
    # Ансамбль
    vote_K: int = 3
    N_conf: Dict[str, int] = None  # сколько баров требуется для уверенности
    N_cooldown: Dict[str, int] = None  # сколько баров заморозки после смены

    # D1: ADX/DI
    T_adx: int = 22
    T_di: int = 5
    adx_thr: float = 18.0

    # D2: Donchian
    donch_window: int = 20

    # D3: Change‑point proxy (z‑score |logret|)
    z_window: int = 64
    z_thr: float = 3.5

    # D4: «HMM‑proxy» по realized volatility
    N_hmm: int = 180
    calm_threshold: float = 0.60  # квантиль RV ниже которой считаем «calm»

    def __post_init__(self):
        object.__setattr__(self, "N_conf", self.N_conf or {"5m": 5, "15m": 5, "1h": 3})
        object.__setattr__(self, "N_cooldown", self.N_cooldown or {"5m": 10, "15m": 8, "1h": 3})


# =============================
# Вспомогательные
# =============================

def _ensure_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        ts = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
        out["ts"] = (ts.view("int64") // 1_000_000).astype("int64")
    if "ts" not in out.columns:
        raise ValueError("ожидался столбец 'ts' или DatetimeIndex")
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ["o", "h", "l", "c"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return out[["ts", "o", "h", "l", "c"]]


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int, n_di: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = h.diff()
    dn_move = (-l.diff())
    pdm = up_move.where((up_move > dn_move) & (up_move > 0.0), 0.0)
    mdm = dn_move.where((dn_move > up_move) & (dn_move > 0.0), 0.0)
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = _ema(tr, n)
    pdi = 100.0 * _ema(pdm, n_di) / atr.replace(0.0, np.nan)
    mdi = 100.0 * _ema(mdm, n_di) / atr.replace(0.0, np.nan)
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan))
    adx = _ema(dx, n)
    return pdi, mdi, adx


def _donch_breakout(c: pd.Series, h: pd.Series, l: pd.Series, n: int) -> pd.Series:
    hh = h.rolling(n, min_periods=max(2, n // 4)).max().shift(1)
    ll = l.rolling(n, min_periods=max(2, n // 4)).min().shift(1)
    sig = pd.Series(0.0, index=c.index)
    sig = sig.where(~(c > hh), 1.0)
    sig = sig.where(~(c < ll), -1.0)
    return sig


def _zscore(x: pd.Series, n: int) -> pd.Series:
    mu = x.rolling(n, min_periods=max(5, n // 5)).mean()
    sd = x.rolling(n, min_periods=max(5, n // 5)).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _rv(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n, min_periods=max(5, n // 5)).std(ddof=1)


# =============================
# Детекторы
# =============================

def _D1_adx_trend(df: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    pdi, mdi, adx = _adx(df["h"], df["l"], df["c"], cfg.T_adx, cfg.T_di)
    dir_raw = np.sign((pdi - mdi).fillna(0.0))
    strong = (adx >= cfg.adx_thr).astype(int)
    sig = (dir_raw * strong).astype(float).fillna(0.0)
    return sig


def _D2_donch(df: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    return _donch_breakout(df["c"], df["h"], df["l"], cfg.donch_window)


def _D3_changepoint(df: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    with np.errstate(divide="ignore"):
        lr = np.log(df["c"]).diff()
    z = _zscore(lr.abs(), cfg.z_window)
    event = (z > cfg.z_thr).astype(int)
    sig = np.sign(lr).fillna(0.0) * event
    return sig.replace(np.nan, 0.0)


def _D4_hmm_proxy(df: pd.DataFrame, cfg: RegimeConfig) -> pd.Series:
    with np.errstate(divide="ignore"):
        lr = np.log(df["c"]).diff()
    rv = _rv(lr, cfg.N_hmm)
    q = rv.rolling(cfg.N_hmm, min_periods=max(8, cfg.N_hmm // 6)).quantile(cfg.calm_threshold)
    calm = (rv <= q).astype(int)
    return calm.fillna(0).astype(float)


# =============================
# Ансамбль/пост‑процессинг
# =============================

def _stable_sign(s: pd.Series, n_conf: int) -> pd.Series:
    x = s.fillna(0.0).astype(float)
    same = np.sign(x).replace(0.0, np.nan)
    grp = (same != same.shift()).cumsum()
    run = same.groupby(grp).transform(lambda g: np.arange(1, len(g) + 1)).fillna(0)
    ok = (run >= n_conf).astype(int)
    return (np.sign(x) * ok).astype(float)


def _apply_cooldown(sig: pd.Series, n_cd: int) -> pd.Series:
    s = sig.fillna(0.0).astype(float).values
    out = np.zeros_like(s, dtype=float)
    cd = 0
    prev = 0.0
    for i, v in enumerate(s):
        if v != 0.0 and np.sign(v) != np.sign(prev):
            cd = n_cd
        if cd > 0:
            out[i] = 0.0
            cd -= 1
        else:
            out[i] = v
        prev = out[i] if out[i] != 0.0 else prev
    return pd.Series(out, index=sig.index)


def infer_regime(df_in: pd.DataFrame, tf: str, cfg: Optional[RegimeConfig] = None) -> pd.DataFrame:
    cfg = cfg or RegimeConfig()
    df = _ensure_input(df_in)

    for col in ["h", "l"]:
        if df[col].isna().all():
            df[col] = df["c"]

    d1 = _D1_adx_trend(df, cfg)
    d2 = _D2_donch(df, cfg)
    d3 = _D3_changepoint(df, cfg)
    d4 = _D4_hmm_proxy(df, cfg)

    votes = pd.DataFrame({"d1": d1, "d2": d2}, index=df.index)

    n_conf = cfg.N_conf.get(tf, 5)
    v1 = _stable_sign(votes["d1"], n_conf)
    v2 = _stable_sign(votes["d2"], n_conf)

    sgn_sum = v1.add(v2, fill_value=0.0)
    score = (v1.ne(0).astype(int) + v2.ne(0).astype(int)).astype(int)

    regime_raw = sgn_sum.apply(np.sign)
    enough = (score >= cfg.vote_K).astype(int)
    regime = (regime_raw * enough).fillna(0.0)

    calm_mask = (d4 >= 1.0)
    regime = regime.where(~calm_mask, 0.0)

    regime = regime.where(d3 == 0.0, 0.0)

    n_cd = cfg.N_cooldown.get(tf, 5)
    regime = _apply_cooldown(regime, n_cd)

    out = pd.DataFrame({
        "ts": df["ts"].astype("int64"),
        "d1_adx_trend": v1.astype("float64"),
        "d2_donch_breakout": v2.astype("float64"),
        "d3_changepoint": d3.astype("float64"),
        "d4_calm": d4.astype("float64"),
    })

    out["score"] = (out[["d1_adx_trend", "d2_donch_breakout"]].ne(0).sum(axis=1)).astype("int64")
    out["regime"] = regime.astype("int8")
    out["regime_name"] = out["regime"].map({-1: "bear", 0: "calm", 1: "bull"}).astype("string")

    out = out[["ts", "regime", "regime_name", "score", "d1_adx_trend", "d2_donch_breakout", "d3_changepoint", "d4_calm"]]
    return out.reset_index(drop=True)
