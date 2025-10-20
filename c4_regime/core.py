# c4_regime/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .detectors import (
    d1_adx_trend,
    d2_donch_width,
    d3_bocpd_proxy,
    d4_hmm_rv_or_quantile,
)

__all__ = ["RegimeConfig", "infer_regime"]


# =============================
# Конфиг C4
# =============================
@dataclass(frozen=True)
class RegimeConfig:
    # Ансамбль
    vote_K: int = 3
    N_conf: Dict[str, int] = None                 # подтверждение: длина пробега одного знака
    N_cooldown: Dict[str, int] = None             # заморозка после смены, баров

    # D1: ADX/DI
    T_adx: float = 22.0
    T_di: float = 5.0

    # D2: Donchian width
    donch_window: int = 20
    q_low: float = 0.0015
    q_high: float = 0.0060

    # D3: BOCPD proxy
    bocpd_lambda: int = 200
    L_min: int = 80
    sigma_L: float = 20.0
    stream: str = "logret"  # или "close_z"

    # D4: HMM‑proxy по дневной RV (или квантильный фолбэк)
    hmm_states: int = 2
    N_hmm: int = 180
    calm_threshold: float = 0.60
    q_high_rv: float = 0.90

    # Версия
    build_version: str = "C4-Regime-1.2"

    def __post_init__(self):
        object.__setattr__(self, "N_conf", self.N_conf or {"5m": 5, "15m": 5, "1h": 3})
        object.__setattr__(self, "N_cooldown", self.N_cooldown or {"5m": 10, "15m": 8, "1h": 3})


# =============================
# Нормализация входа (ожидаем C3)
# =============================

def _iso_from_ms(ms: pd.Series) -> pd.Series:
    ms = pd.to_numeric(ms, errors="coerce").astype("int64")
    base = pd.to_datetime(ms // 1000 * 1000, unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    frac = (ms % 1000).astype(int).map(lambda x: f"{x:03d}")
    return (base + "." + frac + "Z").astype("string")


def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ожидается DataFrame")
    out = df.copy()
    out.columns = [str(c) for c in out.columns]

    # timestamp_ms
    if "timestamp_ms" not in out.columns:
        if "ts" in out.columns:
            out["timestamp_ms"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
        elif isinstance(out.index, pd.DatetimeIndex):
            idx = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
            out["timestamp_ms"] = (idx.asi8 // 1_000_000).astype("int64")
        elif "start_time_iso" in out.columns:
            tsi = pd.to_datetime(out["start_time_iso"], utc=True, errors="coerce")
            out["timestamp_ms"] = (tsi.view("int64") // 1_000_000).astype("int64")
        else:
            raise ValueError("нет timestamp_ms/ts/start_time_iso и нет DatetimeIndex")

    if "start_time_iso" not in out.columns:
        out["start_time_iso"] = _iso_from_ms(out["timestamp_ms"])  # RFC3339 .sssZ

    if "symbol" not in out.columns:
        out["symbol"] = "?"
    if "tf" not in out.columns:
        out["tf"] = "?"

    # close (для резервных потоков)
    if "close" not in out.columns and "c" in out.columns:
        out["close"] = pd.to_numeric(out["c"], errors="coerce").astype("float64")

    out = out.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce").astype("int64")
    if "close" in out.columns:
        out["close"] = pd.to_numeric(out["close"], errors="coerce").astype("float64")
    return out


# =============================
# Подтверждение и заморозка
# =============================

def _stable_sign(s: pd.Series, n_conf: int) -> pd.Series:
    x = s.fillna(0.0).astype(float)
    sign = pd.Series(np.sign(x.values), index=x.index)
    # последовательности одинакового знака
    grp = (sign != sign.shift()).cumsum()
    run = grp.groupby(grp).cumcount() + 1
    ok = (run >= int(n_conf)).astype(int)
    return (sign * ok).astype(float)


def _apply_cooldown_lock(sig: pd.Series, n_cd: int) -> pd.Series:
    s = sig.ffill().fillna(0.0).astype(float).values
    out = np.zeros_like(s, dtype=float)
    prev = 0.0
    cd = 0
    for i, v in enumerate(s):
        proposed = v
        if i == 0:
            out[i] = proposed
            prev = proposed
            continue
        if proposed != prev:
            if cd > 0:
                out[i] = prev
                cd -= 1
            else:
                out[i] = proposed
                prev = proposed
                cd = int(n_cd)
        else:
            out[i] = prev
            if cd > 0:
                cd -= 1
    return pd.Series(out, index=sig.index)


# =============================
# Инференс
# =============================

def infer_regime(df_in: pd.DataFrame, tf: str, cfg: Optional[RegimeConfig] = None) -> pd.DataFrame:
    cfg = cfg or RegimeConfig()
    feats = _ensure_features(df_in)

    # Детекторы D1..D4
    d1_tr, d1_fl, s_adx = d1_adx_trend(feats, T_adx=cfg.T_adx, T_di=cfg.T_di)
    d2_tr, d2_fl, s_don = d2_donch_width(
        feats, low_thr=cfg.q_low, high_thr=cfg.q_high, prefer_window=cfg.donch_window
    )
    d3_tr, d3_fl, p_bocpd_trend, chgpt = d3_bocpd_proxy(
        feats, lam=cfg.bocpd_lambda, L_min=cfg.L_min, sigma_L=cfg.sigma_L, stream=cfg.stream
    )
    d4_tr, d4_fl, s_hmm_rv, high_rv = d4_hmm_rv_or_quantile(
        feats, N_hmm=cfg.N_hmm, calm_threshold=cfg.calm_threshold, q_high_rv=cfg.q_high_rv, n_states=cfg.hmm_states
    )

    # Подтверждение по ТФ
    n_conf = int(cfg.N_conf.get(tf, 5))
    v1 = _stable_sign(d1_tr - d1_fl, n_conf)
    v2 = _stable_sign(d2_tr - d2_fl, n_conf)
    v3 = _stable_sign(d3_tr - d3_fl, n_conf)
    v4 = _stable_sign(d4_tr - d4_fl, n_conf)

    votes_trend = (v1.gt(0).astype(int) + v2.gt(0).astype(int) + v3.gt(0).astype(int) + v4.gt(0).astype(int))
    votes_flat = (v1.lt(0).astype(int) + v2.lt(0).astype(int) + v3.lt(0).astype(int) + v4.lt(0).astype(int))

    # Правило 3‑из‑4
    raw = pd.Series(np.where(votes_trend >= cfg.vote_K, 1, np.where(votes_flat >= cfg.vote_K, -1, np.nan)), index=feats.index)
    regime_ffill = raw.ffill().fillna(0.0)

    # Заморозка
    n_cd = int(cfg.N_cooldown.get(tf, 5))
    regime_locked = _apply_cooldown_lock(regime_ffill, n_cd)

    # Уверенность ансамбля
    regime_conf = (np.maximum(votes_trend, votes_flat) / 4.0).astype(float)

    # Метка
    regime_str = np.where(regime_locked > 0, "trend", "flat")

    out = pd.DataFrame({
        "timestamp_ms": feats["timestamp_ms"].astype("int64"),
        "start_time_iso": feats["start_time_iso"].astype("string"),
        "symbol": feats["symbol"].astype("string"),
        "tf": pd.Series([tf] * len(feats), index=feats.index, dtype="string"),  # tf из аргумента
        "regime": pd.Series(regime_str, index=feats.index, dtype="string"),
        "high_rv": pd.to_numeric(high_rv, errors="coerce").astype("int8"),
        "regime_confidence": regime_conf.astype("float64"),
        "votes_trend": votes_trend.astype("int16"),
        "votes_flat": votes_flat.astype("int16"),
        "det_used": pd.Series(["1111"] * len(feats), index=feats.index, dtype="string"),
        "chgpt": pd.to_numeric(chgpt, errors="coerce").astype("int8"),
        "p_bocpd_trend": pd.to_numeric(p_bocpd_trend, errors="coerce").astype("float64"),
        "p_bocpd_flat": (1.0 - pd.to_numeric(p_bocpd_trend, errors="coerce")).astype("float64"),
        "s_adx": pd.to_numeric(s_adx, errors="coerce").astype("float64"),
        "s_donch": pd.to_numeric(s_don, errors="coerce").astype("float64"),
        "s_hmm_rv": pd.to_numeric(s_hmm_rv, errors="coerce").astype("float64"),
        "hysteresis_state": regime_locked.astype("float64"),
        "build_version": pd.Series([cfg.build_version] * len(feats), index=feats.index, dtype="string"),
    })

    cols = [
        "timestamp_ms", "start_time_iso", "symbol", "tf",
        "regime", "high_rv", "regime_confidence",
        "votes_trend", "votes_flat", "det_used", "chgpt",
        "p_bocpd_trend", "p_bocpd_flat",
        "s_adx", "s_donch", "s_hmm_rv",
        "hysteresis_state", "build_version",
    ]
    out = out[cols]
    return out.reset_index(drop=True)
