# c4_regime/detectors.py
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .bocpd import run_length_bocpd
from .hmm_rv import fit_predict_hmm_rv

__all__ = [
    "d1_adx_trend",
    "d2_donch_width",
    "d3_bocpd_proxy",
    "d4_hmm_rv_or_quantile",
]

# ---------------- helpers ----------------

_NAME_ALIASES = {
    "f_adx14": "f_adx_14",
    "f_pdi14": "f_pdi_14",
    "f_mdi14": "f_mdi_14",
}


def _get_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    alt = _NAME_ALIASES.get(name)
    if alt and alt in df.columns:
        return pd.to_numeric(df[alt], errors="coerce")
    return pd.Series(index=df.index, dtype=float)


def _stable_bool(x: pd.Series) -> pd.Series:
    return x.fillna(0).astype(int)


# ---------------- D1: ADX/DI ----------------

def d1_adx_trend(features: pd.DataFrame, *, T_adx: float, T_di: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    adx = _get_col(features, "f_adx_14")
    if adx.empty:
        adx = _get_col(features, "f_adx14")
    pdi = _get_col(features, "f_pdi_14")
    mdi = _get_col(features, "f_mdi_14")

    di_gap = (pdi - mdi).abs()
    trend = (adx >= float(T_adx)) & (di_gap >= float(T_di))
    trend_vote = _stable_bool(trend)
    flat_vote = 1 - trend_vote

    T_max = float(T_adx) + 20.0
    score = ((adx - float(T_adx)) / (T_max - float(T_adx))).clip(lower=0.0, upper=1.0)
    return trend_vote.astype(float), flat_vote.astype(float), score.astype(float)


# ---------------- D2: Donchian width ----------------

def d2_donch_width(
    features: pd.DataFrame,
    *,
    low_thr: float,  # q_low в долях цены
    high_thr: float,  # q_high в долях цены
    prefer_window: int = 20,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    width = None
    for name in (f"f_donch_width_pct_{prefer_window}", "f_donch_width_pct_55"):
        if name in features.columns:
            width = pd.to_numeric(features[name], errors="coerce")
            break
    if width is None:
        raise KeyError("требуется f_donch_width_pct_{20|55}")

    ql = pd.Series(float(low_thr), index=width.index, dtype="float64")
    qh = pd.Series(float(high_thr), index=width.index, dtype="float64")

    dw = width.diff().fillna(0.0)
    trend = (width >= qh) & (dw >= 0.0)
    flat = width <= ql
    score = ((width - ql) / (qh - ql)).clip(lower=0.0, upper=1.0)

    return _stable_bool(trend).astype(float), _stable_bool(flat).astype(float), score.astype(float)


# ---------------- D3: BOCPD proxy ----------------

def _logret_from_close(features: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(features.get("close"), errors="coerce")
    with np.errstate(divide="ignore"):
        lr = np.log(c).diff()
    return lr


def d3_bocpd_proxy(
    features: pd.DataFrame,
    *,
    lam: float,
    L_min: float,
    sigma_L: float,
    stream: str = "logret",
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if stream == "close_z":
        zcols = [c for c in features.columns if c.startswith("f_close_z_")]
        if not zcols:
            raise KeyError("нет f_close_z_* для BOCPD proxy")
        x = pd.to_numeric(features[zcols[0]], errors="coerce").to_numpy()
    else:
        lr = pd.to_numeric(features.get("f_logret1"), errors="coerce")
        if lr.isna().all():
            lr = _logret_from_close(features)
        x = lr.to_numpy()

    run, p_trend, trend_flag, flat_flag = run_length_bocpd(
        x, lam=float(lam), L_min=float(L_min), sigma_L=float(sigma_L)
    )

    chgpt = np.zeros_like(trend_flag, dtype=np.int8)
    if len(run) > 1:
        chgpt[1:] = ((run[:-1] >= float(L_min)) & (run[1:] == 0)).astype(np.int8)

    return (
        pd.Series(trend_flag, index=features.index, dtype=float),
        pd.Series(flat_flag, index=features.index, dtype=float),
        pd.Series(p_trend, index=features.index, dtype=float),
        pd.Series(chgpt, index=features.index, dtype=np.int8),
    )


# ---------------- D4: HMM RV or quantile fallback ----------------

def _daily_rv_from_logret(ts_ms: pd.Series, lr: pd.Series) -> pd.Series:
    ts_ms = pd.to_numeric(ts_ms, errors="coerce").astype("int64")
    lr = pd.to_numeric(lr, errors="coerce").astype(float)
    day = pd.to_datetime(ts_ms, unit="ms", utc=True).floor("D")
    return lr.groupby(day).std(ddof=1)


def d4_hmm_rv_or_quantile(
    features: pd.DataFrame,
    *,
    N_hmm: int = 180,
    calm_threshold: float = 0.60,
    q_high_rv: float = 0.90,
    n_states: int = 2,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rv_daily = None
    for name in ("rv_daily", "ctx_rv_daily"):
        if name in features.columns:
            rv_daily = pd.to_numeric(features[name], errors="coerce").to_numpy()
            break

    if rv_daily is None:
        lr = pd.to_numeric(features.get("f_logret1"), errors="coerce")
        if lr.isna().all():
            lr = _logret_from_close(features)
        ts = pd.to_numeric(features.get("timestamp_ms"), errors="coerce").astype("int64")
        if ts.isna().all():
            rv_daily = None
        else:
            rv_day = _daily_rv_from_logret(ts, lr)
            left = rv_day.reset_index()
            left.columns = [rv_day.index.name or "ts", rv_day.name or 0]
            left = left.rename(columns={left.columns[0]: "ts", left.columns[1]: "rv"})
            left = left.sort_values("ts")
            right = pd.DataFrame({"ts": pd.to_datetime(ts, unit="ms", utc=True).floor("D")}).sort_values("ts")
            rv_daily = (
                pd.merge_asof(right, left, on="ts", direction="backward")["rv"].fillna(method="ffill").fillna(0.0).to_numpy()
            )

    p_calm = None
    try:
        if rv_daily is not None:
            p_calm = fit_predict_hmm_rv(np.asarray(rv_daily, dtype=float), n_states=int(n_states))
    except Exception:
        p_calm = None

    if p_calm is None:
        x = np.asarray(rv_daily if rv_daily is not None else np.asarray(features.get("f_rv_60", pd.Series(0.0))), dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        s = pd.Series(x)
        roll = s.rolling(N_hmm, min_periods=max(10, N_hmm // 6))
        q_high = roll.quantile(q_high_rv).bfill().fillna(s.quantile(q_high_rv))
        q_calm = roll.quantile(calm_threshold).bfill().fillna(s.quantile(calm_threshold))
        high_flag = (s > q_high).astype(np.int8).to_numpy()
        p_calm = ((q_calm - s) / q_calm.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0).fillna(0.0).to_numpy()
    else:
        s = pd.Series(p_calm)
        thr = s.rolling(N_hmm, min_periods=max(10, N_hmm // 6)).quantile(1.0 - q_high_rv).bfill().fillna(s.quantile(1.0 - q_high_rv))
        high_flag = (s < thr).astype(np.int8).to_numpy()

    trend_vote = (p_calm > 0.5).astype(np.int8)
    flat_vote = (1 - trend_vote).astype(np.int8)

    return (
        pd.Series(trend_vote, index=features.index, dtype=float),
        pd.Series(flat_vote, index=features.index, dtype=float),
        pd.Series(p_calm, index=features.index, dtype=float),
        pd.Series(high_flag, index=features.index, dtype=np.int8),
    )
