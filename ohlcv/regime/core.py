# ohlcv/regime/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["RegimeConfig", "infer_regime"]


# =============================
# Конфиг (строго по C4)
# =============================
@dataclass(frozen=True)
class RegimeConfig:
    # Ансамбль: 3‑из‑4 + гистерезис/подтверждение + cooldown
    vote_K: int = 3                                 # требуемое число голосов из 4
    N_conf: Dict[str, int] = None                   # «подтверждение» по ТФ: сколько подряд баров нужно
    N_cooldown: Dict[str, int] = None               # заморозка после смены, баров (блок повторной смены)

    # D1: ADX/DI (вход — признаки из C3: f_adx_14, f_pdi_14, f_mdi_14)
    T_adx: float = 22.0                             # порог ADX
    T_di: float = 5.0                               # минимальный разрыв |+DI − −DI|

    # D2: Donchian width / impulse (вход — f_donch_width_pct_{20|55})
    donch_window: int = 20
    quantile_window: int = 500                      # не используется в дефолте (порог фиксированный)
    q_low: float = 0.0015                           # 0.15% от цены (в долях)
    q_high: float = 0.0060                          # 0.60% от цены (в долях)

    # D3: BOCPD‑proxy (онлайн run‑length по лог‑ретёрнам / z‑скорам)
    bocpd_lambda: int = 200                         # характерная длина пробега
    L_min: int = 80                                 # порог «длинного пробега» для тренда
    sigma_L: float = 20.0                           # сглаживание уверенности
    stream: str = "logret"                           # или "close_z"

    # D4: HMM‑proxy по дневной RV (резервная агрегация из входной серии при отсутствии C5)
    hmm_states: int = 2                             # 2 состояния: calm/volatile (прокси)
    N_hmm: int = 180                                # окно по дням для квантилей/сглаживания
    calm_threshold: float = 0.60                    # квантиль RV для метки calm
    q_high_rv: float = 0.90                         # квантиль RV для метки high_rv

    # версия/идентификация
    build_version: str = "C4-Regime-1.2"

    def __post_init__(self):
        object.__setattr__(self, "N_conf", self.N_conf or {"5m": 5, "15m": 5, "1h": 3})
        object.__setattr__(self, "N_cooldown", self.N_cooldown or {"5m": 10, "15m": 8, "1h": 3})


# =============================
# Утилиты форматирования
# =============================

def _iso_from_ms(ms: pd.Series) -> pd.Series:
    ms = pd.to_numeric(ms, errors="coerce").astype("int64")
    base = pd.to_datetime(ms // 1000 * 1000, unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    mspart = (ms % 1000).astype(int).map(lambda x: f"{x:03d}")
    return (base + "." + mspart + "Z").astype("string")


# =============================
# Нормализация входа (из C3)
# =============================

def _ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ожидается датафрейм из C3 со схемой минимум:
      timestamp_ms, start_time_iso, symbol, tf, close, f_adx_14, f_pdi_14, f_mdi_14,
      f_donch_width_pct_20 (и/или _55), f_logret1
    Допускается резервный вход OHLCV: ts/o/h/l/c/v — тогда вычислим минимальные прокси.
    Выход: нормализованный df с timestamp_ms:int64 и доступными f_* колонками.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ожидается DataFrame")
    out = df.copy()

    # приведение имён
    out.columns = [str(c) for c in out.columns]
    lower = {c: c.lower() for c in out.columns}
    out = out.rename(columns=lower)

    # timestamp_ms / ts
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

    # базовые поля
    if "start_time_iso" not in out.columns:
        out["start_time_iso"] = _iso_from_ms(out["timestamp_ms"]).astype("string")
    if "symbol" not in out.columns:
        out["symbol"] = "?"
    if "tf" not in out.columns:
        out["tf"] = "?"

    # если нет close, пытаемся собрать из c/close
    if "close" not in out.columns:
        if "c" in out.columns:
            out["close"] = pd.to_numeric(out["c"], errors="coerce").astype("float64")
        else:
            raise ValueError("отсутствует close/c")

    # сортировка/дедуп
    out = out.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last").reset_index(drop=True)

    # типы ядра
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce").astype("int64")
    out["close"] = pd.to_numeric(out["close"], errors="coerce").astype("float64")

    return out


# =============================
# Вспомогательные
# =============================

def _rolling_quantile(x: pd.Series, n: int, q: float) -> pd.Series:
    return x.rolling(n, min_periods=max(5, n // 5)).quantile(q)


def _zscore(x: pd.Series, n: int) -> pd.Series:
    mu = x.rolling(n, min_periods=max(5, n // 5)).mean()
    sd = x.rolling(n, min_periods=max(5, n // 5)).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _realized_vol(logret: pd.Series, n: int) -> pd.Series:
    return logret.rolling(n, min_periods=max(5, n // 5)).std(ddof=1)


def _stable_sign(s: pd.Series, n_conf: int) -> pd.Series:
    x = s.fillna(0.0).astype(float)
    sign = pd.Series(np.sign(x.values), index=x.index).replace(0.0, np.nan).ffill()
    grp = (sign != sign.shift()).cumsum()
    run = grp.groupby(grp).cumcount() + 1
    ok = (run >= n_conf).astype(int)
    return (pd.Series(np.sign(sign.values), index=x.index) * ok).astype(float)


def _apply_cooldown_lock(sig: pd.Series, n_cd: int) -> pd.Series:
    """Блок повторной смены: удерживает предыдущий режим n_cd баров после переключения.
    Вход: sig ∈ {−1, 1}, без нулей (предварительно ffill).
    """
    s = sig.ffill().fillna(0.0).astype(float).values
    out = np.zeros_like(s, dtype=float)
    prev = 0.0
    cd = 0
    for i, v in enumerate(s):
        proposed = v
        if i == 0:
            out[i] = proposed
            prev = out[i]
            continue
        if proposed != prev:
            if cd > 0:
                out[i] = prev
                cd -= 1
            else:
                out[i] = proposed
                prev = out[i]
                cd = n_cd
        else:
            out[i] = prev
            if cd > 0:
                cd -= 1
    return pd.Series(out, index=sig.index)


# =============================
# Детекторы (по формулировкам C4; D3/D4 — proxy без внешних моделей)
# =============================

def _D1_adx_trend(features: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.Series, pd.Series, pd.Series]:
    adx = pd.to_numeric(features.get("f_adx_14"), errors="coerce")
    pdi = pd.to_numeric(features.get("f_pdi_14"), errors="coerce")
    mdi = pd.to_numeric(features.get("f_mdi_14"), errors="coerce")

    di_gap = (pdi - mdi).abs()
    trend = ((adx >= cfg.T_adx) & (di_gap >= cfg.T_di)).astype(int)
    vote_trend = trend.astype(int)
    vote_flat = (1 - vote_trend).astype(int)

    T_max = cfg.T_adx + 20.0
    s_adx = ((adx - cfg.T_adx) / (T_max - cfg.T_adx)).clip(lower=0.0, upper=1.0)

    return vote_trend.astype(float), vote_flat.astype(float), s_adx.astype(float)


def _D2_donch(features: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.Series, pd.Series, pd.Series]:
    w = None
    for name in (f"f_donch_width_pct_{cfg.donch_window}", "f_donch_width_pct_55"):
        if name in features.columns:
            w = pd.to_numeric(features[name], errors="coerce")
            break
    if w is None:
        raise ValueError("нет f_donch_width_pct_{20|55}")

    ql = pd.Series(cfg.q_low, index=w.index, dtype="float64")
    qh = pd.Series(cfg.q_high, index=w.index, dtype="float64")

    dw = w.diff().fillna(0.0)
    trend = ((w >= qh) & (dw >= 0.0)).astype(int)
    flat = (w <= ql).astype(int)

    s_donch = ((w - ql) / (qh - ql)).clip(lower=0.0, upper=1.0)

    return trend.astype(float), flat.astype(float), s_donch.astype(float)


def _D3_bocpd_proxy(features: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    if cfg.stream == "close_z":
        zcols = [c for c in features.columns if c.startswith("f_close_z_")]
        if not zcols:
            raise ValueError("нет f_close_z_* для BOCPD‑proxy")
        x = features[zcols[0]].astype(float)
    else:
        x = features.get("f_logret1")
        if x is None:
            with np.errstate(divide="ignore"):
                x = np.log(pd.to_numeric(features["close"], errors="coerce")).diff()
        x = x.astype(float)

    q = x.abs().rolling(cfg.bocpd_lambda, min_periods=max(10, cfg.bocpd_lambda // 5)).quantile(0.80)
    below = (x.abs() < q).astype(int)
    rl = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        rl[i] = 0.0 if i == 0 else (rl[i - 1] + 1.0) * below.iloc[i]
    run_length = pd.Series(rl, index=x.index)

    p_trend = 1.0 / (1.0 + np.exp(-(run_length - cfg.L_min) / cfg.sigma_L))
    p_flat = 1.0 - p_trend

    vote_trend = (run_length >= cfg.L_min).astype(int)
    vote_flat = (1 - vote_trend).astype(int)

    chgpt = ((run_length.shift(1) >= cfg.L_min) & (run_length == 0)).astype(int)

    return vote_trend.astype(float), vote_flat.astype(float), p_trend.astype(float), chgpt.astype(int)


def _daily_rv_from_logret(ts_ms: pd.Series, logret: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    ts_ms = pd.to_numeric(ts_ms, errors="coerce").astype("int64")
    lr = pd.to_numeric(logret, errors="coerce").astype("float64")
    day = pd.to_datetime(ts_ms, unit="ms", utc=True).floor("D")
    rv_day = lr.groupby(day).std(ddof=1)
    rv_day_roll = rv_day.rolling(window=n, min_periods=max(10, n // 6)).mean()
    rv_day_aligned = rv_day.reindex(day).reset_index(drop=True)
    rv_roll_aligned = rv_day_roll.reindex(day).reset_index(drop=True)
    idx = lr.index
    return pd.Series(rv_day_aligned.values, index=idx), pd.Series(rv_roll_aligned.values, index=idx)


def _D4_hmm_rv_proxy(features: pd.DataFrame, cfg: RegimeConfig) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    x = features.get("f_logret1")
    if x is None:
        with np.errstate(divide="ignore"):
            x = np.log(pd.to_numeric(features["close"], errors="coerce")).diff()
    x = x.astype(float)

    rv_day, rv_day_roll = _daily_rv_from_logret(features["timestamp_ms"], x, cfg.N_hmm)

    thr_calm = rv_day.rolling(cfg.N_hmm, min_periods=max(10, cfg.N_hmm // 6)).quantile(cfg.calm_threshold)
    p_calm = ((thr_calm - rv_day) / thr_calm.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)

    thr_high = rv_day.rolling(cfg.N_hmm, min_periods=max(10, cfg.N_hmm // 6)).quantile(cfg.q_high_rv)
    high_rv = (rv_day > thr_high).astype(int)

    vote_trend = (p_calm > 0.5).astype(int)
    vote_flat = (1 - vote_trend).astype(int)

    return vote_trend.astype(float), vote_flat.astype(float), p_calm.astype(float), high_rv.astype(int)


# =============================
# Инференс
# =============================

def infer_regime(df_in: pd.DataFrame, tf: str, cfg: Optional[RegimeConfig] = None) -> pd.DataFrame:
    cfg = cfg or RegimeConfig()
    df = _ensure_features(df_in)

    # D1..D4
    d1_tr, d1_fl, s_adx = _D1_adx_trend(df, cfg)
    d2_tr, d2_fl, s_donch = _D2_donch(df, cfg)
    d3_tr, d3_fl, p_bocpd_trend, chgpt = _D3_bocpd_proxy(df, cfg)
    d4_tr, d4_fl, s_hmm_rv, high_rv = _D4_hmm_rv_proxy(df, cfg)

    # подтверждение и голоса по ТФ
    n_conf = int(cfg.N_conf.get(tf, 5))
    v1 = _stable_sign(d1_tr - d1_fl, n_conf)
    v2 = _stable_sign(d2_tr - d2_fl, n_conf)
    v3 = _stable_sign(d3_tr - d3_fl, n_conf)
    v4 = _stable_sign(d4_tr - d4_fl, n_conf)

    votes_trend = (v1.gt(0).astype(int) + v2.gt(0).astype(int) + v3.gt(0).astype(int) + v4.gt(0).astype(int))
    votes_flat = (v1.lt(0).astype(int) + v2.lt(0).astype(int) + v3.lt(0).astype(int) + v4.lt(0).astype(int))

    # 3‑из‑4 с удержанием предыдущего режима при недостатке голосов
    regime_signal = pd.Series(np.where(votes_trend >= cfg.vote_K, 1,
                                       np.where(votes_flat >= cfg.vote_K, -1, np.nan)), index=df.index)
    regime_ffill = regime_signal.ffill().fillna(0.0)

    # cooldown: блок повторной смены, не вводит нейтрали
    n_cd = int(cfg.N_cooldown.get(tf, 5))
    regime_locked = _apply_cooldown_lock(regime_ffill, n_cd)

    # уверенность ансамбля
    regime_confidence = (np.maximum(votes_trend, votes_flat) / 4.0).astype(float)

    # финальная метка только {trend, flat}
    regime_str = np.where(regime_locked > 0, "trend", "flat")

    out = pd.DataFrame({
        "timestamp_ms": df["timestamp_ms"].astype("int64"),
        "start_time_iso": df["start_time_iso"].astype("string"),
        "symbol": df["symbol"].astype("string"),
        "tf": pd.Series([tf] * len(df), index=df.index, dtype="string"),
        "regime": pd.Series(regime_str, index=df.index, dtype="string"),
        "high_rv": high_rv.astype("int8"),
        "regime_confidence": regime_confidence.astype("float64"),
        "votes_trend": votes_trend.astype("int16"),
        "votes_flat": votes_flat.astype("int16"),
        "det_used": pd.Series(["1111"] * len(df), index=df.index, dtype="string"),
        "chgpt": chgpt.astype("int8"),
        "p_bocpd_trend": p_bocpd_trend.astype("float64"),
        "p_bocpd_flat": (1.0 - p_bocpd_trend).astype("float64"),
        "s_adx": s_adx.astype("float64"),
        "s_donch": s_donch.astype("float64"),
        "s_hmm_rv": s_hmm_rv.astype("float64"),
        "hysteresis_state": regime_locked.astype("float64"),
        "build_version": pd.Series([cfg.build_version] * len(df), index=df.index, dtype="string"),
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
