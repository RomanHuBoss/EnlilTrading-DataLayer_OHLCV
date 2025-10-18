from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .bocpd import run_length_bocpd, sigmoid
from .detectors import d1_adx, d2_donch, d4_hmm_rv
from .errors import ErrorCodes as E

# Конфиг по умолчанию (совместим с постановкой)
DEFAULT_CFG = {
    "ensemble": {
        "vote_K": 3,
        "N_conf": {"5m": 5, "15m": 5, "1h": 3},
        "N_cooldown": {"5m": 10, "15m": 8, "1h": 3},
    },
    "D1_adx": {"T_adx": 22.0, "T_di": 5.0},
    "D2_donch": {"window": 20, "quantile_window": 500, "q_low": 0.15, "q_high": 0.60},
    "D3_bocpd": {"lambda": 200.0, "sigma_L": 20.0, "stream": "logret", "L_min": 200.0},
    "D4_hmm_rv": {"n_states": 2, "N_hmm": 180, "calm_threshold": 0.6},
    "runtime": {"allow_fallback_daily": True},
}


def _get_tf(df: pd.DataFrame) -> str:
    if "tf" in df.columns:
        return str(df["tf"].iloc[0])
    return "5m"


def _hash_build(cfg: dict) -> str:
    s = repr(sorted(_flatten(cfg).items())).encode()
    return hashlib.sha1(s).hexdigest()[:12]


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: list[tuple[str, object]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


REQUIRED_BASE = {"timestamp_ms", "start_time_iso", "symbol", "tf"}
REQUIRED_FOR_STREAM = {"f_logret1", "f_close_z_60"}


def _choose_stream(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "logret" and "f_logret1" in df.columns:
        return df["f_logret1"].astype(float).to_numpy()
    if name == "close_z" and "f_close_z_60" in df.columns:
        return df["f_close_z_60"].astype(float).to_numpy()
    # Фоллбек: лог-доходности из close
    if "close" in df.columns:
        c = df["close"].astype(float)
        x = np.log(c / c.shift(1)).fillna(0.0).to_numpy()
        return x
    raise KeyError("stream features not found")


@dataclass
class HysteresisState:
    regime: int  # 1=trend, 0=flat
    streak: int  # длина текущей подтверждаемой серии
    cooldown: int  # оставшиеся бары до разрешения новой смены


def _apply_hysteresis(
    regime_raw: np.ndarray,
    tf: str,
    N_conf_map: dict,
    N_cooldown_map: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (regime_final, chgpt, hysteresis_state_int).
    hysteresis_state_int кодирует streak (>0) и cooldown (<0) для отладки.
    """
    n = len(regime_raw)
    out = np.empty(n, dtype=np.int8)
    chg = np.zeros(n, dtype=np.int8)
    state_serial = np.zeros(n, dtype=np.int32)
    N_conf = int(N_conf_map.get(tf, 5))
    N_cooldown = int(N_cooldown_map.get(tf, 5))

    st = HysteresisState(regime=int(regime_raw[0]), streak=1, cooldown=0)
    out[0] = st.regime
    state_serial[0] = st.streak
    for i in range(1, n):
        if st.cooldown > 0:
            st.cooldown -= 1
        if int(regime_raw[i]) == st.regime:
            st.streak = min(st.streak + 1, 10_000)
            out[i] = st.regime
            state_serial[i] = st.streak
            continue
        # Кандидат на смену
        # Нужно N_conf подряд
        needed = True
        if i + 1 >= N_conf:
            if np.all(regime_raw[i - N_conf + 1 : i + 1] == regime_raw[i]):
                needed = True
            else:
                needed = False
        else:
            needed = False
        if needed and st.cooldown == 0:
            prev = st.regime
            st.regime = int(regime_raw[i])
            st.streak = 1
            st.cooldown = N_cooldown
            out[i] = st.regime
            chg[i] = 1 if st.regime != prev else 0
            state_serial[i] = -st.cooldown
        else:
            out[i] = st.regime
            state_serial[i] = st.streak if st.cooldown == 0 else -st.cooldown
    return out, chg, state_serial


def infer_regime(
    features_df: pd.DataFrame,
    context: Optional[pd.DataFrame | Dict] = None,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """Главная точка входа C4. Возвращает датафрейм с колонками режима.
    Обязательные входы/окна описаны в постановке. Работает детерминированно.
    """
    if cfg is None:
        cfg = DEFAULT_CFG
    # Базовые проверки
    missing = [c for c in REQUIRED_BASE if c not in features_df.columns]
    if missing:
        raise ValueError(f"{E.E_MISSING_FEATURE}: base {missing}")

    tf = _get_tf(features_df)

    # --- Детектор D1 (ADX) ---
    try:
        d1_tr, d1_fl, s_adx = d1_adx(features_df, **cfg["D1_adx"])  # type: ignore[arg-type]
        d1_used = True
    except Exception:
        n = len(features_df)
        d1_tr = np.zeros(n, dtype=np.int8)
        d1_fl = np.zeros(n, dtype=np.int8)
        s_adx = np.zeros(n, dtype=float)
        d1_used = False

    # --- Детектор D2 (Donchian width) ---
    try:
        d2_tr, d2_fl, s_donch = d2_donch(features_df, **cfg["D2_donch"])  # type: ignore[arg-type]
        d2_used = True
    except Exception:
        n = len(features_df)
        d2_tr = np.zeros(n, dtype=np.int8)
        d2_fl = np.zeros(n, dtype=np.int8)
        s_donch = np.zeros(n, dtype=float)
        d2_used = False

    # --- Детектор D3 (BOCPD surrogate) ---
    stream_name = str(cfg["D3_bocpd"].get("stream", "logret"))
    x = _choose_stream(features_df, stream_name)
    lam = float(cfg["D3_bocpd"].get("lambda", 200.0))
    sigma_L = float(cfg["D3_bocpd"].get("sigma_L", 20.0))
    L_min = float(cfg["D3_bocpd"].get("L_min", lam))
    run, p_tr_bocpd, d3_tr, d3_fl = run_length_bocpd(x, lam=lam, L_min=L_min, sigma_L=sigma_L)

    # --- Детектор D4 (HMM / RV daily) ---
    allow_fb = bool(cfg["runtime"].get("allow_fallback_daily", True))
    calm_thr = float(cfg["D4_hmm_rv"].get("calm_threshold", 0.6))
    high_rv, s_hmm, rv_daily = d4_hmm_rv(
        features_df, context, int(cfg["D4_hmm_rv"].get("n_states", 2)), calm_thr, allow_fb
    )

    # --- Голосование 3-из-4 ---
    vote_tr = (
        d1_tr.astype(int) + d2_tr.astype(int) + d3_tr.astype(int) + (s_hmm >= calm_thr).astype(int)
    )
    vote_fl = (
        d1_fl.astype(int) + d2_fl.astype(int) + d3_fl.astype(int) + (s_hmm < calm_thr).astype(int)
    )

    K = int(cfg["ensemble"].get("vote_K", 3))
    regime_star = np.where(vote_tr >= K, 1, np.where(vote_fl >= K, 0, 0)).astype(np.int8)

    # --- Гистерезис и cooldown ---
    N_conf_map = cfg["ensemble"].get("N_conf", {"5m": 5, "15m": 5, "1h": 3})
    N_cooldown_map = cfg["ensemble"].get("N_cooldown", {"5m": 10, "15m": 8, "1h": 3})
    regime_final, chg, hyst_state = _apply_hysteresis(regime_star, tf, N_conf_map, N_cooldown_map)

    # --- Уверенность ---
    conf = (np.abs(vote_tr - vote_fl) / 4.0).astype(float)
    conf = conf * (1.0 - 0.5 * high_rv.astype(float))  # затухание при high_rv

    # --- Выходной фрейм ---
    out = pd.DataFrame(index=features_df.index)
    out["regime"] = np.where(regime_final == 1, "trend", "flat")
    out["high_rv"] = high_rv.astype(np.int8)
    out["regime_confidence"] = conf
    out["votes_trend"] = vote_tr
    out["votes_flat"] = vote_fl
    out["det_used"] = (("D1" if d1_used else "") + ("|D2" if d2_used else "") + "|D3|D4").lstrip(
        "|"
    )
    out["chgpt"] = chg.astype(np.int8)
    out["p_bocpd_trend"] = p_tr_bocpd
    out["p_bocpd_flat"] = 1.0 - p_tr_bocpd
    out["s_adx"] = s_adx
    out["s_donch"] = s_donch
    out["s_hmm_rv"] = s_hmm
    out["hysteresis_state"] = hyst_state

    # Служебные поля
    for col in ("symbol", "tf"):
        if col in features_df.columns:
            out[col] = features_df[col].iloc[0]
        else:
            out[col] = "UNKNOWN"

    # build_version — sha1 от конфига
    out["build_version"] = _hash_build(cfg)

    return out
