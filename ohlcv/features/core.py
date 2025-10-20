# ohlcv/features/core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "FeatureConfig",
    "DEFAULTS",
    "ensure_input",
    "compute_features",
    "normalize_and_validate",
]

# =============================
# Конфиг (C3)
# =============================

@dataclass(frozen=True)
class FeatureConfig:
    # окна
    rv: tuple[int, int] = (20, 60)            # std(logret, n), ddof=1
    ema: tuple[int, int] = (20, 50)           # EMA по close
    mom: tuple[int, int, int] = (20, 50, 100) # C/C_{t-n} - 1
    rsi: int = 14                              # RSI Уайлдера
    adx: int = 14                              # ADX/+DI/-DI Уайлдера
    donch: tuple[int, int] = (20, 55)          # Donchian
    z: tuple[int, int] = (20, 60)              # Z-score окна
    vwap_roll: int = 96                        # роллинг-VWAP
    updownvol: int = 20                        # окно для up/down volume
    kama: int = 10                             # KAMA (опц.)

    # версия/идентификация
    build_version: str = "C3-Core-1.4"

DEFAULTS = FeatureConfig()

# =============================
# Нормализация входа
# =============================

def ensure_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вход допускается в двух формах:
      — marketdata: timestamp_ms,start_time_iso,open,high,low,close,volume[,turnover]
      — нормализованная: ts,o,h,l,c,v[,t]
    Выход: DataFrame со столбцами ts:int64, o,h,l,c,v:float64, t:float64? и сортировкой по ts.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ожидается DataFrame")
    out = df.copy()

    # Приведение имён
    lower = {c: str(c).lower() for c in out.columns}
    out = out.rename(columns=lower)
    out = out.rename(columns={
        "timestamp_ms": "ts",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
        "turnover": "t",
    })

    # ts из DatetimeIndex/start_time_iso при необходимости
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            idx = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")
            out["ts"] = (idx.asi8 // 1_000_000).astype("int64")
        elif "start_time_iso" in out.columns:
            tsi = pd.to_datetime(out["start_time_iso"], utc=True, errors="coerce")
            out["ts"] = (tsi.view("int64") // 1_000_000).astype("int64")
        else:
            raise ValueError("нет колонки 'ts'/'timestamp_ms'/'start_time_iso' и нет DatetimeIndex")

    # Типы
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ("o","h","l","c","v","t"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # Стабилизация строк
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    need = {"ts","o","h","l","c","v"}
    miss = need - set(out.columns)
    if miss:
        raise ValueError(f"отсутствуют обязательные колонки: {sorted(miss)}")

    return out[["ts","o","h","l","c","v"] + (["t"] if "t" in out.columns else [])]


# =============================
# Вспомогательные (формулы)
# =============================

_EPS = 1e-12


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def _ema_wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False, min_periods=n).mean()


def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr


def _rsi_wilder(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = _ema_wilder(up, n)
    roll_dn = _ema_wilder(dn, n)
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def _pdm_mdm(h: pd.Series, l: pd.Series) -> tuple[pd.Series, pd.Series]:
    up_move = h.diff()
    dn_move = (-l.diff())
    pdm = up_move.where((up_move > dn_move) & (up_move > 0.0), 0.0)
    mdm = dn_move.where((dn_move > up_move) & (dn_move > 0.0), 0.0)
    return pdm, mdm


def _adx_wilder(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    pdm, mdm = _pdm_mdm(h, l)
    tr = _true_range(h, l, c)
    atr = _ema_wilder(tr, n)
    pdi = 100.0 * _ema_wilder(pdm, n) / atr.replace(0.0, np.nan)
    mdi = 100.0 * _ema_wilder(mdm, n) / atr.replace(0.0, np.nan)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)
    adx = _ema_wilder(dx, n)
    return pdi, mdi, adx


def _rolling_z(x: pd.Series, n: int) -> pd.Series:
    mu = x.rolling(n, min_periods=n).mean()
    sd = x.rolling(n, min_periods=n).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _donch(h: pd.Series, l: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    dh = h.rolling(n, min_periods=n).max()
    dl = l.rolling(n, min_periods=n).min()
    return dh, dl


def _kama(x: pd.Series, n: int) -> pd.Series:
    if n <= 1:
        return x.copy().astype("float64")
    change = (x - x.shift(n)).abs()
    vol = x.diff().abs().rolling(n, min_periods=n).sum()
    er = change / vol.replace(0.0, np.nan)
    sc_fast, sc_slow = 2.0 / (2 + 1), 2.0 / (30 + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
    kama = x.copy().astype("float64")
    kama.iloc[: n] = np.nan
    for i in range(n, len(x)):
        prev = kama.iloc[i - 1] if np.isfinite(kama.iloc[i - 1]) else x.iloc[i - 1]
        alpha = sc.iloc[i]
        kama.iloc[i] = prev + alpha * (x.iloc[i] - prev)
    return kama


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0.0, np.nan) + 0.0)


def _iso_from_ms(ms: pd.Series) -> pd.Series:
    # RFC3339 с миллисекундами, строгое "Z"-окончание
    ms = pd.to_numeric(ms, errors="coerce").astype("int64")
    base = pd.to_datetime(ms // 1000 * 1000, unit="ms", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S")
    mspart = (ms % 1000).astype(int).map(lambda x: f"{x:03d}")
    return (base + "." + mspart + "Z").astype("string")

# =============================
# Основной билд
# =============================

def compute_features(df_in: pd.DataFrame, *, symbol: str, tf: str, cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    cfg = cfg or DEFAULTS
    base = ensure_input(df_in)

    ts = base["ts"].astype("int64")
    o = base["o"].astype("float64")
    h = base["h"].astype("float64")
    l = base["l"].astype("float64")
    c = base["c"].astype("float64")
    v = base["v"].astype("float64")
    t = base["t"].astype("float64") if "t" in base.columns else None

    out = pd.DataFrame({
        "ts": ts,                      # канон C3
        "timestamp_ms": ts,           # для совместимости с внешними витринами
        "start_time_iso": _iso_from_ms(ts),
        "open": o, "high": h, "low": l, "close": c, "volume": v,
    })
    if t is not None:
        out["turnover"] = t
    out["symbol"] = str(symbol)
    out["tf"] = str(tf)

    # Базовые
    f_ret1 = c.pct_change()
    with np.errstate(divide="ignore"):
        f_logret1 = np.log(c).diff()

    f_rv = {n: f_logret1.rolling(n, min_periods=n).std(ddof=1) for n in cfg.rv}

    rng = (h - l).abs()
    body = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l

    f_tr = _true_range(h, l, c)
    f_atr = _ema_wilder(f_tr, cfg.adx)  # ATR по Уайлдеру на n=14 по умолчанию
    f_atr_pct = _safe_div(f_atr, c)
    f_atr_z = _rolling_z(f_atr, cfg.adx)

    # EMA и наклон
    ema_vals = {n: _ema(c, n) for n in cfg.ema}
    ema_slope = {n: _safe_div(ema_vals[n] - ema_vals[n].shift(1), ema_vals[n].shift(1)) for n in cfg.ema}

    # Моментум
    mom_vals = {n: _safe_div(c, c.shift(n)) - 1.0 for n in cfg.mom}

    # RSI/ADX
    f_rsi = _rsi_wilder(c, cfg.rsi)
    f_pdi, f_mdi, f_adx = _adx_wilder(h, l, c, cfg.adx)

    # Donchian
    dh = {}; dl = {}
    for n in cfg.donch:
        dh[n], dl[n] = _donch(h, l, n)
    donch_width_pct = {n: _safe_div(dh[n] - dl[n], c) for n in cfg.donch}
    donch_break_dir = {}
    for n in cfg.donch:
        br = np.where(c > dh[n], 1.0, np.where(c < dl[n], -1.0, 0.0))
        donch_break_dir[n] = pd.Series(br, index=c.index, dtype="float64")

    # Z‑scores
    z_close = {n: _rolling_z(c, n) for n in cfg.z}
    z_range = {n: _rolling_z(h - l, n) for n in cfg.z}
    z_vol = {n: _rolling_z(v.fillna(0.0), n) for n in cfg.z}

    # Up/Down volume
    dprice = c.diff()
    up = v.where(dprice > 0.0, 0.0)
    down = v.where(dprice < 0.0, 0.0)
    ud_n = cfg.updownvol
    f_upvol = up.rolling(ud_n, min_periods=ud_n).sum()
    f_downvol = down.rolling(ud_n, min_periods=ud_n).sum()
    f_vol_balance = (f_upvol - f_downvol) / ((f_upvol + f_downvol).replace(0.0, np.nan))

    # OBV
    sign = np.sign(dprice).fillna(0.0)
    f_obv = (sign * v).fillna(0.0).cumsum()

    # VWAPs
    typical_price = (h + l + c) / 3.0
    pv = typical_price * v
    f_vwap_roll = (pv.rolling(cfg.vwap_roll, min_periods=cfg.vwap_roll).sum() /
                   v.rolling(cfg.vwap_roll, min_periods=cfg.vwap_roll).sum().replace(0.0, np.nan))
    f_vwap_dev_pct = _safe_div(c - f_vwap_roll, f_vwap_roll)

    ts_dt = pd.to_datetime(ts, unit="ms", utc=True)
    day = ts_dt.floor("D")
    pv_cum = pv.groupby(day).cumsum()
    v_cum = v.groupby(day).cumsum()
    f_vwap_session = pv_cum / v_cum.replace(0.0, np.nan)
    f_vwap_session_dev_pct = _safe_div(c - f_vwap_session, f_vwap_session)

    # Прокси ликвидности/спрэда
    f_liq_proxy = (t if t is not None else (c * v)).astype("float64")
    f_spread_proxy = donch_width_pct[cfg.donch[0]]  # ширина Donchian fast

    # KAMA
    f_kama = _kama(c, cfg.kama)

    # Сборка выходных колонок
    put = out.__setitem__
    put("f_ret1", f_ret1.astype("float64"))
    put("f_logret1", f_logret1.astype("float64"))
    for n, s in f_rv.items():
        put(f"f_rv_{n}", s.astype("float64"))

    put("f_range_pct", _safe_div(rng, c).astype("float64"))
    put("f_body_pct", _safe_div(body, c).astype("float64"))
    put("f_wick_upper_pct", _safe_div(upper_wick, c).astype("float64"))
    put("f_wick_lower_pct", _safe_div(lower_wick, c).astype("float64"))

    put("f_tr", f_tr.astype("float64"))
    put(f"f_atr_{cfg.adx}", f_atr.astype("float64"))
    put(f"f_atr_pct_{cfg.adx}", f_atr_pct.astype("float64"))
    put(f"f_atr_z_{cfg.adx}", f_atr_z.astype("float64"))

    for n in cfg.ema:
        put(f"f_ema_close_{n}", ema_vals[n].astype("float64"))
        put(f"f_ema_slope_{n}", ema_slope[n].astype("float64"))

    for n in cfg.mom:
        put(f"f_mom_{n}", mom_vals[n].astype("float64"))

    put(f"f_rsi_{cfg.rsi}", f_rsi.astype("float64"))
    put(f"f_pdi_{cfg.adx}", f_pdi.astype("float64"))
    put(f"f_mdi_{cfg.adx}", f_mdi.astype("float64"))
    put(f"f_adx_{cfg.adx}", f_adx.astype("float64"))

    for n in cfg.donch:
        put(f"f_donch_h_{n}", dh[n].astype("float64"))
        put(f"f_donch_l_{n}", dl[n].astype("float64"))
        put(f"f_donch_break_dir_{n}", donch_break_dir[n].astype("float64"))
        put(f"f_donch_width_pct_{n}", donch_width_pct[n].astype("float64"))

    for n in cfg.z:
        put(f"f_close_z_{n}", z_close[n].astype("float64"))
        put(f"f_range_z_{n}", z_range[n].astype("float64"))
        put(f"f_vol_z_{n}", z_vol[n].astype("float64"))

    put(f"f_upvol_{ud_n}", f_upvol.astype("float64"))
    put(f"f_downvol_{ud_n}", f_downvol.astype("float64"))
    put(f"f_vol_balance_{ud_n}", f_vol_balance.astype("float64"))
    put("f_obv", f_obv.astype("float64"))

    put(f"f_vwap_roll_{cfg.vwap_roll}", f_vwap_roll.astype("float64"))
    put(f"f_vwap_dev_pct_{cfg.vwap_roll}", f_vwap_dev_pct.astype("float64"))
    put("f_vwap_session", f_vwap_session.astype("float64"))
    put("f_vwap_session_dev_pct", f_vwap_session_dev_pct.astype("float64"))

    put("f_liq_proxy", f_liq_proxy.astype("float64"))
    put("f_spread_proxy", f_spread_proxy.astype("float64"))
    put(f"f_kama_{cfg.kama}", f_kama.astype("float64"))

    # f_valid_from и f_build_version
    fcols = [c for c in out.columns if c.startswith("f_")]
    mask_all = ~out[fcols].isna().any(axis=1)
    valid_from_idx = int(mask_all.idxmax()) if mask_all.any() else 0
    out["f_valid_from"] = int(ts.iloc[valid_from_idx]) if len(ts) else 0
    out["f_build_version"] = str(cfg.build_version)

    # Канонический порядок: meta → f_*
    meta = ["ts", "timestamp_ms", "start_time_iso", "open", "high", "low", "close", "volume"]
    if "turnover" in out.columns:
        meta.append("turnover")
    meta += ["symbol", "tf", "f_valid_from", "f_build_version"]
    feat = [c for c in out.columns if c.startswith("f_")]
    out = out[meta + feat]

    # Типы
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce").astype("int64")
    for c in feat:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out["symbol"] = out["symbol"].astype("string")
    out["tf"] = out["tf"].astype("string")
    out["f_build_version"] = out["f_build_version"].astype("string")

    return out


# Совместимость старых имён API из тестов

def normalize_and_validate(df_in: pd.DataFrame) -> pd.DataFrame:
    base = ensure_input(df_in)
    ts = base["ts"].astype("int64")
    out = pd.DataFrame({
        "ts": ts,
        "timestamp_ms": ts,
        "start_time_iso": _iso_from_ms(ts),
        "open": base["o"].astype("float64"),
        "high": base["h"].astype("float64"),
        "low": base["l"].astype("float64"),
        "close": base["c"].astype("float64"),
        "volume": base["v"].astype("float64"),
    })
    if "t" in base.columns:
        out["turnover"] = base["t"].astype("float64")
    return out
