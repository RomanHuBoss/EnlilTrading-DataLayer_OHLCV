# ohlcv/features/core.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple, Optional

import numpy as np
import pandas as pd

from .schema import normalize_and_validate
from .utils import BuildMeta

EPS = 1e-12

# Параметры по умолчанию для набора фич — согласованы с постановкой C3 (минимальный перечень)
DEFAULTS: Dict[str, Any] = {
    "rv_windows": [20, 60],                 # rolling std(logret, n), ddof=1
    "z_windows": [20, 96],                  # окна для z-score по close/range/volume/ATR
    "ema_windows": [20, 50],                # EMA по close
    "sma_windows": [20, 50],                # SMA по close (для контрольных проверок/зеток)
    "mom_windows": [20, 50, 100],           # моментумы по цене закрытия
    "donch_windows": [20, 55],              # Donchian HH/LL/width и ширина в %
    "atr_window": 14,                       # ATR (Wilder)
    "adx_window": 14,                       # ADX/DI (Wilder)
    "rsi_window": 14,                       # RSI классический
    "vwap_roll_window": 96,                 # ~2 торговые сессии при 5m
    "updownvol_window": 20,                 # окна для upvol/downvol/баланса
    "strict": False,                        # режим строгой проверки входа
    "build_version": None,                  # необязательная версия сборки
}

# ----------------
# Базовые примитивы
# ----------------

def _safe_rolling(series: pd.Series, window: int, fn: str) -> pd.Series:
    if window <= 1:
        return series
    roll = series.rolling(window, min_periods=window)
    if fn == "mean":
        return roll.mean()
    if fn == "std":
        return roll.std(ddof=1)
    if fn == "sum":
        return roll.sum()
    if fn == "max":
        return roll.max()
    if fn == "min":
        return roll.min()
    raise ValueError(f"unsupported rolling fn: {fn}")

def ema(x: pd.Series, n: int) -> pd.Series:
    alpha = 2.0 / (n + 1.0)
    return x.ewm(alpha=alpha, adjust=False, min_periods=n).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    hl = (high - low)
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.fillna(0.0)

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    tr = true_range(high, low, close)
    # Wilder smoothing: EMA с alpha = 1/n
    return tr.ewm(alpha=1.0 / float(n), adjust=False, min_periods=n).mean()

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
    mu = _safe_rolling(x, n, "mean")
    sd = _safe_rolling(x, n, "std")
    z = (x - mu) / (sd.replace(0.0, np.nan))
    return z.replace([np.inf, -np.inf], np.nan)

def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = ema(gain, n)
    avg_loss = ema(loss, n)
    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(0.0)

def _typical_price(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high + low + close) / 3.0

def vwap_roll(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int
) -> pd.Series:
    tp = _typical_price(high, low, close)
    pv = tp * volume
    num = _safe_rolling(pv, n, "sum")
    den = _safe_rolling(volume, n, "sum")
    vwap = num / den.replace(0.0, np.nan)
    return vwap

# ----------------
# Основные фичи
# ----------------

def compute_features(
    df: pd.DataFrame,
    symbol: Optional[str] = None,
    tf: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Строит признаки C3 на минутно-агрегированном OHLCV."""

    cfg = {**DEFAULTS, **(config or {})}

    # Нормализация входа и строгие проверки NaN в базовых колонках
    df = normalize_and_validate(df.copy(), strict=bool(cfg.get("strict", False)))

    # Базовые серии
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index, dtype="float64")).astype(float)

    # Доходности
    ret = close.pct_change().fillna(0.0)
    logret = (np.log(close.replace(0.0, np.nan)) - np.log(close.replace(0.0, np.nan)).shift(1)).fillna(0.0)

    # Волатильность (реализованная) — несмещённая std(logret, n)
    out = pd.DataFrame(index=df.index)
    for n in cfg["rv_windows"]:
        out[f"f_rv_{n}"] = _safe_rolling(logret, int(n), "std").fillna(0.0)

    # True Range / ATR
    out["f_tr"] = true_range(high, low, close)
    atr_n = int(cfg["atr_window"])
    out[f"f_atr_{atr_n}"] = atr_wilder(high, low, close, atr_n)

    # ATR в % к цене и Z-оценка ATR
    for n in cfg["z_windows"]:
        out[f"f_atr_pct_{n}"] = (out[f"f_atr_{atr_n}"] / (close.replace(0.0, np.nan))).fillna(0.0)
        out[f"f_atr_z_{n}"] = zscore(out[f"f_atr_{atr_n}"], int(n)).fillna(0.0)

    # DI/ADX
    adx_n = int(cfg["adx_window"])
    pdi, mdi, adx = di_adx(high, low, close, adx_n)
    out[f"f_pdi_{adx_n}"] = pdi
    out[f"f_mdi_{adx_n}"] = mdi
    out[f"f_adx_{adx_n}"] = adx

    # Donchian HH/LL/width и в процентах к цене
    for n in cfg["donch_windows"]:
        hh = _safe_rolling(high, int(n), "max")
        ll = _safe_rolling(low, int(n), "min")
        width = (hh - ll)
        out[f"f_donch_hh_{n}"] = hh
        out[f"f_donch_ll_{n}"] = ll
        out[f"f_donch_width_{n}"] = width
        out[f"f_donch_width_pct_{n}"] = (width / close.replace(0.0, np.nan)).fillna(0.0)
        # Направление пробоя: 1, если close > hh.shift(1); -1, если close < ll.shift(1); 0 иначе.
        prev_hh = hh.shift(1)
        prev_ll = ll.shift(1)
        brk = pd.Series(0.0, index=df.index)
        brk = brk.mask(close > prev_hh, 1.0)
        brk = brk.mask(close < prev_ll, -1.0)
        out[f"f_donch_break_dir_{n}"] = brk.fillna(0.0)

    # Свечные относительные метрики
    body = (close - open_).abs()
    range_ = (high - low)
    upper_wick = (high - np.maximum(open_, close))
    lower_wick = (np.minimum(open_, close) - low)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["f_range_pct"] = (range_ / close.replace(0.0, np.nan)).fillna(0.0)
        out["f_body_pct"] = (body / (range_.replace(0.0, np.nan))).fillna(0.0)
        out["f_wick_upper_pct"] = (upper_wick / (range_.replace(0.0, np.nan))).fillna(0.0)
        out["f_wick_lower_pct"] = (lower_wick / (range_.replace(0.0, np.nan))).fillna(0.0)

    # EMA/SMA и наклоны EMA
    for n in cfg["ema_windows"]:
        ema_n = ema(close, int(n))
        out[f"f_ema_close_{n}"] = ema_n
        # slope как относительное приращение EMA за окно n, в долях
        out[f"f_ema_slope_{n}"] = (ema_n - ema_n.shift(int(n))) / (ema_n.shift(int(n)).replace(0.0, np.nan))
        out[f"f_ema_slope_{n}"] = out[f"f_ema_slope_{n}"].fillna(0.0)

    for n in cfg["sma_windows"]:
        out[f"f_sma_close_{n}"] = _safe_rolling(close, int(n), "mean")

    # Моментумы по цене закрытия
    for n in cfg["mom_windows"]:
        out[f"f_mom_{n}"] = (close / close.shift(int(n)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # RSI
    out[f"f_rsi_{int(cfg['rsi_window'])}"] = rsi(close, int(cfg["rsi_window"]))

    # Z-score по цене закрытия, размаху и объёму
    for n in cfg["z_windows"]:
        out[f"f_close_z_{n}"] = zscore(close, int(n)).fillna(0.0)
        out[f"f_range_z_{n}"] = zscore(range_, int(n)).fillna(0.0)
        out[f"f_vol_z_{n}"] = zscore(volume.replace(0.0, np.nan).fillna(0.0), int(n)).fillna(0.0)

    # Up/Down volume и баланс
    w = int(cfg["updownvol_window"])
    up_mask = close.diff().fillna(0.0) > 0
    down_mask = close.diff().fillna(0.0) < 0
    upvol = (volume.where(up_mask, 0.0)).rolling(w, min_periods=w).sum()
    downvol = (volume.where(down_mask, 0.0)).rolling(w, min_periods=w).sum()
    out[f"f_upvol_{w}"] = upvol.fillna(0.0)
    out[f"f_downvol_{w}"] = downvol.fillna(0.0)
    out[f"f_vol_balance_{w}"] = ((upvol - downvol) / (upvol + downvol).replace(0.0, np.nan)).fillna(0.0)

    # VWAP: скользящий и сессионный
    vrw = int(cfg["vwap_roll_window"])
    out[f"f_vwap_roll_{vrw}"] = vwap_roll(high, low, close, volume, vrw)
    out[f"f_vwap_dev_pct_{vrw}"] = ((close - out[f"f_vwap_roll_{vrw}"]) / out[f"f_vwap_roll_{vrw}"].replace(0.0, np.nan)).fillna(0.0)

    # Сессионный VWAP по календарному дню (UTC) — если есть timestamp_ms
    if "timestamp_ms" in df.columns:
        # индекс UTC мс -> дата
        ts = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        day = ts.dt.floor("D")
        tp = _typical_price(high, low, close)
        # накопительные суммы в пределах дня
        pv = (tp * volume).groupby(day).cumsum()
        vv = volume.groupby(day).cumsum()
        sess_vwap = pv / vv.replace(0.0, np.nan)
        out["f_vwap_session"] = sess_vwap
        out["f_vwap_session_dev_pct"] = ((close - sess_vwap) / sess_vwap.replace(0.0, np.nan)).fillna(0.0)
    else:
        out["f_vwap_session"] = np.nan
        out["f_vwap_session_dev_pct"] = 0.0

    # Столбцы-идентификаторы и метаданные сборки
    out.insert(0, "timestamp_ms", df["timestamp_ms"].astype("int64"))
    out.insert(1, "start_time_iso", df["start_time_iso"].astype("string") if "start_time_iso" in df.columns else pd.Series([""], index=df.index, dtype="string"))
    if symbol is not None:
        out["symbol"] = str(symbol)
    else:
        out["symbol"] = ""
    out["tf"] = str(tf) if tf is not None else ""

    # Минимальная длина «прогрева» для валидности всех фич — максимум окон
    warmup = max(
        int(max(cfg["rv_windows"])),
        int(max(cfg["z_windows"])),
        int(max(cfg["ema_windows"])),
        int(max(cfg["sma_windows"])),
        int(max(cfg["donch_windows"])),
        int(cfg["atr_window"]),
        int(cfg["adx_window"]),
        int(cfg["vwap_roll_window"]),
        int(cfg["updownvol_window"]),
    )
    out["f_valid_from"] = int(warmup)

    # Версия сборки: либо из конфига, либо вычисленная
    if cfg.get("build_version"):
        bv = str(cfg["build_version"])
    else:
        bv = BuildMeta(component="C3", version="0.1.0", params={k: v for k, v in cfg.items() if k != "build_version"}).build_version()
    out["f_build_version"] = bv

    # Чистка бесконечностей
    out = out.replace([np.inf, -np.inf], np.nan)

    return out
