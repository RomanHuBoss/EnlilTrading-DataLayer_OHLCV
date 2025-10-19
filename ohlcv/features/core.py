from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["FeatureConfig", "build_features", "ensure_input"]


# =============================
# Конфиг
# =============================
@dataclass(frozen=True)
class FeatureConfig:
    # окна (по барам tf)
    atr: int = 14
    rsi: int = 14
    adx: int = 14
    donch: int = 20
    z_price: int = 60
    z_vol: int = 60
    vwap_roll: int = 96  # условно ≈ 1 сессия для 15m (настраивается снаружи)
    # версия сборки
    build_version: str = "C3-1.0"


# =============================
# Вспомогательные
# =============================

def ensure_input(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализация входа C3.

    Принимает схему с колонками:
      - либо marketdata: timestamp_ms,start_time_iso,open,high,low,close,volume[,turnover]
      - либо уже нормализованную: ts,o,h,l,c,v[,t]

    Возвращает DataFrame со столбцами ts(int64), o,h,l,c,v[,t], отсортированный и дедуплицированный.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("ожидается DataFrame")
    out = df.copy()

    # Переименование при необходимости
    rename = {
        "timestamp_ms": "ts",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
        "turnover": "t",
    }
    out = out.rename(columns=rename)

    # Индекс→ts при DatetimeIndex
    if "ts" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        ts = out.index
        ts = ts.tz_convert("UTC") if ts.tz is not None else ts.tz_localize("UTC")
        out["ts"] = (ts.view("int64") // 1_000_000).astype("int64")

    if "ts" not in out.columns:
        # старт времени из start_time_iso если есть
        if "start_time_iso" in out.columns:
            tsi = pd.to_datetime(out["start_time_iso"], utc=True, errors="coerce")
            out["ts"] = (tsi.view("int64") // 1_000_000).astype("int64")
        else:
            raise ValueError("нет колонки 'ts'/'timestamp_ms'/'start_time_iso' и нет DatetimeIndex")

    out["ts"] = pd.to_numeric(out["ts"], errors="coerce").astype("int64")
    for c in ["o", "h", "l", "c", "v", "t"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    out = (
        out.sort_values("ts")
        .drop_duplicates(subset=["ts"], keep="last")
        .reset_index(drop=True)
    )

    req = ["o", "h", "l", "c", "v"]
    for c in req:
        if c not in out.columns:
            out[c] = np.nan
    return out[[c for c in ["ts", "o", "h", "l", "c", "v", "t"] if c in out.columns]]


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    roll_up = _ema(up, n)
    roll_dn = _ema(dn, n)
    rs = roll_up / roll_dn.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _pdm_mdm(h: pd.Series, l: pd.Series) -> Tuple[pd.Series, pd.Series]:
    up_move = h.diff()
    dn_move = (-l.diff())
    pdm = up_move.where((up_move > dn_move) & (up_move > 0.0), 0.0)
    mdm = dn_move.where((dn_move > up_move) & (dn_move > 0.0), 0.0)
    return pdm, mdm


def _adx(h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    pdm, mdm = _pdm_mdm(h, l)
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = _ema(tr, n)
    pdi = 100.0 * _ema(pdm, n) / atr.replace(0.0, np.nan)
    mdi = 100.0 * _ema(mdm, n) / atr.replace(0.0, np.nan)
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan))
    adx = _ema(dx, n)
    return pdi, mdi, adx


def _rolling_z(x: pd.Series, n: int) -> pd.Series:
    mu = x.rolling(n, min_periods=max(5, n // 5)).mean()
    sd = x.rolling(n, min_periods=max(5, n // 5)).std(ddof=1)
    return (x - mu) / sd.replace(0.0, np.nan)


def _donch(h: pd.Series, l: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    dh = h.rolling(n, min_periods=max(2, n // 4)).max()
    dl = l.rolling(n, min_periods=max(2, n // 4)).min()
    return dh, dl


def _rolling_vwap(c: pd.Series, v: pd.Series, n: int) -> pd.Series:
    pv = (c * v).rolling(n, min_periods=max(2, n // 4)).sum()
    vv = v.rolling(n, min_periods=max(2, n // 4)).sum()
    return pv / vv.replace(0.0, np.nan)


# =============================
# Основной билд
# =============================

def build_features(
    df_in: pd.DataFrame,
    *,
    symbol: str,
    tf: str,
    cfg: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Строит фичи C3 по нормализованному входу.

    Выход: DataFrame со столбцами
      [ts, symbol, tf, f_valid_from, f_build_version, f_*]
    Все числовые фичи — float64. ts — int64.
    """
    cfg = cfg or FeatureConfig()
    df = ensure_input(df_in)

    ts = df["ts"].astype("int64")
    o, h, l, c, v = df["o"].astype("float64"), df["h"].astype("float64"), df["l"].astype("float64"), df["c"].astype("float64"), df["v"].astype("float64")

    out = pd.DataFrame({"ts": ts})

    # Доходности
    f_ret1 = c.pct_change()
    with np.errstate(divide="ignore"):
        f_logret1 = np.log(c).diff()

    # TR/ATR
    tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f_tr = tr
    f_atr = _ema(tr, cfg.atr)

    # RSI
    f_rsi = _rsi(c, cfg.rsi)

    # ADX и DIs
    f_pdi, f_mdi, f_adx = _adx(h, l, c, cfg.adx)

    # Donchian
    don_h, don_l = _donch(h, l, cfg.donch)

    # VWAP (rolling)
    f_vwap = _rolling_vwap(c, v, cfg.vwap_roll)

    # Z-score цены и объёма
    f_close_z = _rolling_z(c, cfg.z_price)
    f_vol_z = _rolling_z(v.fillna(0.0), cfg.z_vol)

    # Up/Down volume и баланс
    up = v.where(c.diff() > 0.0, 0.0)
    down = v.where(c.diff() < 0.0, 0.0)
    f_upvol = up.rolling(cfg.z_vol, min_periods=max(2, cfg.z_vol // 5)).sum()
    f_downvol = down.rolling(cfg.z_vol, min_periods=max(2, cfg.z_vol // 5)).sum()
    f_vol_balance = (f_upvol - f_downvol) / (f_upvol + f_downvol).replace(0.0, np.nan)

    # Сборка
    out["symbol"] = str(symbol)
    out["tf"] = str(tf)

    out["f_ret1"] = f_ret1.astype("float64")
    out["f_logret1"] = f_logret1.astype("float64")
    out["f_tr"] = f_tr.astype("float64")
    out["f_atr"] = f_atr.astype("float64")
    out["f_rsi14"] = f_rsi.astype("float64")
    out["f_pdi14"] = f_pdi.astype("float64")
    out["f_mdi14"] = f_mdi.astype("float64")
    out["f_adx14"] = f_adx.astype("float64")
    out["f_donch_h"] = don_h.astype("float64")
    out["f_donch_l"] = don_l.astype("float64")
    out["f_vwap_roll"] = f_vwap.astype("float64")
    out["f_close_z"] = f_close_z.astype("float64")
    out["f_vol_z"] = f_vol_z.astype("float64")
    out["f_upvol"] = f_upvol.astype("float64")
    out["f_downvol"] = f_downvol.astype("float64")
    out["f_vol_balance"] = f_vol_balance.astype("float64")

    # Граница валидности: первый индекс, где нет NaN в ключевых фичах
    core_cols = [
        "f_atr", "f_rsi14", "f_pdi14", "f_mdi14", "f_adx14",
        "f_donch_h", "f_donch_l", "f_vwap_roll",
    ]
    mask_valid = ~out[core_cols].isna().any(axis=1)
    if mask_valid.any():
        f_valid_from = int(out.loc[mask_valid, "ts"].iloc[0])
    else:
        f_valid_from = int(out["ts"].iloc[-1]) if len(out) else 0
    out["f_valid_from"] = f_valid_from
    out["f_build_version"] = cfg.build_version

    # Порядок колонок: служебные → фичи
    first = ["ts", "symbol", "tf", "f_valid_from", "f_build_version"]
    rest = [c for c in out.columns if c not in first]
    out = out[first + rest]

    # Стабилизация типов
    out["ts"] = out["ts"].astype("int64")
    for c in out.columns:
        if c not in ("ts", "symbol", "tf", "f_build_version"):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype("float64")
    return out
