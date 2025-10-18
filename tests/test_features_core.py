# tests/test_features_core.py
from __future__ import annotations

import numpy as np
import pandas as pd

from ohlcv.features import compute_features, DEFAULTS, normalize_and_validate


def _mk_df(n: int = 120, start_ts: int = 1_700_000_000_000, step_ms: int = 300_000) -> pd.DataFrame:
    ts = np.arange(start_ts, start_ts + n * step_ms, step_ms, dtype=np.int64)
    # Детеминированный тренд с небольшой волной
    base = 100.0 + np.linspace(0, 1.2, n)
    wave = 0.2 * np.sin(np.linspace(0, 6.28, n))
    close = base + wave
    open_ = np.r_[close[0], close[:-1]]  # open[t] = close[t-1]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1_000.0 + (np.arange(n) % 10) * 10.0
    df = pd.DataFrame(
        {
            "timestamp_ms": ts,
            "start_time_iso": pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return df


def test_compute_features_schema_and_basic_columns():
    df = _mk_df()
    out = compute_features(df, symbol="TEST", tf="5m")

    # Идентификаторы
    assert "timestamp_ms" in out.columns
    assert "symbol" in out.columns and out["symbol"].iat[0] == "TEST"
    assert "tf" in out.columns and out["tf"].iat[0] == "5m"

    # Ключевые признаки из постановки
    need = [
        "f_rv_20", "f_rv_60",
        "f_adx_14", "f_pdi_14", "f_mdi_14",
        "f_donch_width_pct_20", "f_donch_width_pct_55",
        "f_range_pct", "f_body_pct", "f_wick_upper_pct", "f_wick_lower_pct",
        "f_ema_close_20", "f_ema_slope_20", "f_mom_20",
        "f_rsi_14",
        "f_close_z_20", "f_range_z_20", "f_vol_z_20",
        "f_upvol_20", "f_downvol_20", "f_vol_balance_20",
        "f_vwap_roll_96", "f_vwap_dev_pct_96",
        "f_vwap_session", "f_vwap_session_dev_pct",
        "f_valid_from", "f_build_version",
    ]
    missing = [c for c in need if c not in out.columns]
    assert not missing, f"missing columns: {missing}"

    # Значения конечны
    assert np.isfinite(out.select_dtypes(include=[float]).to_numpy()).all()


def test_f_rv_matches_unbiased_std_definition():
    df = _mk_df()
    out = compute_features(df)
    # ручной расчёт std(logret, 20) ddof=1
    close = df["close"].astype(float)
    logret = (np.log(close) - np.log(close.shift(1))).fillna(0.0)
    manual = logret.rolling(20, min_periods=20).std(ddof=1).fillna(0.0)
    assert np.allclose(out["f_rv_20"].values, manual.values, atol=1e-12, rtol=0)


def test_donchian_width_pct_and_break_range():
    df = _mk_df()
    out = compute_features(df)
    width = out["f_donch_width_20"]
    pct = out["f_donch_width_pct_20"]
    close = df["close"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        manual_pct = (width / close.replace(0.0, np.nan)).fillna(0.0)
    assert np.allclose(pct.values, manual_pct.values, atol=1e-12, rtol=0)
    # направление пробоя в допустимом диапазоне
    brk = out["f_donch_break_dir_20"].values
    assert ((brk == -1) | (brk == 0) | (brk == 1)).all()


def test_vwap_roll_and_session_consistency():
    df = _mk_df()
    cfg = dict(DEFAULTS)
    cfg.update({"vwap_roll_window": 5})
    out = compute_features(df, config=cfg)

    # ручной скользящий VWAP(5)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    num = pv.rolling(5, min_periods=5).sum()
    den = df["volume"].rolling(5, min_periods=5).sum()
    manual_roll = (num / den.replace(0.0, np.nan))
    got_roll = out["f_vwap_roll_5"]
    assert np.allclose(got_roll.fillna(0.0).values, manual_roll.fillna(0.0).values, atol=1e-12, rtol=0)

    # сессионный VWAP в пределах одного дня = cumulative pv/v
    ts = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    day = ts.dt.floor("D")
    pv_cum = pv.groupby(day).cumsum()
    v_cum = df["volume"].groupby(day).cumsum()
    manual_sess = pv_cum / v_cum.replace(0.0, np.nan)
    got_sess = out["f_vwap_session"]
    assert np.allclose(got_sess.fillna(0.0).values, manual_sess.fillna(0.0).values, atol=1e-12, rtol=0)


def test_valid_from_and_version_not_empty():
    df = _mk_df(n=64)
    cfg = {
        "rv_windows": [3, 4],
        "z_windows": [3],
        "ema_windows": [3],
        "sma_windows": [3],
        "mom_windows": [3],
        "donch_windows": [3],
        "atr_window": 3,
        "adx_window": 3,
        "rsi_window": 3,
        "vwap_roll_window": 5,
        "updownvol_window": 4,
    }
    out = compute_features(df, config=cfg)
    assert int(out["f_valid_from"].iat[-1]) == 5  # max окон из cfg
    bv = str(out["f_build_version"].iat[-1])
    assert isinstance(bv, str) and len(bv) > 0


def test_bounds_and_sanity_checks():
    df = _mk_df()
    out = compute_features(df)
    # RSI в [0, 100]
    rsi_col = [c for c in out.columns if c.startswith("f_rsi_")][0]
    assert (out[rsi_col] >= 0).all() and (out[rsi_col] <= 100).all()
    # баланс объёма в [-1, 1]
    vb_col = [c for c in out.columns if c.startswith("f_vol_balance_")][0]
    vb = out[vb_col]
    assert (vb <= 1.0 + 1e-12).all() and (vb >= -1.0 - 1e-12).all()
