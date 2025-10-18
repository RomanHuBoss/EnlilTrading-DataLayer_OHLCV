from __future__ import annotations

import numpy as np
import pandas as pd

from c4_regime.core import infer_regime


def _mk_df(n: int = 2000) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 1000.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, size=n))
    open_ = close
    high = close + 1
    low = close - 1
    vol = np.abs(np.random.default_rng(1).normal(100, 10, size=n))
    turn = close * vol
    df = pd.DataFrame(
        {
            "timestamp_ms": (ts.view("int64") // 1_000_000).astype("int64"),
            "start_time_iso": ts.astype("datetime64[ns]").astype(str),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
            "turnover": turn,
            "symbol": "BTCUSDT",
            "tf": "5m",
        }
    )
    # простые фичи, имитирующие C3
    df["f_logret1"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["f_close_z_60"] = (df["close"] - df["close"].rolling(60, min_periods=1).mean()) / df[
        "close"
    ].rolling(60, min_periods=1).std(ddof=1).fillna(1.0)
    df["f_adx14"] = 25.0
    df["f_pdi14"] = 20.0
    df["f_mdi14"] = 10.0
    df["f_donch_width_pct_20"] = (df["high"] - df["low"]) / df["close"].replace(0.0, 1.0)
    return df


def test_infer_smoke():
    df = _mk_df()
    out = infer_regime(df)
    # Проверки столбцов
    req = {
        "regime",
        "high_rv",
        "regime_confidence",
        "votes_trend",
        "votes_flat",
        "det_used",
        "chgpt",
        "p_bocpd_trend",
        "p_bocpd_flat",
        "s_adx",
        "s_donch",
        "s_hmm_rv",
        "hysteresis_state",
        "symbol",
        "tf",
        "build_version",
    }
    assert req.issubset(set(out.columns))
    assert len(out) == len(df)
