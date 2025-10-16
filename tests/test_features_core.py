import numpy as np
import pandas as pd

from ohlcv.features.core import compute_features


def _mk_df(n=300):
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_ms": (ts.view("i8") // 10**6).astype("int64"),
            "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": np.linspace(1.0, 2.0, n),
            "high": np.linspace(1.0, 2.0, n) + 0.1,
            "low": np.linspace(1.0, 2.0, n) - 0.1,
            "close": np.linspace(1.0, 2.0, n),
            "volume": np.ones(n),
            "turnover": np.linspace(100.0, 200.0, n),
        }
    )
    return df


def test_shapes_and_valid_from():
    df = _mk_df(200)
    out = compute_features(df, "BTCUSDT", "5m", None)
    assert "symbol" in out and "tf" in out
    fcols = [c for c in out.columns if c.startswith("f_")]
    assert len(fcols) > 10
    assert out["f_valid_from"].iloc[-1] >= 60


def test_constants_series_vol_zero():
    df = _mk_df(1000)
    df["open"] = 1.0
    df["high"] = 1.0
    df["low"] = 1.0
    df["close"] = 1.0
    out = compute_features(df, "BTCUSDT", "5m", None)
    assert np.allclose(out["f_tr"], 0.0, atol=1e-12)
    assert np.allclose(out.filter(like="f_rv_").fillna(0.0), 0.0, atol=1e-12)
