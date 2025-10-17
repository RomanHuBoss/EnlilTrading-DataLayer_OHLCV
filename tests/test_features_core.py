import numpy as np
import pandas as pd

from ohlcv.features.core import compute_features


def _mk_df(n: int = 300) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_ms": (ts.view("i8") // 10**6).astype("int64"),
            "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": np.linspace(1.0, 2.0, n),
            "high": np.linspace(1.0, 2.0, n) + 0.1,
            "low": np.linspace(1.0, 2.0, n) - 0.1,
            "close": np.linspace(1.0, 2.0, n),
            "volume": np.linspace(1.0, 2.0, n),
        }
    )
    return df


def test_basic_sanity() -> None:
    df = _mk_df(300)
    out = compute_features(df, "BTCUSDT", "5m", None)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(df)

    fcols = [c for c in out.columns if c.startswith("f_")]
    assert len(fcols) > 10
    assert out["f_valid_from"].iloc[-1] >= 60


def test_constants_series_vol_zero() -> None:
    df = _mk_df(1000)
    df["open"] = 1.0
    df["high"] = 1.0
    df["low"] = 1.0
    df["close"] = 1.0

    out = compute_features(df, "BTCUSDT", "5m", None)

    assert np.allclose(out["f_tr"], 0.0, atol=1e-12)
    assert np.allclose(out.filter(like="f_rv_").fillna(0.0), 0.0, atol=1e-12)
