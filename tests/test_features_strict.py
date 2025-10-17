import numpy as np
import pandas as pd
import pytest

from ohlcv.features.core import compute_features


def _mk_df(n: int = 100) -> pd.DataFrame:
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
        }
    )
    return df


def test_strict_raises_on_nan() -> None:
    df = _mk_df(50)
    df.loc[10, "close"] = np.nan
    with pytest.raises(Exception):
        compute_features(df, "BTCUSDT", "5m", {"strict": True})


def test_strict_passes_on_clean() -> None:
    df = _mk_df(200)
    out = compute_features(df, "BTCUSDT", "5m", {"strict": True})
    assert "f_valid_from" in out.columns
    assert out.filter(like="f_").shape[1] >= 20
