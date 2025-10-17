import numpy as np
import pandas as pd
import pytest

from ohlcv.core.resample import resample_ohlcv
from ohlcv.core.validate import ensure_missing_threshold, missing_rate, validate_1m_index


def _mk_df(n: int, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    low = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": low, "c": c, "v": v}, index=idx)


def test_validate_1m_index_ok() -> None:
    df = _mk_df(10)
    validate_1m_index(df)


def test_resample_basic_to_5m() -> None:
    df = _mk_df(10)
    out = resample_ohlcv(df, "5m")
    assert out.shape[0] == 2
    assert set(out.columns) == {"o", "h", "l", "c", "v"}
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None
    assert out.index.is_monotonic_increasing


def test_missing_threshold() -> None:
    df = _mk_df(100)
    drop_idx = df.index[10:20]
    df = df.drop(drop_idx)
    r = missing_rate(df)
    assert 0.09 < r < 0.11
    with pytest.raises(ValueError):
        ensure_missing_threshold(df, threshold=0.05)
