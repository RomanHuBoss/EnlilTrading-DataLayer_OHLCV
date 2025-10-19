import numpy as np
import pandas as pd
import pytest

from ohlcv.core.resample import resample_1m_to
from ohlcv.core.validate import normalize_ohlcv_1m, align_and_flag_gaps


def _mk_df(n: int, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    low = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": low, "c": c, "v": v}, index=idx)


def test_normalize_1m_ok() -> None:
    df = _mk_df(10)
    out, stats = normalize_ohlcv_1m(df)
    assert out.index.tz is not None and out.index.is_monotonic_increasing
    assert set(["o", "h", "l", "c", "v"]).issubset(out.columns)
    assert stats.rows_in == 10 and stats.rows_out == 10


def test_resample_basic_to_5m() -> None:
    df = _mk_df(10)
    out = resample_1m_to(df, "5m")
    assert out.shape[0] == 2
    assert set(out.columns) >= {"o", "h", "l", "c", "v", "is_gap"}
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None
    assert out.index.is_monotonic_increasing


def test_missing_rate_alignment() -> None:
    df = _mk_df(100)
    drop_idx = df.index[10:20]
    df = df.drop(drop_idx)

    aligned, stats = align_and_flag_gaps(df)
    total = len(aligned)
    gaps = int(aligned["is_gap"].sum())
    rate = gaps / total
    assert 0.09 < rate < 0.11

    # Порог 5% должен считаться превышенным
    assert rate > 0.05
