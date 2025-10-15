# Тесты ресемплинга и валидации.
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from ohlcv.core.resample import resample_ohlcv
from ohlcv.core.validate import validate_1m_index, ensure_missing_threshold

def _mk_df(n=10, start="2024-01-01T00:00:00+00:00"):
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")
    df = pd.DataFrame({
        "o": np.arange(n, dtype=float),
        "h": np.arange(n, dtype=float) + 1,
        "l": np.arange(n, dtype=float) - 1,
        "c": np.arange(n, dtype=float) + 0.5,
        "v": np.ones(n, dtype=float),
    }, index=idx)
    return df

def test_validate_ok():
    df = _mk_df(10)
    validate_1m_index(df)

def test_resample_5m_shapes():
    df = _mk_df(10)
    out = resample_ohlcv(df, "5m")
    assert out.shape[0] == 2
    assert list(out.columns) == ["o","h","l","c","v"]

def test_missing_threshold():
    df = _mk_df(10)
    df = df.drop(index=df.index[5])
    ensure_missing_threshold(df, threshold=0.2)
