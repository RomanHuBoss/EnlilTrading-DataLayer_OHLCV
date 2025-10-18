# tests/test_features_strict.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ohlcv.features.schema import normalize_and_validate


def test_strict_missing_required_raises():
    df = pd.DataFrame({
        "timestamp_ms": [1, 2, 3],
        "open": [1.0, 1.0, np.nan],  # NaN в обязательной колонке → ошибка
        "high": [2.0, 2.0, 2.0],
        "low": [0.5, 0.5, 0.5],
        "close": [1.2, 1.1, 1.0],
    })
    with pytest.raises(ValueError):
        normalize_and_validate(df, strict=True)


def test_strict_high_low_invariant_raises():
    df = pd.DataFrame({
        "timestamp_ms": [1, 2, 3],
        "open": [1.0, 1.0, 1.0],
        "high": [0.9, 2.0, 2.0],  # high<low
        "low": [1.0, 0.5, 0.5],
        "close": [1.0, 1.1, 1.2],
    })
    with pytest.raises(ValueError):
        normalize_and_validate(df, strict=True)


def test_non_strict_repairs_and_sorts_and_dedups():
    df = pd.DataFrame({
        "timestamp_ms": [2, 1, 1, 3],  # невозрастающий + дубликат ts
        "open": [1.0, 1.0, 0.9, 1.1],
        "high": [0.9, 2.0, 2.0, 2.0],  # high<low в первой строке
        "low": [1.0, 0.5, 0.5, 0.5],
        "close": [1.0, 1.1, 1.0, 1.2],
        "volume": [np.nan, 10.0, 11.0, -5.0],
    })
    out = normalize_and_validate(df, strict=False)
    # дубликат удалён (keep=last), сортировка по времени
    assert out["timestamp_ms"].tolist() == [1, 2, 3]
    # high/low отремонтированы, volume>=0 и заполнен NaN
    assert (out["high"] >= out["low"]).all()
    assert (out["volume"] >= 0).all()


def test_timestamp_ms_from_datetime_index():
    ts = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [1.0, 1.0, 1.0],
        "high": [1.2, 1.2, 1.2],
        "low": [0.8, 0.8, 0.8],
        "close": [1.1, 1.1, 1.1],
    }, index=ts)
    out = normalize_and_validate(df)
    assert "timestamp_ms" in out.columns and out["timestamp_ms"].dtype == "int64"
    assert "start_time_iso" in out.columns
