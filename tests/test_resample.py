import pandas as pd
import numpy as np
import pytest

from ohlcv.core.resample import resample_ohlcv
from ohlcv.core.validate import validate_1m_index, missing_rate, ensure_missing_threshold


def _mk_df(n: int, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    low = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": low, "c": c, "v": v}, index=idx)


def test_validate_ok():
    df = _mk_df(5)
    validate_1m_index(df)  # не должно бросить исключение


def test_resample_5m_shapes():
    df = _mk_df(10)
    out = resample_ohlcv(df, "5m")
    # 10 минут с окном 5m и политикой label="right", closed="left" → 2 бара
    assert out.shape[0] == 2
    # Проверка агрегатов по колонкам
    assert set(out.columns) == {"o", "h", "l", "c", "v"}


def test_missing_threshold():
    df = _mk_df(100)
    # удалим 10 последовательных минут (10%)
    drop_idx = df.index[10:20]
    df = df.drop(drop_idx)
    r = missing_rate(df)
    assert 0.09 < r < 0.11  # около 0.10
    with pytest.raises(ValueError):
        ensure_missing_threshold(df, threshold=0.05)  # 5% порог — должно упасть
