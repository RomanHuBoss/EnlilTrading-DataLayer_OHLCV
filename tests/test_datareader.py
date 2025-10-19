from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")  # требуется для фильтров при чтении parquet

from ohlcv.datareader import DataReader
from ohlcv.io.parquet_store import write_idempotent
from ohlcv.core.resample import resample_1m_to

MINUTE_MS = 60_000


def _mk_1m(n: int, start_iso: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start_iso), periods=n, freq="1T", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}, index=idx)


def _drop_range(df: pd.DataFrame, i0: int, i1: int) -> pd.DataFrame:
    return df.drop(df.index[i0:i1])


def _ts_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def test_latest_ts_and_full_read(tmp_path: Path):
    root = tmp_path
    sym = "BTCUSDT"

    df = _mk_1m(10)
    write_idempotent(root, sym, "1m", df)

    r = DataReader(root)
    latest = r.latest_ts(sym, "1m")
    assert latest == _ts_ms(df.index[-1])

    out, st = r.read_range(sym, "1m")
    assert st.rows == 10 and not st.used_filters
    assert out.index.tz is not None and out.index.is_monotonic_increasing
    assert set(["o", "h", "l", "c", "v"]).issubset(out.columns)


def test_read_with_filters_and_columns(tmp_path: Path):
    root = tmp_path
    sym = "ETHUSDT"

    df = _mk_1m(30)
    write_idempotent(root, sym, "1m", df)

    start_ms = _ts_ms(df.index[10])
    end_ms = _ts_ms(df.index[25])

    r = DataReader(root)
    cols = ["o", "c"]
    out, st = r.read_range(sym, "1m", start_ms=start_ms, end_ms=end_ms, columns=cols)

    assert st.used_filters and 14 <= st.rows <= 15  # полуинтервал
    assert list(out.columns)[:2] == cols  # порядок колонок сохранён


def test_align_1m_adds_is_gap_and_counts(tmp_path: Path):
    root = tmp_path
    sym = "SOLUSDT"

    full = _mk_1m(60)
    sparse = _drop_range(full, 10, 20)  # 10 пропусков
    write_idempotent(root, sym, "1m", sparse)

    r = DataReader(root)
    start_ms = _ts_ms(full.index[0])
    end_ms = _ts_ms(full.index[-1] + pd.Timedelta(minutes=1))

    out, st = r.read_range(sym, "1m", start_ms=start_ms, end_ms=end_ms, align_1m=True)
    assert st.rows == 60
    assert "is_gap" in out.columns
    assert int(out["is_gap"].sum()) == 10


def test_read_day_and_iter_days(tmp_path: Path):
    root = tmp_path
    sym = "XRPUSDT"

    # 1 сутки по 1 минуте (сэкономим объём: 180 минут вместо 1440)
    df = _mk_1m(180, start_iso="2024-01-01T00:00:00Z")
    write_idempotent(root, sym, "1m", df)

    r = DataReader(root)
    day_df, st = r.read_day_utc(sym, "1m", "2024-01-01")
    assert st.rows == 180

    # iter_days на диапазоне из двух дней — второй день пустой
    days = list(r.iter_days(sym, "1m", "2024-01-01", "2024-01-03"))
    assert len(days) == 2
    d0, df0 = days[0]
    d1, df1 = days[1]
    assert d0 == "2024-01-01" and len(df0) == 180
    assert d1 == "2024-01-02" and len(df1) == 0


def test_reader_for_5m_without_align(tmp_path: Path):
    root = tmp_path
    sym = "ADAUSDT"

    src = _mk_1m(30)
    write_idempotent(root, sym, "1m", src)

    # агрегируем в 5m и сохраним
    agg = resample_1m_to(src, "5m")
    write_idempotent(root, sym, "5m", agg)

    r = DataReader(root)
    out, st = r.read_range(sym, "5m", columns=["o", "c", "is_gap"])  # без align_1m
    assert st.rows == len(agg)
    assert set(["o", "c"]).issubset(out.columns)
    assert "is_gap" in out.columns
