import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from ohlcv.io.parquet_store import write_idempotent, parquet_path


def _mk_df(idx):
    n = len(idx)
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}, index=idx)


def test_write_and_merge(tmp_path):
    root = tmp_path
    symbol = "TESTUSDT"
    tf = "1m"

    idx1 = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="min", tz="UTC")  # 00:00..00:02
    idx2 = pd.date_range("2024-01-01T00:02:00Z", periods=3, freq="min", tz="UTC")  # 00:02..00:04

    df1 = _mk_df(idx1)
    df2 = _mk_df(idx2) * 2.0  # разные значения, чтобы проверить "new wins"

    # первая запись
    p1 = write_idempotent(root, symbol, tf, df1)
    assert p1.exists()

    # вторая запись (с перекрытием 1 бара)
    p2 = write_idempotent(root, symbol, tf, df2)
    assert p2 == p1

    # чтение и проверка объединения
    df = pd.read_parquet(p1)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()

    # ожидание: 5 баров (00:00..00:04), перекрытие на 00:02 взято из df2
    assert df.shape[0] == 5

    ts_overlap = pd.Timestamp("2024-01-01T00:02:00Z", tz="UTC")
    # значения на перекрытии должны соответствовать df2*2.0
    for col in ["o", "h", "l", "c", "v"]:
        assert np.isclose(df.loc[ts_overlap, col], df2.loc[ts_overlap, col])

    # метаданные Parquet
    t = pq.read_table(p1)
    md = t.schema.metadata or {}
    assert b"c1.meta" in md


def test_idempotent_rewrite_same_data(tmp_path):
    root = tmp_path
    symbol = "TEST2"
    tf = "1m"
    idx = pd.date_range("2024-01-02T00:00:00Z", periods=4, freq="min", tz="UTC")
    df = _mk_df(idx)

    p = write_idempotent(root, symbol, tf, df)
    # повторная запись тех же данных не должна менять форму и дублировать строки
    p = write_idempotent(root, symbol, tf, df)
    out = pd.read_parquet(p)
    assert out.shape[0] == 4
