from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ohlcv.io.parquet_store import write_idempotent


def _mk_df(n: int, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="min", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    low = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": low, "c": c, "v": v}, index=idx)


def test_write_and_merge(tmp_path: Path) -> None:
    root = tmp_path
    sym = "BTCUSDT"
    tf = "1m"

    df1 = _mk_df(10, "2024-01-01T00:00:00Z")
    out1 = write_idempotent(root, sym, tf, df1)
    assert out1.exists()

    read1 = pd.read_parquet(out1)
    assert read1.shape[0] == 10
    assert "ts" in read1.columns

    df2 = _mk_df(10, "2024-01-01T00:05:00Z").copy()
    df2["c"] = df2["c"] + 1.0
    out2 = write_idempotent(root, sym, tf, df2)
    assert out2 == out1

    read2 = pd.read_parquet(out2)
    assert read2.shape[0] == 15

    ts_overlap = pd.Timestamp("2024-01-01T00:10:00Z", tz="UTC")
    row = read2.loc[read2["ts"] == ts_overlap]
    assert not row.empty

    df2_ts = df2.reset_index().rename(columns={"index": "ts"})
    df2_ts["ts"] = pd.to_datetime(df2_ts["ts"], utc=True)
    expected_close = float(df2_ts.loc[df2_ts["ts"] == ts_overlap, "c"].iloc[0])
    assert abs(float(row["c"].iloc[0]) - expected_close) < 1e-12

    md = pq.read_table(out2).schema.metadata or {}
    assert b"c1.meta" in md

    cols = list(read2.columns)
    assert cols[:6] == ["ts", "o", "h", "l", "c", "v"]
