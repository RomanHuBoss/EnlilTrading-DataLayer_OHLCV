import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")  # требуется для parquet
pq = pytest.importorskip("pyarrow.parquet")

from ohlcv.io.parquet_store import write_idempotent, parquet_path


def _mk_1m(n: int, start: str = "2024-01-01T00:00:00Z", with_t: bool = True, with_gap: bool = True) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=n, freq="1T", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    v = np.ones(n)
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}, index=idx)
    if with_t:
        df["t"] = (df["o"] + df["h"] + df["l"] + df["c"]) / 4.0
    if with_gap:
        df["is_gap"] = False
    return df


def test_parquet_path_layout(tmp_path: Path):
    p = parquet_path(tmp_path, "BTCUSDT", "1m")
    assert str(p).endswith("BTCUSDT/1m.parquet")


def test_write_and_merge_idempotent(tmp_path: Path):
    root = tmp_path
    sym = "BTCUSDT"

    df1 = _mk_1m(5)
    path = write_idempotent(root, sym, "1m", df1)
    assert Path(path).exists()

    got = pd.read_parquet(path)
    assert set(["ts", "o", "h", "l", "c", "v"]).issubset(got.columns)
    assert len(got) == 5

    # перекрывающаяся вставка: последние 2 минуты с другими значениями close
    df2 = df1.iloc[-2:].copy()
    df2["c"] = df2["c"] + 10.0
    path2 = write_idempotent(root, sym, "1m", df2)
    assert path2 == path

    got2 = pd.read_parquet(path)
    # новые значения победили
    tail_close = got2["c"].astype(float).tail(2).to_numpy()
    assert np.allclose(tail_close, df2["c"].to_numpy())
    # без дубликатов по времени
    ts = pd.to_datetime(got2["ts"], utc=True)
    assert ts.is_monotonic_increasing and not ts.duplicated().any()


def test_footer_metadata_and_params(tmp_path: Path):
    root = tmp_path
    sym = "ETHUSDT"

    df = _mk_1m(3)
    path = write_idempotent(root, sym, "1m", df)

    # проверяем наличие user-metadata "c1.meta"
    md = pq.read_metadata(path)
    meta = md.metadata or {}
    assert b"c1.meta" in meta
    payload = json.loads(meta[b"c1.meta"].decode("utf-8"))

    # обязательные поля
    for k in ("symbol", "tf", "rows", "min_ts", "max_ts", "generated_at", "data_hash", "build_signature", "schema", "schema_version", "zstd_level", "row_group_size"):
        assert k in payload

    assert payload["symbol"] == sym
    assert payload["tf"] == "1m"
    assert int(payload["rows"]) >= 3
    assert int(payload["zstd_level"]) == 7
    assert int(payload["row_group_size"]) == 256_000


def test_optional_columns_preserved_and_order(tmp_path: Path):
    root = tmp_path
    sym = "SOLUSDT"

    df = _mk_1m(4, with_t=True, with_gap=True)
    path = write_idempotent(root, sym, "1m", df)

    got = pd.read_parquet(path)
    cols = list(got.columns)
    # порядок: ts, o,h,l,c,v, затем t, затем is_gap, затем хвост
    expected_prefix = ["ts", "o", "h", "l", "c", "v", "t", "is_gap"]
    assert cols[: len(expected_prefix)] == expected_prefix[: len(cols)]
