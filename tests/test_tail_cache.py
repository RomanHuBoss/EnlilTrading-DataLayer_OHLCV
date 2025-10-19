from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ohlcv.io import tail_cache as tail
from ohlcv.io.parquet_store import write_idempotent


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1T", tz="UTC")
    o = np.arange(5, dtype=float) + 100.0
    df = pd.DataFrame({
        "o": o,
        "h": o + 0.2,
        "l": o - 0.2,
        "c": o + 0.1,
        "v": 1.0,
    }, index=idx)
    return df


def _sidecar_path(root: Path, symbol: str, tf: str) -> Path:
    return root / symbol / f"{tf}.latest.json"


def test_read_default_when_missing(tmp_path: Path):
    info = tail.read(tmp_path, "BTCUSDT", "1m")
    assert info.symbol == "BTCUSDT" and info.tf == "1m"
    assert info.latest_ts_ms is None and info.rows_total is None and info.data_hash is None
    assert isinstance(info.updated_at, str) and len(info.updated_at) > 0


def test_write_and_read_roundtrip(tmp_path: Path):
    info_in = tail.TailInfo(symbol="ETHUSDT", tf="5m", latest_ts_ms=1700000000000, rows_total=123, data_hash="deadbeef", updated_at="2024-01-01T00:00:00Z")
    p = tail.write(tmp_path, info_in)
    assert p.exists()

    raw = json.loads(p.read_text("utf-8"))
    for k in ("symbol", "tf", "latest_ts_ms", "rows_total", "data_hash", "updated_at"):
        assert k in raw

    info_out = tail.read(tmp_path, "ETHUSDT", "5m")
    assert info_out == info_in


def test_update_convenience(tmp_path: Path):
    info = tail.update(tmp_path, "SOLUSDT", "15m", latest_ts_ms=1700000100000, rows_total=456, data_hash=None)
    assert info.symbol == "SOLUSDT" and info.tf == "15m"
    assert info.latest_ts_ms == 1700000100000
    # sidecar существует
    p = _sidecar_path(tmp_path, "SOLUSDT", "15m")
    assert p.exists()


def test_refresh_from_parquet_reads_latest_ts(tmp_path: Path, sample_df: pd.DataFrame):
    # создаём parquet через write_idempotent
    write_idempotent(tmp_path, "TEST", "1m", sample_df)

    info = tail.refresh_from_parquet(tmp_path, "TEST", "1m")
    assert info.symbol == "TEST" and info.tf == "1m"
    # последняя минута из sample_df
    last_ms = int(sample_df.index[-1].value // 1_000_000)
    assert info.latest_ts_ms == last_ms


def test_update_from_df_uses_dataframe_tail(tmp_path: Path, sample_df: pd.DataFrame):
    info = tail.update_from_df(tmp_path, "XRPUSDT", "1h", sample_df, data_hash="abc123")
    last_ms = int(sample_df.index[-1].value // 1_000_000)
    assert info.latest_ts_ms == last_ms and info.data_hash == "abc123"

    # пустой df не перезаписывает sidecar
    info2 = tail.update_from_df(tmp_path, "XRPUSDT", "1h", sample_df.iloc[0:0])
    assert info2.latest_ts_ms == last_ms  # осталось прежним
