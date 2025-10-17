from pathlib import Path

import numpy as np
import pandas as pd

from ohlcv.features.cli import main as features_main


def test_cli_build_parquet(tmp_path: Path) -> None:
    # Синтетический вход — 180 баров 5m (15 часов)
    n = 180
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    base = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {
            "timestamp_ms": (ts.astype("int64") // 10**6).astype("int64"),
            "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": np.linspace(1.0, 2.0, n),
        }
    )

    inp = tmp_path / "in.csv"
    out = tmp_path / "out.parquet"
    df.to_csv(inp, index=False)

    rc = features_main(
        [
            "build",
            "--input",
            str(inp),
            "--symbol",
            "BTCUSDT",
            "--tf",
            "5m",
            "--output",
            str(out),
            "--build-version",
            "TEST",
            "--strict",
        ]
    )
    assert rc == 0

    res = pd.read_parquet(out)
    assert "f_build_version" in res.columns
    assert res["f_build_version"].iloc[-1] == "TEST"
    # эвристика: фич должно быть достаточно
    assert res.filter(like="f_").shape[1] >= 20
