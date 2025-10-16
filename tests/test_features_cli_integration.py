import numpy as np
import pandas as pd
from pathlib import Path

from ohlcv.features.cli import main as features_main


def test_cli_build_parquet(tmp_path: Path):
    # make tiny CSV input
    n = 180
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp_ms": (ts.view("i8") // 10**6).astype("int64"),
            "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "o": np.linspace(1.0, 2.0, n),  # internal schema to test auto-rename
            "h": np.linspace(1.0, 2.0, n) + 0.1,
            "l": np.linspace(1.0, 2.0, n) - 0.1,
            "c": np.linspace(1.0, 2.0, n),
            "v": np.ones(n),
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
    assert "f_build_version" in res
    assert res["f_build_version"].iloc[-1] == "TEST"
    assert res.filter(like="f_").shape[1] >= 20
