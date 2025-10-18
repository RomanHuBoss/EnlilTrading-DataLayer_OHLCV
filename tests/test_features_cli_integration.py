# tests/test_features_cli_integration.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ohlcv.features.cli import main as cli_main


def _mk_df(n: int = 128, start_ts: int = 1_700_000_000_000, step_ms: int = 300_000) -> pd.DataFrame:
    ts = np.arange(start_ts, start_ts + n * step_ms, step_ms, dtype=np.int64)
    base = 100.0 + np.linspace(0, 1.2, n)
    wave = 0.2 * np.sin(np.linspace(0, 6.28, n))
    close = base + wave
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = 1_000.0 + (np.arange(n) % 10) * 10.0
    return pd.DataFrame(
        {
            "timestamp_ms": ts,
            "start_time_iso": pd.to_datetime(ts, unit="ms", utc=True).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.mark.parametrize("ext", [".csv"])  # CSV без внешних зависимостей
def test_cli_build_csv_roundtrip(tmp_path: Path, ext: str):
    inp = tmp_path / f"input{ext}"
    out = tmp_path / "features.csv"

    df = _mk_df()
    if ext == ".csv":
        df.to_csv(inp, index=False)
    else:  # pragma: no cover
        df.to_parquet(inp, index=False)

    rc = cli_main(["build", str(inp), str(out), "--symbol", "TEST", "--tf", "5m"])
    assert rc == 0

    got = pd.read_csv(out)
    # минимальный набор столбцов
    must = [
        "timestamp_ms",
        "symbol",
        "tf",
        "f_rv_20",
        "f_rv_60",
        "f_adx_14",
        "f_pdi_14",
        "f_mdi_14",
        "f_donch_width_pct_20",
        "f_donch_width_pct_55",
        "f_vwap_roll_96",
        "f_vwap_dev_pct_96",
        "f_valid_from",
        "f_build_version",
    ]
    missing = [c for c in must if c not in got.columns]
    assert not missing, f"missing: {missing}"


def test_cli_build_parquet_output(tmp_path: Path):
    pyarrow = pytest.importorskip("pyarrow")  # noqa: F841
    inp = tmp_path / "input.csv"
    out = tmp_path / "features.parquet"

    _mk_df().to_csv(inp, index=False)

    rc = cli_main(["build", str(inp), str(out)])
    assert rc == 0

    got = pd.read_parquet(out)
    assert "f_rv_20" in got.columns and "f_build_version" in got.columns


def test_cli_with_yaml_config(tmp_path: Path):
    yaml = pytest.importorskip("yaml")  # noqa: F841
    inp = tmp_path / "input.csv"
    out = tmp_path / "features.csv"
    cfg = tmp_path / "features.yaml"

    _mk_df().to_csv(inp, index=False)

    cfg.write_text(
        """
rv_windows: [20]
vwap_roll_window: 5
""".strip(),
        encoding="utf-8",
    )

    rc = cli_main(["build", str(inp), str(out), "--config", str(cfg)])
    assert rc == 0

    got = pd.read_csv(out)
    assert "f_vwap_roll_5" in got.columns
    assert "f_rv_20" in got.columns
