import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ohlcv.core.validate import align_and_flag_gaps
from ohlcv.io.parquet_store import write_idempotent
from ohlcv.cli import main as cli_main

MINUTE_MS = 60_000


def _mk_1m(n: int, start_iso: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start_iso), periods=n, freq="1T", tz="UTC")
    o = np.linspace(100.0, 100.0 + n - 1, n)
    c = o + 0.1
    h = np.maximum(o, c)
    l = np.minimum(o, c)
    v = np.ones(n)
    return pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}, index=idx)


def _with_gap(df: pd.DataFrame, start_i: int, end_i: int) -> pd.DataFrame:
    drop_idx = df.index[start_i:end_i]
    return df.drop(drop_idx)


def test_align_and_flag_gaps_rate():
    # 120 минут, удалим 12 минут (10%)
    full = _mk_1m(120)
    sparse = _with_gap(full, 20, 32)  # 12 минут

    aligned, stats = align_and_flag_gaps(sparse)
    total = len(aligned)
    gaps = int(aligned["is_gap"].sum())
    gap_pct = gaps / total * 100.0

    assert total == 120
    assert gaps == 12
    assert abs(gap_pct - 10.0) < 1e-9


@pytest.mark.parametrize("threshold, expected_rc", [(5.0, 2), (15.0, 0)])
def test_cli_report_missing_threshold(tmp_path: Path, threshold: float, expected_rc: int):
    # Подготовка хранилища с пропусками ≈10%
    root = tmp_path
    sym = "TEST"

    full = _mk_1m(120)
    sparse = _with_gap(full, 10, 22)  # 12/120 = 10%

    write_idempotent(root, sym, "1m", sparse)

    start_ms = int(full.index[0].value // 1_000_000)
    end_ms = int(full.index[-1].value // 1_000_000 + MINUTE_MS)

    rc = cli_main([
        "report-missing",
        "--symbol", sym,
        "--store", str(root),
        "--since-ms", str(start_ms),
        "--until-ms", str(end_ms),
        "--fail-gap-pct", str(threshold),
    ])
    assert rc == expected_rc


def test_cli_report_missing_payload(tmp_path: Path, capsys):
    root = tmp_path
    sym = "BTCUSDT"

    full = _mk_1m(60)
    sparse = _with_gap(full, 0, 6)  # 6/60 = 10%
    write_idempotent(root, sym, "1m", sparse)

    start_ms = int(full.index[0].value // 1_000_000)
    end_ms = int(full.index[-1].value // 1_000_000 + MINUTE_MS)

    rc = cli_main([
        "report-missing",
        "--symbol", sym,
        "--store", str(root),
        "--since-ms", str(start_ms),
        "--until-ms", str(end_ms),
    ])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["symbol"] == sym
    assert payload["total_minutes"] == 60
    assert payload["gaps"] == 6
    assert abs(float(payload["gap_pct"]) - 10.0) < 1e-9
