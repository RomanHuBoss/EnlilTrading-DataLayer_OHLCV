from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ohlcv.cli import main as cli_main
from ohlcv.api.bybit import FetchStats

MINUTE_MS = 60_000


# -------------------------
# Вспомогательные генераторы
# -------------------------

def _mk_minutes(start_ms: int, end_ms: int) -> pd.DatetimeIndex:
    if end_ms <= start_ms:
        return pd.DatetimeIndex([], name="ts", tz="UTC")
    s = pd.to_datetime(start_ms, unit="ms", utc=True)
    e = pd.to_datetime(end_ms - MINUTE_MS, unit="ms", utc=True)
    # включительная правая граница: добавим +1 минуту и возьмём inclusive="both"
    rng = pd.date_range(s, e, freq="1T", inclusive="both")
    return pd.DatetimeIndex(rng, name="ts")


def _fake_df(start_ms: int, end_ms: int) -> pd.DataFrame:
    idx = _mk_minutes(start_ms, end_ms)
    n = len(idx)
    if n == 0:
        return pd.DataFrame(index=idx)
    # детерминированные значения на основе порядкового номера бара
    i = np.arange(n, dtype=float)
    o = 100.0 + i
    c = o + 0.1
    h = np.maximum(o, c) + 0.2
    l = np.minimum(o, c) - 0.2
    v = np.ones(n)
    df = pd.DataFrame({"o": o, "h": h, "l": l, "c": c, "v": v}, index=idx)
    df["t"] = (o + h + l + c) / 4.0
    return df


# -------------------------
# Патч HTTP‑клиента Bybit
# -------------------------

@pytest.fixture()
def patch_bybit(monkeypatch):
    from ohlcv.api import bybit as bybit_mod

    def fake_fetch(self, symbol, start_ms, end_ms, *_, **__):  # noqa: D401
        df = _fake_df(int(start_ms), int(end_ms))
        stats = FetchStats(
            symbol=symbol,
            category="linear",
            rows=int(len(df)),
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            requests=1,
            retries=0,
            rate_limited=0,
        )
        return df, stats

    monkeypatch.setattr(bybit_mod.BybitClient, "fetch_ohlcv_1m", fake_fetch)
    return True


# -------------------------
# Тесты CLI
# -------------------------


def test_backfill_creates_1m_parquet_and_json(tmp_path: Path, capsys, patch_bybit):
    root = tmp_path
    sym = "TEST"
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 10 * MINUTE_MS

    rc = cli_main([
        "backfill",
        "--symbol", sym,
        "--store", str(root),
        "--since-ms", str(start_ms),
        "--until-ms", str(end_ms),
    ])
    assert rc == 0

    out = json.loads(capsys.readouterr().out.strip())
    assert out["symbol"] == sym and out["tf"] == "1m"
    path = Path(out["path"])  # <root>/<sym>/1m.parquet
    assert path.exists()

    got = pd.read_parquet(path)
    # ожидаем 10 рядов (полуинтервал), ts монотонный
    assert len(got) == 10
    ts = pd.to_datetime(got["ts"], utc=True)
    assert ts.is_monotonic_increasing and not ts.duplicated().any()


def test_update_appends_without_duplicates(tmp_path: Path, patch_bybit):
    root = tmp_path
    sym = "TEST"

    s1 = 1_700_000_000_000
    e1 = s1 + 10 * MINUTE_MS  # 10 минут
    rc1 = cli_main(["backfill", "--symbol", sym, "--store", str(root), "--since-ms", str(s1), "--until-ms", str(e1)])
    assert rc1 == 0

    s2 = s1 + 8 * MINUTE_MS
    e2 = s1 + 15 * MINUTE_MS  # пересечение 8..9 и ещё 5 новых минут
    rc2 = cli_main(["update", "--symbol", sym, "--store", str(root), "--since-ms", str(s2), "--until-ms", str(e2)])
    assert rc2 == 0

    path = root / sym / "1m.parquet"
    got = pd.read_parquet(path)
    ts = pd.to_datetime(got["ts"], utc=True)
    # ожидаем 15 уникальных минут: 0..14
    assert len(got) == 15
    assert ts.is_monotonic_increasing and not ts.duplicated().any()


def test_resample_then_read_export_csv(tmp_path: Path, patch_bybit):
    root = tmp_path
    sym = "TEST"
    s = 1_700_000_000_000

    # backfill 15 минут сплошняком → 3 бара 5m
    rc1 = cli_main(["backfill", "--symbol", sym, "--store", str(root), "--since-ms", str(s), "--until-ms", str(s + 15 * MINUTE_MS)])
    assert rc1 == 0

    rc2 = cli_main(["resample", "--symbol", sym, "--store", str(root), "--dst-tf", "5m"])
    assert rc2 == 0

    # файл 5m создан
    p5 = root / sym / "5m.parquet"
    assert p5.exists()

    # read → csv
    csv_out = root / "slice.csv"
    rc3 = cli_main(["read", "--symbol", sym, "--store", str(root), "--tf", "5m", "--output", str(csv_out)])
    assert rc3 == 0 and csv_out.exists()

    got = pd.read_csv(csv_out)
    # ожидаем ≥3 строк (3 полных окна)
    assert len(got) >= 3


def test_report_missing_json_and_threshold(tmp_path: Path):
    # Подготовим хранилище с пропусками вручную (без обращения к API)
    from ohlcv.io.parquet_store import write_idempotent

    root = tmp_path
    sym = "MISS"

    # создаём 60 минут и удаляем 6 → 10% пропусков
    idx = pd.date_range("2024-01-01", periods=60, freq="1T", tz="UTC")
    o = np.arange(60) + 100.0
    df = pd.DataFrame({"o": o, "h": o + 0.2, "l": o - 0.2, "c": o + 0.1, "v": 1.0}, index=idx)
    df = df.drop(idx[:6])  # удалим первые 6 минут

    write_idempotent(root, sym, "1m", df)

    start_ms = int(idx[0].value // 1_000_000)
    end_ms = int((idx[-1] + pd.Timedelta(minutes=1)).value // 1_000_000)

    # Без порога
    from ohlcv.cli import main as cli

    rc = cli(["report-missing", "--symbol", sym, "--store", str(root), "--since-ms", str(start_ms), "--until-ms", str(end_ms)])
    assert rc == 0

    # С порогом 5% → код 2
    rc2 = cli(["report-missing", "--symbol", sym, "--store", str(root), "--since-ms", str(start_ms), "--until-ms", str(end_ms), "--fail-gap-pct", "5.0"])
    assert rc2 == 2
