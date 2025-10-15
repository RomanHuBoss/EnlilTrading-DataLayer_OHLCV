# ohlcv/cli.py
import os
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from .api.bybit import fetch_klines_1m, iter_klines_1m
from .core.resample import resample_ohlcv
from .core.validate import validate_1m_index, ensure_missing_threshold
from .io.parquet_store import write_idempotent, parquet_path
from .utils.timeframes import tf_minutes

DATA_ROOT = Path(os.environ.get("C1_DATA_ROOT", "/data"))


def _df_from_rows(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    cols = ["o", "h", "l", "c", "v"]
    if "t" in df.columns:
        cols.append("t")
    return df[cols]


def _print(msg: str):
    print(msg, flush=True)


def _print_progress(sym: str, fetched: int, total: int, last_ts: datetime, started_at: datetime):
    pct = (fetched / total * 100.0) if total > 0 else 0.0
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    speed = fetched / elapsed if elapsed > 0 else 0.0  # bars/sec
    eta_sec = max(0.0, (total - fetched) / speed) if speed > 0 else 0.0
    eta_min = int(eta_sec // 60)
    eta_rem = int(eta_sec % 60)
    _print(f"[{sym}] {fetched}/{total} ({pct:5.1f}%)  up to {last_ts.strftime('%Y-%m-%d %H:%M')}Z  speed {speed:5.1f} bars/s  ETA {eta_min:02d}:{eta_rem:02d}")


def cmd_backfill(args):
    symbols = args.symbols.split(",")
    since = datetime.fromisoformat(args.since)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    else:
        since = since.astimezone(timezone.utc)
    until = (
        datetime.fromisoformat(args.until).astimezone(timezone.utc)
        if args.until
        else datetime.now(timezone.utc)
    )
    # выравнивание по минутам
    since = since.replace(second=0, microsecond=0)
    until = until.replace(second=0, microsecond=0)

    for sym in symbols:
        total = int((until - since).total_seconds() // 60)
        fetched = 0
        started_at = datetime.now(timezone.utc)
        _print(f"[{sym}] backfill 1m {since.isoformat()} → {until.isoformat()}  total≈{total} bars")

        acc = []
        for chunk in iter_klines_1m(
            sym,
            since,
            until,
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
        ):
            acc.extend(chunk)
            fetched += len(chunk)
            last_ts = datetime.fromisoformat(chunk[-1]["ts"])
            _print_progress(sym, fetched, total, last_ts, started_at)

        _print(f"[{sym}] fetched={fetched} bars. Валидация…")
        df = _df_from_rows(acc)
        if df.empty:
            _print(f"[{sym}] нет данных в заданном окне")
            continue
        validate_1m_index(df)
        ensure_missing_threshold(df, threshold=0.0001)
        _print(f"[{sym}] запись в Parquet…")
        out_path = write_idempotent(DATA_ROOT, sym, "1m", df)
        _print(f"[{sym}] OK → {out_path}")


def cmd_update(args):
    symbols = args.symbols.split(",")
    now = datetime.now(timezone.utc)
    end = now.replace(second=0, microsecond=0)
    for sym in symbols:
        path = parquet_path(DATA_ROOT, sym, "1m")
        if path.exists():
            exist = pd.read_parquet(path)
            exist["ts"] = pd.to_datetime(exist["ts"], utc=True)
            last_ts = exist["ts"].max()
            start = last_ts + pd.Timedelta(minutes=1)
        else:
            start = end - pd.Timedelta(days=7)
        _print(f"[{sym}] update 1m from {start.isoformat()} to {end.isoformat()}")
        rows = fetch_klines_1m(sym, start, end)
        if not rows:
            _print(f"[{sym}] нет новых баров")
            continue
        df = _df_from_rows(rows)
        validate_1m_index(df)
        out_path = write_idempotent(DATA_ROOT, sym, "1m", df)
        _print(f"[{sym}] OK → {out_path}")


def cmd_resample(args):
    symbols = args.symbols.split(",")
    for sym in symbols:
        src = parquet_path(DATA_ROOT, sym, args.from_tf)
        if not src.exists():
            raise SystemExit(f"Нет исходного файла: {src}")
        _print(f"[{sym}] resample {args.from_tf} → {args.to_tf}")
        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        out = resample_ohlcv(df, args.to_tf)
        out_path = write_idempotent(DATA_ROOT, sym, args.to_tf, out)
        _print(f"[{sym}] OK → {out_path}")


def cmd_report_missing(args):
    symbols = args.symbols.split(",")
    out_rows = []
    for sym in symbols:
        src = parquet_path(DATA_ROOT, sym, args.tf)
        if not src.exists():
            _print(f"[{sym}] файл не найден: {src}")
            continue
        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        freq = f"{tf_minutes(args.tf)}min"
        full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
        missing = full.difference(df.index)
        out_rows.append({
            "symbol": sym,
            "tf": args.tf,
            "bars": len(full),
            "present": len(df),
            "missing": len(missing),
            "missing_rate": 1.0 - (len(df) / len(full))
        })
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    _print(f"report → {args.out}")


def main():
    p = argparse.ArgumentParser(prog="c1-ohlcv")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="Догрузка истории 1m (с прогрессом)")
    b.add_argument("--symbols", required=True, help="CSV тикеров, пример: BTCUSDT,ETHUSDT")
    b.add_argument("--since", required=True, help="ISO дата начала (UTC)")
    b.add_argument("--until", required=False, help="ISO дата окончания (UTC)")
    b.set_defaults(func=cmd_backfill)

    u = sub.add_parser("update", help="Обновление хвоста 1m до текущего времени - 1 бар")
    u.add_argument("--symbols", required=True)
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("resample", help="Ресемплинг из 1m в 5m/15m/1h")
    r.add_argument("--symbols", required=True)
    r.add_argument("--from-tf", required=True, choices=["1m"])
    r.add_argument("--to-tf", required=True, choices=["5m", "15m", "1h"])
    r.set_defaults(func=cmd_resample)

    rep = sub.add_parser("report-missing", help="Отчёт о пропусках")
    rep.add_argument("--symbols", required=True)
    rep.add_argument("--tf", required=True, choices=["1m", "5m", "15m", "1h"])
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=cmd_report_missing)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
