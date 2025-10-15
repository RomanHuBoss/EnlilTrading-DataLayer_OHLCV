# CLI для бэкапа, обновления и отчётов.
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd

from .api.bybit import fetch_klines_1m
from .core.resample import resample_ohlcv
from .core.validate import validate_1m_index, ensure_missing_threshold
from .io.parquet_store import write_idempotent, parquet_path
from .utils.timeframes import tf_minutes

DATA_ROOT = Path(os.environ.get("C1_DATA_ROOT", "/data"))

def _df_from_rows(rows):
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    df = df[["o","h","l","c","v"] + ([ "t" ] if "t" in df.columns else [])]
    return df

def cmd_backfill(args):
    symbols = args.symbols.split(",")
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    until = datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc) if args.until else datetime.now(timezone.utc)
    for sym in symbols:
        rows = fetch_klines_1m(sym, since, until, api_key=os.getenv("BYBIT_API_KEY"), api_secret=os.getenv("BYBIT_API_SECRET"))
        df = _df_from_rows(rows)
        validate_1m_index(df)
        ensure_missing_threshold(df, threshold=0.0001)
        write_idempotent(DATA_ROOT, sym, "1m", df)

def cmd_update(args):
    symbols = args.symbols.split(",")
    now = datetime.now(timezone.utc)
    end = now.replace(second=0, microsecond=0)  # до ближайшей минуты
    for sym in symbols:
        path = parquet_path(DATA_ROOT, sym, "1m")
        if path.exists():
            exist = pd.read_parquet(path)
            last_ts = pd.to_datetime(exist["ts"], utc=True).max()
            start = last_ts + pd.Timedelta(minutes=1)
        else:
            # Если нет истории — замените на желаемую глубину
            start = end - pd.Timedelta(days=7)
        rows = fetch_klines_1m(sym, start, end)
        if not rows:
            continue
        df = _df_from_rows(rows)
        validate_1m_index(df)
        write_idempotent(DATA_ROOT, sym, "1m", df)

def cmd_resample(args):
    symbols = args.symbols.split(",")
    for sym in symbols:
        src = parquet_path(DATA_ROOT, sym, args.from_tf)
        if not src.exists():
            raise SystemExit(f"Нет исходного файла: {src}")
        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        out = resample_ohlcv(df, args.to_tf)
        write_idempotent(DATA_ROOT, sym, args.to_tf, out)

def cmd_report_missing(args):
    symbols = args.symbols.split(",")
    out_rows = []
    for sym in symbols:
        src = parquet_path(DATA_ROOT, sym, args.tf)
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
        freq = f"{tf_minutes(args.tf)}T"
        full = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz="UTC")
        missing = full.difference(df.index)
        out_rows.append({
            "symbol": sym,
            "tf": args.tf,
            "bars": len(full),
            "present": len(df),
            "missing": len(missing),
            "missing_rate": 1.0 - (len(df)/len(full))
        })
    pd.DataFrame(out_rows).to_csv(args.out, index=False)

def main():
    p = argparse.ArgumentParser(prog="c1-ohlcv")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="Догрузка истории 1m")
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
    r.add_argument("--to-tf", required=True, choices=["5m","15m","1h"])
    r.set_defaults(func=cmd_resample)

    rep = sub.add_parser("report-missing", help="Отчёт о пропусках")
    rep.add_argument("--symbols", required=True)
    rep.add_argument("--tf", required=True, choices=["1m","5m","15m","1h"])
    rep.add_argument("--out", required=True)
    rep.set_defaults(func=cmd_report_missing)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
