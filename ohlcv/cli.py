# ohlcv/cli.py — CLI с прогрессом, репарацией 1m, корнем данных (--data-root),
# категорией рынка (--category) и выравниванием спота по дате запуска фьючерса (--spot-align-futures)
import os
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from .api.bybit import fetch_klines_1m, iter_klines_1m, get_launch_time
from .core.resample import resample_ohlcv
from .core.validate import validate_1m_index, ensure_missing_threshold, fill_1m_gaps
from .io.parquet_store import write_idempotent, parquet_path
from .utils.timeframes import tf_minutes


def _data_root(arg: str | None) -> Path:
    base = Path(arg) if arg else Path(os.environ.get("C1_DATA_ROOT", Path.cwd() / "data"))
    return base.expanduser().resolve()


def _df_from_rows(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    cols = ["o", "h", "l", "c", "v"] + (["t"] if "t" in df.columns else [])
    return df[cols]


def _print(msg: str):
    print(msg, flush=True)


def _print_progress(sym: str, fetched: int, total: int, last_ts: datetime, started_at: datetime):
    pct = (fetched / total * 100.0) if total > 0 else 0.0
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    speed = fetched / elapsed if elapsed > 0 else 0.0
    eta_sec = max(0.0, (total - fetched) / speed) if speed > 0 else 0.0
    eta_min, eta_rem = int(eta_sec // 60), int(eta_sec % 60)
    _print(
        f"[{sym}] {fetched}/{total} ({pct:5.1f}%)  up to {last_ts.strftime('%Y-%m-%d %H:%M')}Z  "
        f"speed {speed:5.1f} bars/s  ETA {eta_min:02d}:{eta_rem:02d}"
    )


def _min_launch_time(symbol: str):
    lt_lin = get_launch_time(symbol, category="linear")
    lt_inv = get_launch_time(symbol, category="inverse")
    if lt_lin and lt_inv:
        return min(lt_lin, lt_inv)
    return lt_lin or lt_inv


def cmd_backfill(args):
    symbols = args.symbols.split(",")
    since = datetime.fromisoformat(args.since)
    since = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since.astimezone(timezone.utc)
    until = datetime.fromisoformat(args.until).astimezone(timezone.utc) if args.until else datetime.now(timezone.utc)
    since, until = since.replace(second=0, microsecond=0), until.replace(second=0, microsecond=0)

    data_root = _data_root(args.data_root)
    _print(f"[cfg] data_root={str(data_root)}; category={args.category}; spot_align_futures={args.spot_align_futures}")

    for sym in symbols:
        eff_since = since
        if args.category == "spot" and args.spot_align_futures:
            fut_lt = _min_launch_time(sym)
            if fut_lt and fut_lt > eff_since:
                eff_since = fut_lt
                _print(f"[{sym}] spot aligned to futures launchTime → {eff_since.isoformat()}")

        total = int((until - eff_since).total_seconds() // 60)
        fetched = 0
        started_at = datetime.now(timezone.utc)
        _print(f"[{sym}] backfill 1m {eff_since.isoformat()} → {until.isoformat()}  total≈{total} bars")

        acc = []
        for chunk in iter_klines_1m(
            sym,
            eff_since,
            until,
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            category=args.category,
        ):
            acc.extend(chunk)
            fetched += len(chunk)
            _print_progress(sym, fetched, total, datetime.fromisoformat(chunk[-1]["ts"]), started_at)

        _print(f"[{sym}] fetched={fetched} bars. Репарация/валидация…")
        df = _df_from_rows(acc)
        if df.empty:
            _print(f"[{sym}] нет данных в заданном окне")
            continue

        df, n_filled = fill_1m_gaps(df)
        _print(f"[{sym}] заполнено синтетикой минут: {n_filled}")

        validate_1m_index(df)
        ensure_missing_threshold(df, threshold=1e-9)

        _print(f"[{sym}] запись в Parquet…")
        out_path = write_idempotent(data_root, sym, "1m", df)
        _print(f"[{sym}] OK → {out_path.resolve()}")


def cmd_update(args):
    symbols = args.symbols.split(",")
    now = datetime.now(timezone.utc)
    end = now.replace(second=0, microsecond=0)

    data_root = _data_root(args.data_root)
    _print(f"[cfg] data_root={str(data_root)}; category={args.category}; spot_align_futures={args.spot_align_futures}")

    for sym in symbols:
        path = parquet_path(data_root, sym, "1m")
        if path.exists():
            exist = pd.read_parquet(path)
            exist["ts"] = pd.to_datetime(exist["ts"], utc=True)
            last_ts = exist["ts"].max()
            start = last_ts + pd.Timedelta(minutes=1)
        else:
            start = end - pd.Timedelta(days=7)
            if args.category == "spot" and args.spot_align_futures:
                fut_lt = _min_launch_time(sym)
                if fut_lt and fut_lt > start:
                    start = fut_lt
                    _print(f"[{sym}] update start aligned to futures launchTime → {start.isoformat()}")

        _print(f"[{sym}] update 1m from {start.isoformat()} to {end.isoformat()}")
        rows = fetch_klines_1m(sym, start, end, category=args.category)
        if not rows:
            _print(f"[{sym}] нет новых баров")
            continue

        df = _df_from_rows(rows)
        df, n_filled = fill_1m_gaps(df)
        _print(f"[{sym}] заполнено синтетикой минут: {n_filled}")
        validate_1m_index(df)

        out_path = write_idempotent(data_root, sym, "1m", df)
        _print(f"[{sym}] OK → {out_path.resolve()}")


def cmd_resample(args):
    symbols = args.symbols.split(",")
    data_root = _data_root(args.data_root)
    _print(f"[cfg] data_root={str(data_root)}")

    for sym in symbols:
        src = parquet_path(data_root, sym, args.from_tf)
        if not src.exists():
            raise SystemExit(f"Нет исходного файла: {src}")
        _print(f"[{sym}] resample {args.from_tf} → {args.to_tf}")

        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()

        out = resample_ohlcv(df, args.to_tf)
        out_path = write_idempotent(data_root, sym, args.to_tf, out)
        _print(f"[{sym}] OK → {out_path.resolve()}")


def cmd_report_missing(args):
    symbols = args.symbols.split(",")
    data_root = _data_root(args.data_root)
    _print(f"[cfg] data_root={str(data_root)}")

    out_rows = []
    for sym in symbols:
        src = parquet_path(data_root, sym, args.tf)
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
            "missing_rate": 1.0 - (len(df) / len(full)),
        })

    out_path = Path(args.out).expanduser().resolve()
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    _print(f"report → {out_path}")


def main():
    p = argparse.ArgumentParser(prog="c1-ohlcv")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="Догрузка истории 1m (с прогрессом и репарацией)")
    b.add_argument("--symbols", required=True, help="CSV тикеров, пример: BTCUSDT,ETHUSDT")
    b.add_argument("--since", required=True, help="ISO дата начала (UTC)")
    b.add_argument("--until", required=False, help="ISO дата окончания (UTC)")
    b.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    b.add_argument("--category", required=False, choices=["spot", "linear", "inverse"], default="spot")
    b.add_argument("--spot-align-futures", action="store_true", help="Для spot: сдвигать since к дате запуска фьючерса (min(linear,inverse))")
    b.set_defaults(func=cmd_backfill)

    u = sub.add_parser("update", help="Обновление хвоста 1m до текущего времени - 1 бар")
    u.add_argument("--symbols", required=True)
    u.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    u.add_argument("--category", required=False, choices=["spot", "linear", "inverse"], default="spot")
    u.add_argument("--spot-align-futures", action="store_true", help="Для spot: сдвигать старт первой догрузки к дате запуска фьючерса")
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("resample", help="Ресемплинг из 1m в 5m/15m/1h")
    r.add_argument("--symbols", required=True)
    r.add_argument("--from-tf", required=True, choices=["1m"])
    r.add_argument("--to-tf", required=True, choices=["5m", "15m", "1h"])
    r.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    r.set_defaults(func=cmd_resample)

    rep = sub.add_parser("report-missing", help="Отчёт о пропусках")
    rep.add_argument("--symbols", required=True)
    rep.add_argument("--tf", required=True, choices=["1m", "5m", "15m", "1h"])
    rep.add_argument("--out", required=True)
    rep.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    rep.set_defaults(func=cmd_report_missing)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
