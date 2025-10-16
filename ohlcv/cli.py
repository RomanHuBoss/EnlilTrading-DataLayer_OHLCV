# ohlcv/cli.py — C1/C2 CLI: backfill/update/resample/report + quality-validate
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
from .quality.validator import validate as dq_validate
from .quality.validator import QualityConfig


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


def _hb_printer(sym: str, since: datetime, until: datetime):
    state = {"last_ms": None}
    total_ms = int((until - since).total_seconds() * 1000) or 1

    def _cb(cursor_ms: int, end_ms: int):
        if state["last_ms"] is None or (cursor_ms - state["last_ms"]) >= 6 * 60 * 60 * 1000:
            dt = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc)
            pct = (cursor_ms - int(since.timestamp() * 1000)) / total_ms * 100.0
            _print(f"[{sym}] scanning… up to {dt.strftime('%Y-%m-%d %H:%M')}Z ({pct:5.1f}%)")
            state["last_ms"] = cursor_ms

    return _cb


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
            lt_candidates = [get_launch_time(sym, category=c) for c in ("linear", "inverse")]
            fut_lt = min([lt for lt in lt_candidates if lt], default=None)
            if fut_lt and fut_lt > eff_since:
                eff_since = fut_lt
                _print(f"[{sym}] spot aligned to futures launchTime → {eff_since.isoformat()}")

        total = int((until - eff_since).total_seconds() // 60)
        fetched = 0
        started_at = datetime.now(timezone.utc)
        _print(f"[{sym}] backfill 1m {eff_since.isoformat()} → {until.isoformat()}  total≈{total} bars")

        acc = []
        heartbeat = _hb_printer(sym, eff_since, until)
        for chunk in iter_klines_1m(
            sym,
            eff_since,
            until,
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            category=args.category,
            on_advance=heartbeat,
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
                lt_candidates = [get_launch_time(sym, category=c) for c in ("linear", "inverse")]
                fut_lt = min([lt for lt in lt_candidates if lt], default=None)
                if fut_lt and fut_lt > start:
                    start = fut_lt
                    _print(f"[{sym}] update aligned to futures launchTime → {start.isoformat()}")

        _print(f"[{sym}] update 1m from {start.isoformat()} to {end.isoformat()}")
        heartbeat = _hb_printer(sym, start, end)
        rows_acc: list = []
        for chunk in iter_klines_1m(sym, start, end, category=args.category, on_advance=heartbeat):
            rows_acc.extend(chunk)

        if not rows_acc:
            _print(f"[{sym}] нет новых баров")
            continue

        df = _df_from_rows(rows_acc)
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


def cmd_quality_validate(args):
    data_root = _data_root(args.data_root)
    symbols = args.symbols.split(",")
    tf = args.tf
    _print(f"[cfg] data_root={str(data_root)}; tf={tf}; write={args.write}; issues={args.issues}")

    cfg = QualityConfig(
        missing_fill_threshold=args.miss_fill_threshold,
        spike_window=args.spike_window,
        spike_k=args.spike_k,
        flat_streak_threshold=args.flat_streak,
    )

    for sym in symbols:
        src = parquet_path(data_root, sym, tf)
        if not src.exists():
            _print(f"[{sym}] файл не найден: {src}")
            continue
        _print(f"[{sym}] quality-validate {tf} ← {src}")

        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()

        clean, issues = dq_validate(df, tf=tf, symbol=sym, repair=not args.no_repair, config=cfg)

        if args.issues:
            out_issues = Path(args.issues).expanduser().resolve()
            out_issues.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if out_issues.exists() else "w"
            header = not out_issues.exists()
            issues.assign(symbol=sym, tf=tf).to_csv(out_issues, mode=mode, header=header, index=False)
            _print(f"[{sym}] issues → {out_issues}")

        if args.write:
            out_path = write_idempotent(data_root, sym, tf, clean)
            _print(f"[{sym}] clean {tf} → {out_path.resolve()}")

        total = len(df)
        total_clean = len(clean)
        n_issues = len(issues)
        _print(f"[{sym}] summary: rows_in={total} rows_out={total_clean} issues={n_issues}")


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
    b.add_argument("--spot-align-futures", action="store_true", help="Для spot: сдвигать окно к дате запуска фьючерса")
    b.set_defaults(func=cmd_backfill)

    u = sub.add_parser("update", help="Обновление хвоста 1m до текущего времени - 1 бар")
    u.add_argument("--symbols", required=True)
    u.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    u.add_argument("--category", required=False, choices=["spot", "linear", "inverse"], default="spot")
    u.add_argument("--spot-align-futures", action="store_true", help="Для spot: сдвигать старт к дате запуска фьючерса")
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("resample", help="Ресемплинг из 1m в 5m/15m/1h")
    r.add_argument("--symbols", required=True)
    r.add_argument("--from-tf", required=True, choices=["1m"])
    r.add_argument("--to-tf", required=True, choices=["5m", "15m", "1h"])
    r.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    r.set_defaults(func=cmd_resample)

    q = sub.add_parser("quality-validate", help="C2 DataQuality: валидация и санитайз Parquet")
    q.add_argument("--symbols", required=True, help="CSV тикеров")
    q.add_argument("--tf", required=True, choices=["1m", "5m", "15m", "1h"])
    q.add_argument("--data-root", required=False, help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT")
    q.add_argument("--write", action="store_true", help="Перезаписать очищенный файл Parquet")
    q.add_argument("--issues", required=False, help="Путь для CSV-журнала проблем; если файл существует — дописывать")
    q.add_argument("--no-repair", action="store_true", help="Только диагностировать, без автоправок")
    q.add_argument("--miss-fill-threshold", type=float, default=0.0001, help="Порог заполнения пропусков для 1m")
    q.add_argument("--spike-window", type=int, default=200, help="Окно MAD-детектора всплесков")
    q.add_argument("--spike-k", type=float, default=12.0, help="Порог MAD-критерия")
    q.add_argument("--flat-streak", type=int, default=300, help="Длина серии нулевого объёма")
    q.set_defaults(func=cmd_quality_validate)

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
