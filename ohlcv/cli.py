"""CLI C1/C2: backfill/update/resample/report + quality-validate."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from .api.bybit import KlineRow, get_launch_time, iter_klines_1m
from .core.resample import resample_ohlcv
from .core.validate import ensure_missing_threshold, fill_1m_gaps, validate_1m_index
from .io.parquet_store import parquet_path, write_idempotent
from .quality.validator import QualityConfig
from .quality.validator import validate as dq_validate
from .utils.timeframes import tf_minutes

# -------------------- общие утилиты --------------------


def _data_root(arg: str | None) -> Path:
    base = Path(arg) if arg else Path(os.environ.get("C1_DATA_ROOT", Path.cwd() / "data"))
    return base.expanduser().resolve()


def _df_from_rows(rows: Sequence[KlineRow]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["o", "h", "l", "c", "v"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    cols = ["o", "h", "l", "c", "v"] + (["t"] if "t" in df.columns else [])
    return df[cols]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _print_progress(
    sym: str,
    fetched: int,
    total: int,
    last_ts: datetime,
    started_at: datetime,
) -> None:
    pct = (fetched / total * 100.0) if total > 0 else 0.0
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    speed = fetched / elapsed if elapsed > 0 else 0.0
    eta_sec = max(0.0, (total - fetched) / speed) if speed > 0 else 0.0
    eta_min, eta_rem = int(eta_sec // 60), int(eta_sec % 60)
    _print(
        f"[{sym}] {fetched}/{total} ({pct:5.1f}%)  up to "
        f"{last_ts.strftime('%Y-%m-%d %H:%M')}Z  "
        f"speed {speed:5.1f} bars/s  ETA {eta_min:02d}:{eta_rem:02d}"
    )


def _hb_printer(sym: str, since: datetime, until: datetime) -> Callable[[int, int], None]:
    state: Dict[str, int | None] = {"last_ms": None}
    total_ms = int((until - since).total_seconds() * 1000) or 1

    def _cb(cursor_ms: int, end_ms: int) -> None:  # noqa: ARG001
        if state["last_ms"] is None or (cursor_ms - int(state["last_ms"])) >= 6 * 60 * 60 * 1000:
            dt = datetime.fromtimestamp(cursor_ms / 1000, tz=timezone.utc)
            pct = (cursor_ms - int(since.timestamp() * 1000)) / total_ms * 100.0
            _print(f"[{sym}] scanning… up to {dt.strftime('%Y-%m-%d %H:%M')}Z ({pct:5.1f}%)")
            state["last_ms"] = cursor_ms

    return _cb


def _align_since_with_launch(
    sym: str,
    eff_since: datetime,
    category: str,
    spot_align_futures: bool,
) -> datetime:
    """Кламп начала окна к launchTime категории.
    Для spot — опционально к min(launchTime linear, inverse).
    """
    lt = get_launch_time(sym, category=category)
    if lt and lt > eff_since:
        eff_since = lt
        _print(f"[{sym}] aligned to {category} launchTime → {eff_since.isoformat()}")
    if category == "spot" and spot_align_futures:
        fut_candidates = [get_launch_time(sym, category=c) for c in ("linear", "inverse")]
        fut_lt = min([x for x in fut_candidates if x is not None], default=None)
        if fut_lt and fut_lt > eff_since:
            eff_since = fut_lt
            _print(f"[{sym}] spot aligned to futures launchTime → {eff_since.isoformat()}")
    return eff_since


# -------------------- команды --------------------


def cmd_backfill(args: argparse.Namespace) -> None:
    symbols = args.symbols.split(",")
    since = datetime.fromisoformat(args.since)
    since = (
        since.replace(tzinfo=timezone.utc)
        if since.tzinfo is None
        else since.astimezone(timezone.utc)
    )
    until = (
        datetime.fromisoformat(args.until).astimezone(timezone.utc)
        if getattr(args, "until", None)
        else datetime.now(timezone.utc)
    )
    since, until = since.replace(second=0, microsecond=0), until.replace(second=0, microsecond=0)

    data_root = _data_root(getattr(args, "data_root", None))
    _print(
        f"[cfg] data_root={str(data_root)}; category={args.category}; "
        f"spot_align_futures={args.spot_align_futures}"
    )

    for sym in symbols:
        eff_since = _align_since_with_launch(sym, since, args.category, args.spot_align_futures)

        total = int((until - eff_since).total_seconds() // 60)
        fetched = 0
        started_at = datetime.now(timezone.utc)
        _print(
            f"[{sym}] backfill 1m {eff_since.isoformat()} → {until.isoformat()}  "
            f"total≈{total} bars"
        )

        acc: List[KlineRow] = []
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
            _print_progress(
                sym,
                fetched,
                total,
                datetime.fromisoformat(chunk[-1]["ts"]),
                started_at,
            )

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


def cmd_update(args: argparse.Namespace) -> None:
    symbols = args.symbols.split(",")
    now = datetime.now(timezone.utc)
    end = now.replace(second=0, microsecond=0)

    data_root = _data_root(getattr(args, "data_root", None))
    _print(
        f"[cfg] data_root={str(data_root)}; category={args.category}; "
        f"spot_align_futures={args.spot_align_futures}"
    )

    for sym in symbols:
        path = parquet_path(data_root, sym, "1m")
        if path.exists():
            exist = pd.read_parquet(path)
            exist["ts"] = pd.to_datetime(exist["ts"], utc=True)
            last_ts = exist["ts"].max()
            start = last_ts + pd.Timedelta(minutes=1)
        else:
            start = end - pd.Timedelta(days=7)
        start = _align_since_with_launch(sym, start, args.category, args.spot_align_futures)

        _print(f"[{sym}] update 1m from {start.isoformat()} to {end.isoformat()}")
        heartbeat = _hb_printer(sym, start, end)
        rows_acc: List[KlineRow] = []
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


def cmd_resample(args: argparse.Namespace) -> None:
    symbols = args.symbols.split(",")
    data_root = _data_root(getattr(args, "data_root", None))
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


def cmd_quality_validate(args: argparse.Namespace) -> None:
    data_root = _data_root(getattr(args, "data_root", None))
    symbols = args.symbols.split(",")
    tf = args.tf
    _print(
        f"[cfg] data_root={str(data_root)}; tf={tf}; write={args.write}; "
        f"issues_csv={args.issues}; issues_parquet={args.issues_parquet}"
    )

    # Только поддержанные параметры конструктора QualityConfig
    cfg = QualityConfig(missing_fill_threshold=args.miss_fill_threshold)

    summary_rows: List[Dict[str, Any]] = []
    all_issues: List[pd.DataFrame] = []

    for sym in symbols:
        src = parquet_path(data_root, sym, tf)
        if not src.exists():
            _print(f"[{sym}] файл не найден: {src}")
            continue
        _print(f"[{sym}] quality-validate {tf} ← {src}")

        df = pd.read_parquet(src)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()

        clean, issues = dq_validate(
            df,
            tf=tf,
            symbol=sym,
            repair=not args.no_repair,
            config=cfg,
        )

        if not issues.empty:
            issues = issues.assign(symbol=sym, tf=tf)
            all_issues.append(issues)

        if args.write:
            out_path = write_idempotent(data_root, sym, tf, clean)
            _print(f"[{sym}] clean {tf} → {out_path.resolve()}")

        total = len(df)
        total_clean = len(clean)
        n_issues = int(0 if issues is None else len(issues))
        _print(f"[{sym}] summary: rows_in={total} rows_out={total_clean} issues={n_issues}")

        summary_rows.append(
            {
                "symbol": sym,
                "tf": tf,
                "rows_in": total,
                "rows_out": total_clean,
                "issues": n_issues,
            }
        )

    if all_issues:
        issues_cat = pd.concat(all_issues, axis=0, ignore_index=True)
        if getattr(args, "issues", None):
            out_issues_csv = Path(args.issues).expanduser().resolve()
            out_issues_csv.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if (out_issues_csv.exists() and not args.truncate) else "w"
            header = not out_issues_csv.exists() or args.truncate
            issues_cat.to_csv(out_issues_csv, index=False, mode=mode, header=header)
            _print(f"[issues] csv → {out_issues_csv}")
        if getattr(args, "issues_parquet", None):
            out_issues_parquet = Path(args.issues_parquet).expanduser().resolve()
            out_issues_parquet.parent.mkdir(parents=True, exist_ok=True)
            issues_cat.to_parquet(out_issues_parquet, index=False)
            _print(f"[issues] parquet → {out_issues_parquet}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if getattr(args, "quality_summary_csv", None):
            out_csv = Path(args.quality_summary_csv).expanduser().resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if (out_csv.exists() and not args.truncate) else "w"
            header = not out_csv.exists() or args.truncate
            summary_df.to_csv(out_csv, index=False, mode=mode, header=header)
            _print(f"[summary] csv → {out_csv}")
        if getattr(args, "quality_summary_json", None):
            out_json = Path(args.quality_summary_json).expanduser().resolve()
            out_json.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "by_symbol": summary_df.groupby("symbol")["issues"].sum().to_dict(),
                "total_issues": int(summary_df["issues"].sum()),
                "rows_in": int(summary_df["rows_in"].sum()),
                "rows_out": int(summary_df["rows_out"].sum()),
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            _print(f"[summary] json → {out_json}")


def cmd_report_missing(args: argparse.Namespace) -> None:
    symbols = args.symbols.split(",")
    data_root = _data_root(getattr(args, "data_root", None))
    _print(f"[cfg] data_root={str(data_root)}")

    out_rows: List[Dict[str, Any]] = []
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

        out_rows.append(
            {
                "symbol": sym,
                "tf": args.tf,
                "bars": len(full),
                "present": len(df),
                "missing": len(missing),
                "missing_rate": 1.0 - (len(df) / len(full)),
            }
        )

    out_path = Path(args.out).expanduser().resolve()
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    _print(f"report → {out_path}")


# -------------------- точка входа --------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="c1-ohlcv")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("backfill", help="Догрузка истории 1m (с прогрессом и репарацией)")
    b.add_argument("--symbols", required=True, help="CSV тикеров, пример: BTCUSDT,ETHUSDT")
    b.add_argument("--since", required=True, help="ISO дата начала (UTC)")
    b.add_argument("--until", required=False, help="ISO дата окончания (UTC)")
    b.add_argument(
        "--data-root",
        required=False,
        help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT",
    )
    b.add_argument(
        "--category",
        required=False,
        choices=["spot", "linear", "inverse"],
        default="spot",
    )
    b.add_argument(
        "--spot-align-futures",
        action="store_true",
        help="Для spot: сдвигать окно к дате запуска фьючерса",
    )
    b.set_defaults(func=cmd_backfill)

    u = sub.add_parser("update", help="Обновление хвоста 1m до текущего времени - 1 бар")
    u.add_argument("--symbols", required=True)
    u.add_argument(
        "--data-root",
        required=False,
        help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT",
    )
    u.add_argument(
        "--category",
        required=False,
        choices=["spot", "linear", "inverse"],
        default="spot",
    )
    u.add_argument(
        "--spot-align-futures",
        action="store_true",
        help="Для spot: сдвигать старт к дате запуска фьючерса",
    )
    u.set_defaults(func=cmd_update)

    r = sub.add_parser("resample", help="Ресемплинг из 1m в 5m/15m/1h")
    r.add_argument("--symbols", required=True)
    r.add_argument("--from-tf", required=True, choices=["1m"])
    r.add_argument("--to-tf", required=True, choices=["5m", "15m", "1h"])
    r.add_argument(
        "--data-root",
        required=False,
        help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT",
    )
    r.set_defaults(func=cmd_resample)

    q = sub.add_parser("quality-validate", help="C2 DataQuality: валидация и санитайз Parquet")
    q.add_argument("--symbols", required=True, help="CSV тикеров")
    q.add_argument("--tf", required=True, choices=["1m", "5m", "15m", "1h"])
    q.add_argument(
        "--data-root",
        required=False,
        help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT",
    )
    q.add_argument("--write", action="store_true", help="Перезаписать очищенный файл Parquet")
    q.add_argument(
        "--issues",
        required=False,
        help="Путь для CSV-журнала проблем; append по умолчанию",
    )
    q.add_argument(
        "--issues-parquet",
        required=False,
        help="Путь для Parquet-журнала проблем; перезапись",
    )
    q.add_argument(
        "--quality-summary-csv",
        required=False,
        help="Путь для сводного CSV по качеству; append по умолчанию",
    )
    q.add_argument(
        "--quality-summary-json",
        required=False,
        help="Путь для сводного JSON по качеству; перезапись",
    )
    q.add_argument(
        "--truncate",
        action="store_true",
        help="Сначала очистить целевые CSV перед записью",
    )
    q.add_argument(
        "--no-repair",
        action="store_true",
        help="Только диагностировать, без автоправок",
    )
    q.add_argument(
        "--miss-fill-threshold",
        type=float,
        default=0.0001,
        help="Порог заполнения пропусков для 1m",
    )
    # Доп. флаги CLI оставлены для совместимости, но не используются конструктором QualityConfig
    q.add_argument("--spike-window", type=int, default=200, help="Окно MAD-детектора всплесков")
    q.add_argument("--spike-k", type=float, default=12.0, help="Порог MAD-критерия")
    q.add_argument("--flat-streak", type=int, default=300, help="Длина серии нулевого объёма")
    q.set_defaults(func=cmd_quality_validate)

    rep = sub.add_parser("report-missing", help="Отчёт о пропусках")
    rep.add_argument("--symbols", required=True)
    rep.add_argument("--tf", required=True, choices=["1m", "5m", "15m", "1h"])
    rep.add_argument("--out", required=True)
    rep.add_argument(
        "--data-root",
        required=False,
        help="Каталог данных; по умолчанию ./data или C1_DATA_ROOT",
    )
    rep.set_defaults(func=cmd_report_missing)

    args = p.parse_args()
    func: Callable[[argparse.Namespace], None] = args.func
    func(args)


if __name__ == "__main__":
    main()
