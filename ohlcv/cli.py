# ohlcv/cli.py
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# Локальные импорты
from .api.bybit import BybitClient
from .core.validate import normalize_ohlcv_1m, align_and_flag_gaps
from .core.resample import resample_1m_to
from .io.parquet_store import write_idempotent, parquet_path
from .utils.timeframes import MINUTE_MS, parse_tf


# ---------------------
# Вспомогательные
# ---------------------

def _now_ms() -> int:
    return int(time.time() * 1000)


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _read_store_or_empty(root: Path | str, symbol: str, tf: str) -> pd.DataFrame:
    path = parquet_path(root, symbol, tf)
    if not Path(path).exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="ts", tz="UTC"))
    df = pd.read_parquet(path)
    # нормализация типов/индекса
    if "ts" not in df.columns:
        raise ValueError("ожидается столбец 'ts' в parquet")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def _latest_ts_ms(root: Path | str, symbol: str, tf: str) -> Optional[int]:
    path = parquet_path(root, symbol, tf)
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p, columns=["ts"])  # быстрое чтение только индекса
    if len(df) == 0:
        return None
    ts = pd.to_datetime(df["ts"], utc=True)
    return int(ts.iloc[-1].value // 1_000_000)


# ---------------------
# Команды
# ---------------------

def cmd_backfill(args: argparse.Namespace) -> int:
    symbol = args.symbol
    store = Path(args.store)
    _ensure_parent(store / symbol)

    since = int(args.since_ms)
    until = int(args.until_ms) if args.until_ms is not None else _now_ms()

    client = BybitClient(max_retries=args.retries, jitter=args.jitter)

    df_raw, stats = client.fetch_ohlcv_1m(
        symbol=symbol,
        start_ms=since,
        end_ms=until,
        category=args.category,
        limit=args.limit,
        slice_minutes=args.slice_minutes,
    )

    if df_raw.empty:
        print(json.dumps({"rows": 0, "message": "no data"}))
        return 0

    # Нормализация к канону 1m (o/h/l/c/v + t)
    df_norm, _ = normalize_ohlcv_1m(df_raw)

    # Запись в хранилище 1m
    path = write_idempotent(store, symbol, "1m", df_norm)

    meta = {
        "symbol": symbol,
        "tf": "1m",
        "rows": int(len(df_norm)),
        "path": str(path),
        "requests": stats.requests,
        "retries": stats.retries,
        "rate_limited": stats.rate_limited,
        "start_ms": stats.start_ms,
        "end_ms": stats.end_ms,
    }
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    symbol = args.symbol
    store = Path(args.store)
    _ensure_parent(store / symbol)

    last = _latest_ts_ms(store, symbol, "1m")
    since = int(args.since_ms) if args.since_ms is not None else (last + MINUTE_MS if last is not None else _now_ms() - 86_400_000)
    until = int(args.until_ms) if args.until_ms is not None else _now_ms()

    client = BybitClient(max_retries=args.retries, jitter=args.jitter)
    df_raw, stats = client.fetch_ohlcv_1m(
        symbol=symbol,
        start_ms=since,
        end_ms=until,
        category=args.category,
        limit=args.limit,
        slice_minutes=args.slice_minutes,
    )

    if df_raw.empty:
        print(json.dumps({"rows": 0, "message": "no new data"}))
        return 0

    df_norm, _ = normalize_ohlcv_1m(df_raw)
    path = write_idempotent(store, symbol, "1m", df_norm)

    meta = {
        "symbol": symbol,
        "tf": "1m",
        "rows": int(len(df_norm)),
        "path": str(path),
        "requests": stats.requests,
        "retries": stats.retries,
        "rate_limited": stats.rate_limited,
        "start_ms": stats.start_ms,
        "end_ms": stats.end_ms,
    }
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def cmd_resample(args: argparse.Namespace) -> int:
    symbol = args.symbol
    store = Path(args.store)

    src_tf = "1m"
    dst_tf = args.dst_tf

    df = _read_store_or_empty(store, symbol, src_tf)
    if df.empty:
        print(json.dumps({"rows": 0, "message": "no source data"}))
        return 0

    out = resample_1m_to(df, dst_tf, allow_partial=args.allow_partial)
    if out.empty:
        print(json.dumps({"rows": 0, "message": "no aggregated data"}))
        return 0

    path = write_idempotent(store, symbol, dst_tf, out)
    print(json.dumps({"symbol": symbol, "tf": dst_tf, "rows": int(len(out)), "path": str(path)}, ensure_ascii=False))
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    symbol = args.symbol
    store = Path(args.store)
    tf = args.tf

    df = _read_store_or_empty(store, symbol, tf)
    if df.empty:
        print(json.dumps({"rows": 0, "message": "empty"}))
        return 0

    if args.since_ms is not None:
        s = pd.to_datetime(int(args.since_ms), unit="ms", utc=True)
        df = df.loc[df.index >= s]
    if args.until_ms is not None:
        e = pd.to_datetime(int(args.until_ms), unit="ms", utc=True)
        df = df.loc[df.index < e]

    # вывод
    out = df.reset_index()
    p = Path(args.output)
    _ensure_parent(p)
    if p.suffix.lower() == ".csv":
        out.to_csv(p, index=False)
    else:
        out.to_parquet(p, index=False)

    print(json.dumps({"rows": int(len(out)), "path": str(p)}, ensure_ascii=False))
    return 0


def cmd_report_missing(args: argparse.Namespace) -> int:
    symbol = args.symbol
    store = Path(args.store)

    df = _read_store_or_empty(store, symbol, "1m")
    if df.empty:
        print(json.dumps({"rows": 0, "message": "no data"}))
        return 0

    # диапазон интереса
    since = int(args.since_ms) if args.since_ms is not None else int(df.index[0].value // 1_000_000)
    until = int(args.until_ms) if args.until_ms is not None else int(df.index[-1].value // 1_000_000 + MINUTE_MS)

    # Выравнивание и флаг пропусков
    aligned, stats = align_and_flag_gaps(df, start_ms=since, end_ms=until)

    total_minutes = int(len(aligned))
    gaps = int(aligned["is_gap"].sum())
    present = int(total_minutes - gaps)
    gap_pct = (gaps / total_minutes * 100.0) if total_minutes else 0.0

    report = {
        "symbol": symbol,
        "range": [since, until],
        "total_minutes": total_minutes,
        "present": present,
        "gaps": gaps,
        "gap_pct": round(gap_pct, 6),
    }

    if args.output:
        p = Path(args.output)
        _ensure_parent(p)
        pd.DataFrame(aligned[["is_gap"]]).reset_index().to_parquet(p, index=False) if p.suffix.lower() != ".csv" else pd.DataFrame(aligned[["is_gap"]]).reset_index().to_csv(p, index=False)
        report["details_path"] = str(p)

    print(json.dumps(report, ensure_ascii=False))
    # CI-гейт по порогу
    if args.fail_gap_pct is not None and gap_pct > float(args.fail_gap_pct):
        return 2
    return 0


# ---------------------
# Аргументы
# ---------------------

def _add_common_range_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--since-ms", type=int, required=False, help="левая граница [мс] включительно")
    p.add_argument("--until-ms", type=int, required=False, help="правая граница [мс] исключительно")


def _add_http_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--category", type=str, default="linear", help="Bybit category: linear|inverse|spot")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--slice-minutes", type=int, default=1000)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--jitter", type=float, default=0.2)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="c1-ohlcv", description="Data Layer C1 — загрузка 1m, ресемплинг и отчёты о пропусках")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # backfill
    p = sub.add_parser("backfill", help="Изначальная выгрузка минутных OHLCV и запись в Parquet store")
    p.add_argument("--symbol", required=True)
    p.add_argument("--store", required=True)
    _add_common_range_args(p)
    _add_http_args(p)
    p.set_defaults(func=cmd_backfill)

    # update
    p = sub.add_parser("update", help="Догрузка новых минутных OHLCV и запись в Parquet store")
    p.add_argument("--symbol", required=True)
    p.add_argument("--store", required=True)
    _add_common_range_args(p)
    _add_http_args(p)
    p.set_defaults(func=cmd_update)

    # resample
    p = sub.add_parser("resample", help="Ресемплинг 1m → целевой ТФ и запись в Parquet store")
    p.add_argument("--symbol", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--dst-tf", required=True, choices=["5m", "15m", "1h"])
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(func=cmd_resample)

    # read
    p = sub.add_parser("read", help="Чтение из хранилища и экспорт в файл")
    p.add_argument("--symbol", required=True)
    p.add_argument("--store", required=True)
    p.add_argument("--tf", required=True)
    _add_common_range_args(p)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_read)

    # report-missing
    p = sub.add_parser("report-missing", help="Отчёт по пропускам минутных баров")
    p.add_argument("--symbol", required=True)
    p.add_argument("--store", required=True)
    _add_common_range_args(p)
    p.add_argument("--output", required=False, help="CSV/Parquet с детализацией по минутам")
    p.add_argument("--fail-gap-pct", type=float, required=False, help="Порог для неуспешного кода возврата")
    p.set_defaults(func=cmd_report_missing)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    try:
        return int(func(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
