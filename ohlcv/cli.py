from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd

# === C2 ===
from ohlcv.quality.validator import validate, QualityConfig
from ohlcv.quality.issues import normalize_issues_df, summarize_issues


# =============================
# Общие утилиты IO и времени (C1/C2)
# =============================

def _ensure_dt_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df = df.set_index(pd.to_datetime(df["ts"], utc=True))
        else:
            raise ValueError("ожидается DatetimeIndex или колонка ts")
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx
    return out.sort_index()


def _read_parquet(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    return _ensure_dt_index_utc(df)


def _write_parquet(df: pd.DataFrame, path: Path, *, compression: str = "zstd") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Минимальная совместимость с C1: сортировка индекса и явная компрессия
    _ensure_dt_index_utc(df).to_parquet(path, engine="pyarrow", compression=compression)


def _infer_tf_from_name(p: Path) -> Optional[str]:
    name = p.stem.lower()
    for tf in ("1m", "5m", "15m", "1h"):
        if name.endswith(f"_{tf}") or name.endswith(f".{tf}") or f"_{tf}_" in name:
            return tf
    return None


def _infer_symbol(p: Path) -> Optional[str]:
    stem = p.stem
    for sep in ("_", ".", "-"):
        if sep in stem:
            return stem.split(sep)[0]
    return p.parent.name if p.parent and p.parent.name else None


def _quality_summary(df: pd.DataFrame, issues_df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    flagged = int((df.get("dq_flags", 0) != 0).sum()) if total else 0
    pct = (flagged / total) if total else 0.0
    agg = summarize_issues(issues_df)
    agg_records = agg.assign(severity=agg["severity"].astype(str)).to_dict(orient="records")
    return {
        "bars_total": total,
        "bars_flagged": flagged,
        "bars_flagged_ratio": pct,
        "issues_by_code": agg_records,
    }


# =============================
# Группа CLI
# =============================

@click.group(help="OHLCV утилиты. Команды C1 (данные) и C2 (качество).")
def main() -> None:
    pass


# =============================
# C1 — операции данных
# =============================

@main.command(name="c1-read", help="Чтение parquet в stdout-описание (размер, даты, колонки).")
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def c1_read(src: Path) -> None:
    df = _read_parquet(src)
    click.echo(json.dumps({
        "rows": int(len(df)),
        "start": df.index.min().isoformat() if len(df) else None,
        "end": df.index.max().isoformat() if len(df) else None,
        "columns": list(df.columns),
        "tz": str(df.index.tz),
    }, ensure_ascii=False, indent=2))


def _resample(df: pd.DataFrame, dst_tf: str) -> pd.DataFrame:
    rule = {"5m": "5min", "15m": "15min", "1h": "1h"}[dst_tf]
    g = df.resample(rule, label="right", closed="right")
    out = pd.DataFrame({
        "o": g["o"].first(),
        "h": g["h"].max(),
        "l": g["l"].min(),
        "c": g["c"].last(),
    })
    if "v" in df.columns:
        out["v"] = g["v"].sum()
    if "t" in df.columns:
        out["t"] = g["t"].sum()
    if "is_gap" in df.columns:
        out["is_gap"] = g["is_gap"].max()
    return out.dropna(how="all")


@main.command(name="c1-resample", help="Ресемплинг 1m → 5m/15m/1h с правилами OHLCV.")
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--dst-tf", type=click.Choice(["5m", "15m", "1h"]), required=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--compression", type=click.Choice(["zstd", "snappy", "gzip", "brotli", "none"]), default="zstd")
def c1_resample(src: Path, dst_tf: str, out_path: Path, compression: str) -> None:
    df = _read_parquet(src)
    if _infer_tf_from_name(src) not in (None, "1m"):
        raise click.ClickException("ожидается вход 1m")
    out = _resample(df, dst_tf)
    comp = None if compression == "none" else compression
    _write_parquet(out, out_path, compression=comp or "zstd")


@main.command(name="c1-report-missing", help="Отчёт по пропускам 1m. Выводит JSON со сводкой и опциональным fail п/порогу.")
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--fail-gap-pct", type=float, default=None, help="Порог доли пропусков (0..1) для выхода с кодом 2.")
def c1_report_missing(src: Path, fail_gap_pct: Optional[float]) -> None:
    df = _read_parquet(src)
    if len(df) == 0:
        res = {"rows": 0, "gap_ratio": 1.0, "gaps": 0, "expected": 0}
        click.echo(json.dumps(res, ensure_ascii=False, indent=2))
        if fail_gap_pct is not None and res["gap_ratio"] > fail_gap_pct:
            raise SystemExit(2)
        return

    start, end = df.index.min(), df.index.max()
    cal = pd.date_range(start, end, freq="min", tz="UTC")
    expected = len(cal)
    observed = len(df)
    gaps = expected - observed
    gap_ratio = max(0.0, gaps / expected) if expected else 0.0

    res = {
        "rows": int(observed),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected": int(expected),
        "gaps": int(gaps),
        "gap_ratio": float(gap_ratio),
    }
    click.echo(json.dumps(res, ensure_ascii=False, indent=2))
    if fail_gap_pct is not None and gap_ratio > fail_gap_pct:
        raise SystemExit(2)


@main.command(name="c1-backfill", help="Backfill 1m из провайдера и запись в parquet (идемпотентно при наличии store).")
@click.option("--exchange", type=click.Choice(["bybit"]), default="bybit")
@click.option("--symbol", type=str, required=True)
@click.option("--start", type=str, required=True, help="ISO-8601 UTC")
@click.option("--end", type=str, required=True, help="ISO-8601 UTC")
@click.option("--out", "out_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--compression", type=click.Choice(["zstd", "snappy", "gzip", "brotli", "none"]), default="zstd")
def c1_backfill(exchange: str, symbol: str, start: str, end: str, out_path: Path, compression: str) -> None:
    mod_api = import_module(f"ohlcv.api.{exchange}")
    # Ожидается функция fetch_ohlcv_1m(symbol, start, end) -> DataFrame(UTC, ohlcv)
    if not hasattr(mod_api, "fetch_ohlcv_1m"):
        raise click.ClickException("в провайдере нет fetch_ohlcv_1m")
    df = mod_api.fetch_ohlcv_1m(symbol=symbol, start=start, end=end)
    df = _ensure_dt_index_utc(df)

    # Попытка использовать store.write_idempotent, иначе — минимальный fallback
    try:
        mod_store = import_module("ohlcv.io.parquet_store")
        if hasattr(mod_store, "write_idempotent"):
            mod_store.write_idempotent(df, out_path, compression=None if compression == "none" else compression)
            return
    except Exception:
        pass

    # Fallback: простая запись/слияние по индексу
    old = _read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    merged = pd.concat([old, df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    comp = None if compression == "none" else compression
    _write_parquet(merged, out_path, compression=comp or "zstd")


@main.command(name="c1-update", help="Инкрементальное обновление 1m до now() UTC.")
@click.option("--exchange", type=click.Choice(["bybit"]), default="bybit")
@click.option("--symbol", type=str, required=True)
@click.option("--dst", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--since", type=str, default=None, help="ISO-8601 UTC; если не задано — берём последний бар в dst")
@click.option("--compression", type=click.Choice(["zstd", "snappy", "gzip", "brotli", "none"]), default="zstd")
def c1_update(exchange: str, symbol: str, dst: Path, since: Optional[str], compression: str) -> None:
    if not dst.exists() and since is None:
        raise click.ClickException("dst не существует и не задан --since")
    start = since
    if start is None:
        df_old = _read_parquet(dst)
        if len(df_old) == 0:
            raise click.ClickException("dst пуст и не задан --since")
        start = (df_old.index.max()).isoformat()

    mod_api = import_module(f"ohlcv.api.{exchange}")
    if hasattr(mod_api, "fetch_ohlcv_1m"):
        df_new = mod_api.fetch_ohlcv_1m(symbol=symbol, start=start, end=None)
    elif hasattr(mod_api, "update_ohlcv_1m"):
        df_new = mod_api.update_ohlcv_1m(symbol=symbol, start=start)
    else:
        raise click.ClickException("в провайдере нет fetch_ohlcv_1m/update_ohlcv_1m")

    df_new = _ensure_dt_index_utc(df_new)
    old = _read_parquet(dst) if dst.exists() else pd.DataFrame()
    merged = pd.concat([old, df_new]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    comp = None if compression == "none" else compression
    _write_parquet(merged, dst, compression=comp or "zstd")


# =============================
# C2 — валидация качества данных
# =============================

@main.command(name="c2-validate", help="Валидация C2: читает parquet, запускает validate, сохраняет санитайз и issues.")
@click.argument("input_parquet", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--tf", "tf_opt", type=click.Choice(["1m", "5m", "15m", "1h"]), default=None)
@click.option("--symbol", "symbol_opt", type=str, default=None)
@click.option("--ref-1m", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--out-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--out-parquet", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--issues-parquet", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--quality-json", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--write-inplace", is_flag=True, default=False)
@click.option("--cfg-json", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
def c2_validate(
    input_parquet: Path,
    tf_opt: Optional[str],
    symbol_opt: Optional[str],
    ref_1m: Optional[Path],
    out_dir: Optional[Path],
    out_parquet: Optional[Path],
    issues_parquet: Optional[Path],
    quality_json: Optional[Path],
    write_inplace: bool,
    cfg_json: Optional[Path],
) -> None:
    tf = tf_opt or _infer_tf_from_name(input_parquet) or "1m"
    symbol = symbol_opt or _infer_symbol(input_parquet)

    base_dir = out_dir or input_parquet.parent
    stem = input_parquet.stem

    out_parquet_path = input_parquet if write_inplace else (out_parquet or base_dir / f"{stem}.sanitized.parquet")
    issues_parquet_path = issues_parquet or base_dir / f"{stem}.issues.parquet"
    quality_json_path = quality_json or base_dir / f"{stem}.quality.json"

    cfg = QualityConfig(**json.loads(Path(cfg_json).read_text())) if cfg_json else QualityConfig()

    df = _read_parquet(input_parquet)
    ref_df = _read_parquet(ref_1m) if ref_1m else None

    df_out, issues_df = validate(df, tf=tf, symbol=symbol, repair=True, config=cfg, ref_1m=ref_df)

    _write_parquet(df_out, out_parquet_path)

    issues_df = normalize_issues_df(issues_df)
    if len(issues_df) == 0:
        empty = pd.DataFrame(columns=["ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"])
        empty.to_parquet(issues_parquet_path, engine="pyarrow")
    else:
        issues_df.to_parquet(issues_parquet_path, engine="pyarrow")

    summary = _quality_summary(df_out, issues_df)
    quality_json_path.parent.mkdir(parents=True, exist_ok=True)
    quality_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
