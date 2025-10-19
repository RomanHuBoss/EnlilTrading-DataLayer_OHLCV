from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import click
import pandas as pd

# === C1 ===
from ohlcv.core.resample import resample_1m_to
from ohlcv.core.validate import align_and_flag_gaps, normalize_ohlcv_1m
from ohlcv.io.parquet_store import write_idempotent, parquet_path
from ohlcv.io import tail_cache as tail

# === C2 ===
from ohlcv.quality.validator import validate, QualityConfig
from ohlcv.quality.issues import normalize_issues_df

# === API (C1 backfill)
try:
    from ohlcv.api.bybit import BybitClient
except Exception:  # pragma: no cover
    BybitClient = None  # type: ignore


# =============================
# Общие утилиты
# =============================

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _write_parquet(df: pd.DataFrame, path: Path, *, compression: Optional[str] = "zstd") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression in (None, "none"):
        df.to_parquet(path)
    else:
        df.to_parquet(path, compression=compression)


def _infer_tf_from_name(path: Path) -> Optional[str]:
    name = path.stem.lower()
    for tf in ("1m", "5m", "15m", "1h"):
        if name.endswith(tf):
            return tf
    return None


# =============================
# Группы
# =============================
@click.group()
def main() -> None:
    pass

@click.group(name="datalayer")
def datalayer() -> None:
    pass

@click.group(name="dataquality")
def dataquality() -> None:
    pass

# Зарегистрировать группы под main
main.add_command(datalayer)
main.add_command(dataquality)


# =============================
# Команды C1 (базовые @click.command)
# =============================
@click.command(name="resample", help="Ресемплинг 1m → 5m/15m/1h с правилами OHLCV и is_gap (C1).")
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--dst-tf", type=click.Choice(["5m", "15m", "1h"]), required=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path, dir_okay=False), required=True)
@click.option("--compression", type=click.Choice(["zstd", "snappy", "gzip", "brotli", "none"]), default="zstd")
def cmd_resample(src: Path, dst_tf: str, out_path: Path, compression: str) -> None:
    df = _read_parquet(src)
    if _infer_tf_from_name(src) not in (None, "1m"):
        raise click.ClickException("ожидается вход 1m")
    out = resample_1m_to(df, dst_tf)
    comp = None if compression == "none" else compression
    _write_parquet(out, out_path, compression=comp)


@click.command(name="missing-report", help="Отчёт по пропускам 1m. JSON со сводкой, fail по порогу.")
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--since-ms", type=int, default=None)
@click.option("--until-ms", type=int, default=None)
@click.option("--fail-gap-pct", type=float, default=None, help="Порог доли пропусков (0..1)")
def cmd_missing_report(src: Path, since_ms: Optional[int], until_ms: Optional[int], fail_gap_pct: Optional[float]) -> None:
    df = _read_parquet(src)
    df = normalize_ohlcv_1m(df)
    if since_ms is not None:
        df = df[df.index >= pd.to_datetime(since_ms, unit="ms", utc=True)]
    if until_ms is not None:
        df = df[df.index < pd.to_datetime(until_ms, unit="ms", utc=True)]
    aligned, _stats = align_and_flag_gaps(df, strict=False)
    gaps = int(aligned["is_gap"].sum()) if "is_gap" in aligned.columns else 0
    total = int(len(aligned))
    pct = float(gaps / total) if total else 0.0
    payload: Dict[str, Any] = {
        "rows": total,
        "gaps": gaps,
        "gap_pct": pct,
    }
    click.echo(json.dumps(payload, ensure_ascii=False))
    if fail_gap_pct is not None and pct > fail_gap_pct:
        raise SystemExit(2)


@click.command(name="backfill", help="Загрузка 1m OHLCV с Bybit и запись в Parquet Store (C1).")
@click.argument("symbol", type=str)
@click.option("--store", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--since-ms", type=int, required=True)
@click.option("--until-ms", type=int, required=True)
@click.option("--page-size", type=int, default=1000)
@click.option("--base-url", type=str, default="https://api.bybit.com")
@click.option("--timeout-s", type=int, default=10)
@click.option("--max-retries", type=int, default=5)
@click.option("--max-concurrent", type=int, default=2)
def cmd_backfill(symbol: str, store: Path, since_ms: int, until_ms: int, page_size: int, base_url: str, timeout_s: int, max_retries: int, max_concurrent: int) -> None:  # noqa: E501
    if BybitClient is None:
        raise click.ClickException("API Bybit недоступен в сборке")
    client = BybitClient(base_url=base_url, read_only_key=None, timeout_s=timeout_s, max_retries=max_retries, max_concurrent=max_concurrent)  # type: ignore  # noqa: E501
    cur = since_ms
    buf = []
    while cur < until_ms:
        chunk, next_ms = client.fetch_klines_1m(symbol, cur, min(until_ms, cur + page_size * 60_000))  # type: ignore
        if chunk is None or len(chunk) == 0:
            break
        buf.append(chunk)
        if next_ms is None or next_ms <= cur:
            break
        cur = next_ms
    if len(buf) == 0:
        raise click.ClickException("пустой ответ API")
    df = pd.concat(buf, ignore_index=True)
    if "ts" not in df.columns:
        raise click.ClickException("ответ API без 'ts'")
    path = write_idempotent(store, symbol, "1m", df)
    tail.write(store, tail.TailInfo(symbol=symbol, tf="1m", latest_ts_ms=int(df["ts"].max())))
    click.echo(str(path))


# Зарегистрировать C1-команды в группе
datalayer.add_command(cmd_resample)
datalayer.add_command(cmd_missing_report)
datalayer.add_command(cmd_backfill)


# =============================
# Команды C2 (базовые @click.command)
# =============================
@click.command(name="validate", help="Валидация C2: читает parquet, запускает validate, пишет df и issues.")
@click.argument("input_parquet", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--tf", "tf_opt", type=click.Choice(["1m", "5m", "15m", "1h"]), default=None)
@click.option("--symbol", "symbol_opt", type=str, default=None)
@click.option("--ref-1m", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
@click.option("--out-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
@click.option("--quality-json", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.option("--write-inplace", is_flag=True, default=False)
@click.option("--cfg-json", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None)
def cmd_validate(input_parquet: Path, tf_opt: Optional[str], symbol_opt: Optional[str], ref_1m: Optional[Path], out_dir: Optional[Path], quality_json: Optional[Path], write_inplace: bool, cfg_json: Optional[Path]) -> None:  # noqa: E501
    df = _read_parquet(input_parquet)
    tf = tf_opt or _infer_tf_from_name(input_parquet) or "1m"
    symbol = symbol_opt or input_parquet.parent.name
    ref_df = _read_parquet(ref_1m) if ref_1m else None

    cfg = None
    if cfg_json:
        cfg = QualityConfig(**json.loads(Path(cfg_json).read_text(encoding="utf-8")))

    df_out, issues_df = validate(df, tf=tf, symbol=symbol, repair=True, config=cfg, ref_1m=ref_df)

    if write_inplace:
        _write_parquet(df_out, input_parquet)
        issues_path = input_parquet.with_suffix(input_parquet.suffix + ".issues.parquet")
    else:
        out_dir = out_dir or input_parquet.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet_path = out_dir / input_parquet.name
        _write_parquet(df_out, out_parquet_path)
        issues_path = out_dir / (input_parquet.stem + ".issues.parquet")

    issues_df = normalize_issues_df(issues_df)
    if len(issues_df) == 0:
        empty = pd.DataFrame(columns=["ts", "code", "note", "severity", "action", "dq_rank", "symbol", "tf"])
        empty.to_parquet(issues_path)
    else:
        issues_df.to_parquet(issues_path)

    if quality_json:
        summary = {
            "rows_out": int(len(df_out)),
            "issues": int(len(issues_df)),
            "symbol": symbol,
            "tf": tf,
        }
        Path(quality_json).parent.mkdir(parents=True, exist_ok=True)
        Path(quality_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


# Зарегистрировать C2-команду в группе
dataquality.add_command(cmd_validate)


# =============================
# Топ-уровневые алиасы (совместимость с тестами и старым интерфейсом)
# =============================
# report-missing из store по (symbol, tf=1m)
@click.command(name="report-missing", help="Совместимость: отчёт по пропускам 1m из store/\n(symbol, --since-ms/--until-ms, --fail-gap-pct).")
@click.option("--symbol", type=str, required=True)
@click.option("--store", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.option("--since-ms", type=int, default=None)
@click.option("--until-ms", type=int, default=None)
@click.option("--fail-gap-pct", type=float, default=None)
def cmd_report_missing_legacy(symbol: str, store: Path, since_ms: Optional[int], until_ms: Optional[int], fail_gap_pct: Optional[float]) -> None:
    src = parquet_path(store, symbol, "1m")
    cmd_missing_report.callback(src, since_ms, until_ms, fail_gap_pct)  # type: ignore[attr-defined]

# c1-update: мердж произвольного parquet в store по инференсу TF
@click.command(name="c1-update", help="Совместимость: слияние parquet-файла в store по (symbol, tf inferred).")
@click.option("--symbol", type=str, required=True)
@click.option("--store", type=click.Path(path_type=Path, file_okay=False), required=True)
@click.argument("src", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def cmd_update_legacy(symbol: str, store: Path, src: Path) -> None:
    df = _read_parquet(src)
    tf = _infer_tf_from_name(src) or "1m"
    write_idempotent(store, symbol, tf, df)

# Зарегистрировать алиасы в main
main.add_command(cmd_resample, name="c1-resample")
main.add_command(cmd_backfill, name="c1-backfill")
main.add_command(cmd_report_missing_legacy)
main.add_command(cmd_update_legacy)
main.add_command(cmd_validate, name="c2-validate")


if __name__ == "__main__":
    main()
