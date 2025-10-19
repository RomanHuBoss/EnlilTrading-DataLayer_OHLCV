from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import click
import pandas as pd

from ohlcv.quality.issues import normalize_issues_df, summarize_issues


# =============================
# IO
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


def _read_parquet(p: Optional[Path]) -> pd.DataFrame:
    if p is None or not Path(p).exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return _ensure_dt_index_utc(df)


# =============================
# Метрики качества
# =============================

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


def _daily_flagged_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0 or "dq_flags" not in df.columns:
        return pd.DataFrame(columns=["date", "flagged_ratio"])
    s = (df["dq_flags"] != 0).astype("int32")
    g = s.groupby(df.index.date).mean().rename("flagged_ratio").reset_index(names=["date"])  # date as python date
    return g


def make_quality_summary(
    sanitized_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    *,
    tf: Optional[str] = None,
    symbol: Optional[str] = None,
) -> Dict[str, Any]:
    df = _ensure_dt_index_utc(sanitized_df) if len(sanitized_df) else sanitized_df
    issues = normalize_issues_df(issues_df) if len(issues_df) else issues_df

    total = int(len(df))
    flagged = int((df.get("dq_flags", 0) != 0).sum()) if total else 0

    # Пропуски календаря для 1m (индикативно)
    gap_ratio = None
    if total:
        start, end = df.index.min(), df.index.max()
        freq = pd.infer_freq(df.index)
        if (freq is None and tf == "1m") or freq in ("T", "min"):
            cal = pd.date_range(start, end, freq="min", tz="UTC")
            gap_ratio = float(max(0.0, 1.0 - (len(df) / len(cal))))

    agg_by_code = summarize_issues(issues)
    agg_records = agg_by_code.assign(severity=agg_by_code["severity"].astype(str)).to_dict(orient="records") if len(agg_by_code) else []

    daily = _daily_flagged_ratio(df)
    daily_records = daily.assign(date=daily["date"].astype(str)).to_dict(orient="records") if len(daily) else []

    summary: Dict[str, Any] = {
        "symbol": symbol,
        "tf": tf,
        "span": {
            "start": df.index.min().isoformat() if total else None,
            "end": df.index.max().isoformat() if total else None,
        },
        "bars_total": total,
        "bars_flagged": flagged,
        "bars_flagged_ratio": (flagged / total) if total else 0.0,
        "calendar_gap_ratio_1m": gap_ratio,
        "issues_total": int(len(issues)) if len(issues) else 0,
        "issues_by_code": agg_records,
        "daily_flagged_ratio": daily_records,
    }
    return summary


# =============================
# CLI
# =============================

@click.command(help="Сводный отчёт качества C2 по sanitized parquet и issues parquet. Пишет JSON и, опционально, Markdown.")
@click.argument("sanitized_parquet", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--issues-parquet", type=click.Path(path_type=Path, exists=False, dir_okay=False), default=None, help="Путь к issues.parquet. Если отсутствует — отчёт без issues.")
@click.option("--out-json", type=click.Path(path_type=Path, dir_okay=False), required=True, help="Путь для quality.json")
@click.option("--out-md", type=click.Path(path_type=Path, dir_okay=False), default=None, help="Путь для краткого отчёта в Markdown")
@click.option("--tf", "tf_opt", type=click.Choice(["1m", "5m", "15m", "1h"]), default=None)
@click.option("--symbol", "symbol_opt", type=str, default=None)
def main(sanitized_parquet: Path, issues_parquet: Optional[Path], out_json: Path, out_md: Optional[Path], tf_opt: Optional[str], symbol_opt: Optional[str]) -> None:
    df = _read_parquet(sanitized_parquet)
    issues = _read_parquet(issues_parquet) if (issues_parquet and Path(issues_parquet).exists()) else pd.DataFrame()

    tf = tf_opt or _infer_tf_from_name(sanitized_parquet)
    symbol = symbol_opt or _infer_symbol(sanitized_parquet)

    summary = make_quality_summary(df, issues, tf=tf, symbol=symbol)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"# Quality Report — {symbol or ''} {tf or ''}".strip())
        lines.append("")
        lines.append(f"Span: {summary['span']['start']} → {summary['span']['end']}")
        lines.append(f"Bars: {summary['bars_total']} | Flagged: {summary['bars_flagged']} ({summary['bars_flagged_ratio']:.4f})")
        if summary.get("calendar_gap_ratio_1m") is not None:
            lines.append(f"Calendar gap ratio (1m): {summary['calendar_gap_ratio_1m']:.6f}")
        lines.append("")
        if summary["issues_by_code"]:
            lines.append("## Issues by code")
            for rec in summary["issues_by_code"]:
                lines.append(f"- {rec['code']}: {rec['count']} [{rec['severity']}/{rec['action']}]")
        if summary["daily_flagged_ratio"]:
            lines.append("")
            lines.append("## Daily flagged ratio")
            for rec in summary["daily_flagged_ratio"]:
                lines.append(f"- {rec['date']}: {rec['flagged_ratio']:.6f}")
        out_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
