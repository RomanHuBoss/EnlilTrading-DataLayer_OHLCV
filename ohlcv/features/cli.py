from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import DEFAULTS, compute_features
from .utils import BuildMeta


def _load_csv(path: Path) -> pd.DataFrame:
    dtypes = {
        "timestamp_ms": "int64",
        "start_time_iso": "string",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "turnover": "float64",
    }
    df = pd.read_csv(path, dtype={k: v for k, v in dtypes.items() if k != "turnover"})
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    return df


def _load_params(path: Path | None):
    if not path:
        return DEFAULTS
    try:
        import yaml
    except Exception as e:
        raise RuntimeError("Для чтения YAML-конфига требуется пакет PyYAML") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return {**DEFAULTS, **cfg}


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="features-core",
        description="C3 Features.Core — build features from OHLCV CSV/Parquet",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build features parquet/csv from CSV/Parquet input")
    b.add_argument("--input", required=True, type=Path)
    b.add_argument("--symbol", required=True)
    b.add_argument("--tf", required=True)
    b.add_argument("--config", type=Path, default=None, help="YAML с параметрами окон (optional)")
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--build-version", default=None, help="Override f_build_version (optional)")
    b.add_argument(
        "--strict",
        action="store_true",
        help="строгая проверка схемы/NaN (падать при несоответствии)",
    )

    args = p.parse_args(argv)

    if args.cmd == "build":
        # CSV/Parquet auto-load
        if args.input.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(args.input)
        else:
            df = _load_csv(args.input)

        params = _load_params(args.config)
        if args.strict:
            params["strict"] = True

        out = compute_features(df, symbol=args.symbol, tf=args.tf, params=params)

        meta = BuildMeta(component="C3.Features.Core", version="0.1.0", params=params)
        bv = args.build_version or meta.build_version()
        out["f_build_version"] = bv

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() in [".parquet", ".pq"]:
            out.to_parquet(args.output, index=False)
        elif args.output.suffix.lower() in [".csv"]:
            out.to_csv(args.output, index=False)
        else:
            out.to_parquet(args.output, index=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
