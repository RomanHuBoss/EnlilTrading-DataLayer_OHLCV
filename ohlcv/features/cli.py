# ohlcv/features/cli.py
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .core import DEFAULTS, FeatureConfig, compute_features, ensure_input
from .schema import validate_features_df

try:  # YAML опционален
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["make_parser", "main", "build_cmd"]


# ---------------- I/O ----------------

def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return _load_parquet(path)
    if suf == ".csv":
        return _load_csv(path)
    return _load_parquet(path)


def _dump_any(df: pd.DataFrame, path: Path) -> None:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    elif suf == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


# -------------- Config --------------

def _coerce_cfg_types(updates: Dict[str, Any]) -> Dict[str, Any]:
    tuple_fields = {"rv", "ema", "mom", "donch", "z"}
    out: Dict[str, Any] = {}
    for k, v in updates.items():
        if k in tuple_fields:
            if isinstance(v, list):
                out[k] = tuple(int(x) for x in v)
            elif isinstance(v, tuple):
                out[k] = tuple(int(x) for x in v)
            elif isinstance(v, int):
                out[k] = (int(v),)
            else:
                raise ValueError(f"Некорректный тип для поля {k}: {type(v)}")
        else:
            out[k] = v
    return out


def _build_config(cfg_file: Optional[Path], build_version: Optional[str]) -> FeatureConfig:
    if cfg_file is None:
        base = DEFAULTS
        return replace(base, build_version=str(build_version) if build_version else base.build_version)

    if yaml is None:
        raise RuntimeError("PyYAML не установлен, а передан --config")

    with cfg_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError("Некорректный YAML-конфиг: нужен mapping с ключами из FeatureConfig")

    allowed = {fld.name for fld in fields(FeatureConfig)}
    updates: Dict[str, Any] = {k: v for k, v in raw.items() if k in allowed}

    if build_version:
        updates["build_version"] = str(build_version)

    updates = _coerce_cfg_types(updates)
    return replace(DEFAULTS, **updates)


# -------------- Command --------------

def build_cmd(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    cfg = _build_config(Path(args.config) if args.config else None, args.build_version)

    df = _load_any(inp)
    ensure_input(df)

    out = compute_features(df, symbol=args.symbol, tf=args.tf, cfg=cfg)
    out = validate_features_df(out, strict=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_any(out, out_path)

    return 0


# -------------- Parser --------------

def _add_build_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "build",
        help="Построить признаки C3 из входного OHLCV",
        description=(
            "Строит набор признаков C3 по постановке. Вход — CSV/Parquet со схемой: "
            "timestamp_ms,start_time_iso,open,high,low,close,volume[,turnover] или ts,o,h,l,c,v[,t]."
        ),
    )
    p.add_argument(
        "input",
        type=str,
        help=(
            "Входной CSV/Parquet. Допустимы колонки marketdata (timestamp_ms, ...) или нормализованные ts,o,h,l,c,v[,t]."
        ),
    )
    p.add_argument("output", type=str, help="Файл для записи признаков (Parquet/CSV по расширению)")
    p.add_argument("--symbol", type=str, required=True, help="Идентификатор инструмента")
    p.add_argument("--tf", type=str, required=True, help="Таймфрейм входных баров, напр. '5m'")
    p.add_argument("--config", type=str, default=None, help="YAML-конфиг с полями FeatureConfig")
    p.add_argument(
        "--build-version", type=str, default=None, help="Версия сборки (перекрывает значение из конфига)"
    )
    p.set_defaults(func=build_cmd)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="c3-features")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_build_parser(sub)
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
