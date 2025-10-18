# ohlcv/features/cli.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .core import DEFAULTS, compute_features
from .schema import EXPECTED_DTYPES, normalize_and_validate

try:  # YAML опционален, CLI должен работать и без конфигов
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _load_csv(path: Path) -> pd.DataFrame:
    dtypes: Dict[str, str] = {
        "timestamp_ms": "int64",
        "start_time_iso": "string",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
    }
    # Чтение без парсинга дат для стабильных типов и скорости
    df = pd.read_csv(path, dtype=dtypes)
    return df


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Нормализация типов к канону
    cast_map = {c: EXPECTED_DTYPES[c] for c in EXPECTED_DTYPES if c in df.columns}
    for c, dt in cast_map.items():
        try:
            df[c] = df[c].astype(dt)
        except Exception:
            pass
    return df


def _load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return _load_parquet(path)
    if suf == ".csv":
        return _load_csv(path)
    # попытка по умолчанию
    return _load_parquet(path)


def _dump_any(df: pd.DataFrame, path: Path) -> None:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
        return
    if suf == ".csv":
        df.to_csv(path, index=False)
        return
    # по умолчанию — parquet
    df.to_parquet(path, index=False)


def _merge_config(
    cfg_file: Optional[Path], strict: bool, build_version: Optional[str]
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(DEFAULTS)
    if cfg_file:
        if yaml is None:
            raise RuntimeError("PyYAML не установлен, а передан --config")
        with cfg_file.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Некорректный формат YAML-конфига: требуется mapping")
            cfg.update(loaded)
    if strict:
        cfg["strict"] = True
    if build_version:
        cfg["build_version"] = str(build_version)
    return cfg


def build_cmd(args: argparse.Namespace) -> int:
    inp = Path(args.input)
    if not inp.exists():
        raise FileNotFoundError(str(inp))

    cfg = _merge_config(
        Path(args.config) if args.config else None,
        strict=bool(args.strict),
        build_version=args.build_version,
    )

    df = _load_any(inp)
    # Приводим схему — для раннего выявления проблем
    df = normalize_and_validate(df, strict=bool(cfg.get("strict", False)))

    out = compute_features(df, symbol=args.symbol, tf=args.tf, config=cfg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _dump_any(out, out_path)

    return 0


def _add_build_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser(
        "build",
        help="Построить признаки C3 из входного OHLCV",
        description="Строит минимальный набор признаков C3 по постановке. Вход — CSV/Parquet.",
    )
    p.add_argument(
        "input",
        type=str,
        help="Входной CSV/Parquet с колонками: timestamp_ms,start_time_iso,open,high,low,close,volume",
    )
    p.add_argument("output", type=str, help="Файл для записи признаков (Parquet/CSV по расширению)")
    p.add_argument("--symbol", type=str, default="", help="Идентификатор инструмента")
    p.add_argument("--tf", type=str, default="", help="Таймфрейм входных баров, например '5m'")
    p.add_argument("--config", type=str, default=None, help="YAML-конфиг с параметрами C3")
    p.add_argument("--strict", action="store_true", help="Строгая проверка входа и NaN")
    p.add_argument(
        "--build-version", type=str, default=None, help="Версия сборки (перекрывает вычисленную)"
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
