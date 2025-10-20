# c4_regime/cli.py
from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Используем актуальную реализацию из пакета ohlcv.regime
from .core import RegimeConfig, infer_regime

try:  # YAML опционален
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

__all__ = ["make_parser", "main", "infer_cmd"]


# ---------------- I/O ----------------

def _read_any(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def _write_any(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)


# ---------------- Config ----------------

def _coerce_cfg_types(upd: Dict[str, Any]) -> Dict[str, Any]:
    tuple_fields = {"N_conf", "N_cooldown"}
    out: Dict[str, Any] = {}
    for k, v in upd.items():
        if k in tuple_fields and isinstance(v, dict):
            out[k] = {str(kk): int(vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _build_cfg(cfg_path: Optional[str]) -> RegimeConfig:
    if cfg_path is None:
        return RegimeConfig()
    p = Path(cfg_path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML не установлен, а передан --config")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        import json
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Формат конфига не поддерживается: нужен .yml/.yaml или .json")

    if not isinstance(raw, dict):
        raise ValueError("Конфиг должен быть mapping с полями RegimeConfig")

    allowed = {fld.name for fld in fields(RegimeConfig)}
    upd = {k: v for k, v in raw.items() if k in allowed}
    upd = _coerce_cfg_types(upd)
    return replace(RegimeConfig(), **upd)


# ---------------- Command ----------------

def infer_cmd(args: argparse.Namespace) -> int:
    feats = _read_any(args.features)
    cfg = _build_cfg(args.config)
    out = infer_regime(feats, tf=args.tf, cfg=cfg)
    _write_any(out, args.output)
    return 0


# ---------------- Parser ----------------

def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="c4-regime", description="C4 Regime ensemble")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("infer", help="Инференс режима рынка")
    p.add_argument("--features", required=True, help="Путь к фичам (parquet/csv)")
    p.add_argument("--output", required=True, help="Путь для результата (parquet/csv)")
    p.add_argument("--tf", required=True, help="Таймфрейм входных баров, напр. '5m'")
    p.add_argument("--config", required=False, help="Конфигурация Regime (yaml/json)")
    p.set_defaults(func=infer_cmd)

    return ap


def main(argv: Optional[list[str]] = None) -> int:
    ap = make_parser()
    args = ap.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        ap.print_help()
        return 1
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
