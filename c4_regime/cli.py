from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # noqa: BLE001
    yaml = None

from .core import infer_regime, DEFAULT_CFG


def _load_cfg(path: Optional[str]) -> dict:
    if path is None:
        return DEFAULT_CFG
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("pyyaml not installed")
        return dict(yaml.safe_load(p.read_text(encoding="utf-8")))
    if p.suffix.lower() == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError("Unsupported config format")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="regime_ensemble", description="C4.Regime ensemble inference")
    sub = ap.add_subparsers(dest="cmd", required=True)

    inf = sub.add_parser("infer", help="run regime inference")
    inf.add_argument("--features", required=True, help="Path to features parquet/csv")
    inf.add_argument("--context", required=False, help="Path to context parquet/csv")
    inf.add_argument("--config", required=False, help="Path to regime config (yaml/json)")
    inf.add_argument("--output", required=True, help="Path to output parquet/csv")

    args = ap.parse_args(argv)

    def _read_any(path: str) -> pd.DataFrame:
        if path.endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)

    if args.cmd == "infer":
        feats = _read_any(args.features)
        ctx = _read_any(args.context) if args.context else None
        cfg = _load_cfg(args.config)
        try:
            out = infer_regime(feats, context=ctx, cfg=cfg)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(json.dumps({"error": str(e)}) + "\n")
            return 2
        # save
        if args.output.endswith(".parquet"):
            out.to_parquet(args.output, index=False)
        else:
            out.to_csv(args.output, index=False)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
