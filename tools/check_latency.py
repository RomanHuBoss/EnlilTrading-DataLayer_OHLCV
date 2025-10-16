#!/usr/bin/env python
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
grace = int(sys.argv[2]) if len(sys.argv) > 2 else 90
now = datetime.now(timezone.utc)
viol = []
for sym in os.listdir(root):
    p = root / sym / "1m.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p, columns=["ts"])
    ts = pd.to_datetime(df["ts"], utc=True).max().to_pydatetime()
    delay = (now - ts).total_seconds()
    if delay > grace:
        viol.append((sym, int(delay)))
if viol:
    print("FAIL: latency > grace:", viol)
    raise SystemExit(2)
print("OK: latency within", grace, "seconds")
