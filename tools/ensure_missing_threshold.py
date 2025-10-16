#!/usr/bin/env python
import sys
import pandas as pd

path = sys.argv[1]
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4

df = pd.read_csv(path)
bad = df[df["missing_rate"] > thr]
if not bad.empty:
    print("FAIL: missing_rate exceeds threshold")
    print(bad.to_string(index=False))
    raise SystemExit(2)
print("OK:", path)
