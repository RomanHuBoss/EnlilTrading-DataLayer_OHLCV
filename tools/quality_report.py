#!/usr/bin/env python
import sys, json
import pandas as pd
issues_path = sys.argv[1]
csv_out = sys.argv[2] if len(sys.argv) > 2 else "./reports/quality_summary.csv"
json_out = sys.argv[3] if len(sys.argv) > 3 else "./reports/quality_summary.json"
df = pd.read_parquet(issues_path) if issues_path.endswith(".parquet") else pd.read_csv(issues_path)
if df.empty:
    summary = pd.DataFrame([{"symbol":"-", "tf":"-", "issues":0, "rows_in":0, "rows_out":0}])
else:
    if "symbol" not in df.columns or "tf" not in df.columns:
        df["symbol"] = "-"
        df["tf"] = "-"
    summary = (df.groupby(["symbol","tf"])["code"]
                 .count().rename("issues").reset_index()
                 .sort_values(["symbol","tf"]))
summary.to_csv(csv_out, index=False)
with open(json_out, "w", encoding="utf-8") as f:
    json.dump({
        "total_issues": int(summary["issues"].sum()),
        "by_symbol": summary.groupby("symbol")["issues"].sum().to_dict()
    }, f, ensure_ascii=False, indent=2)
print(csv_out, json_out)
