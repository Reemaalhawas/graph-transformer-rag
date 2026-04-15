"""Collect and compare results from all three GNN architecture experiments."""
import json, os
import pandas as pd

BASE = "/home/ubuntu/GraphRAG/qa_results"
rows = []
for f in os.listdir(BASE):
    if f.endswith("_results.json"):
        with open(os.path.join(BASE, f)) as fh:
            rows.append(json.load(fh))

df = pd.DataFrame(rows)
print("\n=== QA RESULTS COMPARISON ===")
print(df.to_string(index=False))
df.to_csv(os.path.join(BASE, "comparison_table.csv"), index=False)
print(f"\nSaved: {BASE}/comparison_table.csv")
