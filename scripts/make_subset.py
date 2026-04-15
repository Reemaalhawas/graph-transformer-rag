import csv
import os

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prime", "neo4j_export")
NODES_IN = os.path.join(BASE, "nodes.csv")
EDGES_IN = os.path.join(BASE, "edges.csv")
NODES_OUT = os.path.join(BASE, "nodes_subset.csv")
EDGES_OUT = os.path.join(BASE, "edges_subset.csv")
MAX_NODES = 1000

# pick first MAX_NODES node_ids (by file order)
node_ids = []
with open(NODES_IN, newline="", encoding="utf-8") as f_in, open(NODES_OUT, "w", newline="", encoding="utf-8") as f_out:
    r = csv.reader(f_in)
    w = csv.writer(f_out)
    header = next(r)
    w.writerow(header)
    for i, row in enumerate(r):
        if i >= MAX_NODES:
            break
        w.writerow(row)
        try:
            node_ids.append(int(row[0]))
        except Exception:
            node_ids.append(row[0])

node_set = set(node_ids)

# filter edges where both ends are in subset
with open(EDGES_IN, newline="", encoding="utf-8") as f_in, open(EDGES_OUT, "w", newline="", encoding="utf-8") as f_out:
    r = csv.reader(f_in)
    w = csv.writer(f_out)
    header = next(r)
    w.writerow(header)
    for row in r:
        try:
            s = int(row[0]); d = int(row[1])
        except Exception:
            s = row[0]; d = row[1]
        if s in node_set and d in node_set:
            w.writerow(row)

print("Wrote subset:", NODES_OUT, EDGES_OUT)