"""
Adds desc and label fields to pre-computed subgraphs.
Run once: python enrich_subgraphs.py

Only needed if:
  - desc is empty in the subgraph objects, AND
  - nodeIds are stored in the subgraph objects (check with Section 3.3 commands)

Requires: Neo4j KG running and accessible via .env credentials
"""
import os, json, re, torch
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASSWORD")
assert URI,  "NEO4J_URI not set in .env"
assert PASS, "NEO4J_PASSWORD not set in .env"

FS = "/home/ubuntu/stark-graphrag-storage"
SPLITS = {
    "train": f"{FS}/stark_train_subgraphs.pt",
    "val":   f"{FS}/stark_val_subgraphs.pt",
    "test":  f"{FS}/stark_test_subgraphs.pt",
}


def parse_ids(s):
    try:
        v = json.loads(str(s))
        return v if isinstance(v, list) else [v]
    except Exception:
        return [int(x) for x in re.findall(r'\d+', str(s))]


def get_node_info(ids, driver):
    r = driver.execute_query("""
        UNWIND $ids AS nid
        MATCH (n {nodeId: nid})
        RETURN n.nodeId AS nodeId, n.name AS name,
               coalesce(n.details, '') AS desc
    """, parameters_={"ids": ids})
    return {rec["nodeId"]: rec for rec in [x.data() for x in r.records]}


def get_edge_info(src_ids, tgt_ids, driver):
    pairs = [[int(s), int(t)] for s, t in zip(src_ids, tgt_ids)]
    r = driver.execute_query("""
        UNWIND $pairs AS pair
        MATCH (s {nodeId: pair[0]})-[e]->(t {nodeId: pair[1]})
        RETURN s.nodeId AS src, type(e) AS etype, t.nodeId AS dst
    """, parameters_={"pairs": pairs})
    return [x.data() for x in r.records]


def make_desc(node_info, edge_rows):
    lines = ["node_id,node_attr"]
    for nid, info in node_info.items():
        name = info.get("name", str(nid))
        desc = str(info.get("desc", ""))[:300]
        lines.append(f'{nid},"{name}: {desc}"')
    lines.append("src,edge_attr,dst")
    for e in edge_rows:
        lines.append(f'{e["src"]},{e["etype"]},{e["dst"]}')
    return "\n".join(lines)


def find_node_ids_attr(item):
    """Find which attribute stores the original nodeIds."""
    for attr in ["node_ids", "nodeIds", "n_id", "original_ids", "mapping"]:
        if hasattr(item, attr):
            return getattr(item, attr)
    return None


def enrich(path, out_path, driver):
    if not os.path.exists(path):
        print(f"SKIP (not found): {path}")
        return
    print(f"\nLoading {path}...")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, list):
        data = list(data.values())
    print(f"Enriching {len(data)} subgraphs...")

    no_ids_count = 0
    for item in tqdm(data):
        if hasattr(item, "desc") and item.desc and item.desc != "":
            continue  # already enriched

        node_ids_raw = find_node_ids_attr(item)
        if node_ids_raw is None:
            no_ids_count += 1
            item.desc = ""
            item.label = str(getattr(item, "answer", ""))
            continue

        nids = node_ids_raw.tolist() if hasattr(node_ids_raw, "tolist") else list(node_ids_raw)
        nids = [int(n) for n in nids]
        node_info = get_node_info(nids, driver)

        edge_rows = []
        if hasattr(item, "edge_index") and item.edge_index is not None:
            src = [nids[i] for i in item.edge_index[0].tolist()]
            tgt = [nids[i] for i in item.edge_index[1].tolist()]
            edge_rows = get_edge_info(src, tgt, driver)

        item.desc = make_desc(node_info, edge_rows)

        ans_ids = parse_ids(getattr(item, "answer", "[]"))
        ans_names = [node_info.get(aid, {}).get("name", str(aid)) for aid in ans_ids]
        item.label = " | ".join(ans_names)

    if no_ids_count > 0:
        print(f"WARNING: {no_ids_count} items had no nodeId mapping — desc left empty")
    torch.save(data, out_path)
    print(f"Saved: {out_path}")
    d = torch.load(out_path, map_location="cpu", weights_only=False)
    print(f"Verify — desc sample: {str(d[0].desc)[:150]}")
    print(f"Verify — label: {d[0].label}")


if __name__ == "__main__":
    driver = GraphDatabase.driver(URI, auth=(USER, PASS))
    print(f"Connected to Neo4j: {URI}")
    for split, path in SPLITS.items():
        out = path.replace(".pt", "_enriched.pt")
        enrich(path, out, driver)
    driver.close()
    print("\nEnrichment complete.")
