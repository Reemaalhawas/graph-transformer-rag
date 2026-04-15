# experiments/node_classification/build_graph.py
"""
Pulls the full STaRK-Prime graph from Neo4j and saves as PyG Data.
Run once before node classification training.
Usage: python experiments/node_classification/build_graph.py
"""
import os, sys, torch, numpy as np, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from neo4j import GraphDatabase
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
from dotenv import load_dotenv

load_dotenv()
URI  = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER", "neo4j")
PASS = os.getenv("NEO4J_PASSWORD")
SAVE = "/home/ubuntu/GraphRAG/node_clf_data.pt"

TYPE_TO_IDX = {
    "Drug": 0,
    "GeneOrProtein": 1,
    "Disease": 2,
    "BiologicalProcess": 3,
    "MolecularFunction": 4,
    "Pathway": 5,
    "Anatomy": 6,
}


def build():
    driver = GraphDatabase.driver(URI, auth=(USER, PASS))
    print(f"Connected: {URI}")

    print("Pulling nodes...")
    r = driver.execute_query("""
        MATCH (n:_Entity_)
        RETURN n.nodeId AS nodeId, n.textEmbedding AS emb,
               labels(n)[0] AS node_type
        ORDER BY n.nodeId
    """)
    nodes = pd.DataFrame([x.data() for x in r.records])
    print(f"Nodes: {len(nodes)}")

    print("Pulling edges...")
    r = driver.execute_query("""
        MATCH (s:_Entity_)-[e]->(t:_Entity_)
        RETURN s.nodeId AS src, t.nodeId AS tgt
    """)
    edges = pd.DataFrame([x.data() for x in r.records])
    print(f"Edges: {len(edges)}")
    driver.close()

    print("\nClass distribution:")
    for name, idx in TYPE_TO_IDX.items():
        count = (nodes["node_type"] == name).sum()
        print(f"  {name}: {count}")

    id2idx = {nid: i for i, nid in enumerate(nodes["nodeId"])}
    x = torch.tensor(np.stack(nodes["emb"].values), dtype=torch.float)
    y = torch.tensor(
        [TYPE_TO_IDX.get(t, -1) for t in nodes["node_type"]], dtype=torch.long
    )

    valid = edges["src"].isin(id2idx) & edges["tgt"].isin(id2idx)
    edges = edges[valid]
    ei = torch.tensor([
        [id2idx[s] for s in edges["src"]],
        [id2idx[t] for t in edges["tgt"]]
    ], dtype=torch.long)

    data = Data(x=x, edge_index=ei, y=y)
    data = RandomNodeSplit("train_rest", num_val=0.15, num_test=0.15)(data)
    print(f"\nSplits — Train: {data.train_mask.sum()} | "
          f"Val: {data.val_mask.sum()} | Test: {data.test_mask.sum()}")

    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    torch.save(data, SAVE)
    print(f"Saved: {SAVE}")


if __name__ == "__main__":
    build()
