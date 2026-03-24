"""
Build and save pre-computed subgraphs from Neo4j + OpenAI embeddings.
Saves each Q&A pair as an individual .pt file ready for train.py.

Usage:
    python build_subgraphs.py
    python build_subgraphs.py --start 0 --end 1000   # process a slice
    python build_subgraphs.py --skip_existing         # resume interrupted run
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from langchain_openai import OpenAIEmbeddings
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv('db.env', override=True)

NEO4J_URI      = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY,
)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def chunks(xs, n=500):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def embed(doc_list: List[str]) -> List:
    embeddings = []
    for batch in chunks(doc_list):
        embeddings.extend(embedding_model.embed_documents(batch))
    return embeddings


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def get_nodes_by_vector_search(prompt: str, driver: Driver, k: int = 4) -> List:
    # Embed query in Python first (avoids needing genai plugin in Neo4j)
    query_embedding = embedding_model.embed_query(prompt)
    res = driver.execute_query("""
        CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
        RETURN node.nodeId AS nodeId
        """,
        parameters_={
            "index": "text_embeddings",
            "k": k,
            "query_embedding": query_embedding,
        })
    return [rec.data()['nodeId'] for rec in res.records]


def get_subgraph_rels(node_ids: List, driver: Driver) -> pd.DataFrame:
    res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        WITH collect(node) AS sources, collect(node) AS targets
        UNWIND sources as source
        UNWIND targets as target
        WITH source, target
        WHERE source > target
        MATCH (source)-[rl]->{0,2}(target)
        UNWIND rl AS r
        WITH DISTINCT r
        MATCH (m)-[r]->(n)
        RETURN
          m.nodeId AS src,
          n.nodeId AS tgt,
          n.name + ' - ' + type(r) + ' -> ' + m.name AS text
        """,
        parameters_={"nodeIds": node_ids})
    return pd.DataFrame([rec.data() for rec in res.records])


def get_node_df(node_ids: List, rel_df: pd.DataFrame, driver: Driver) -> pd.DataFrame:
    all_ids = set(node_ids)
    if rel_df.shape[0] > 0:
        all_ids.update(rel_df['src'].tolist())
        all_ids.update(rel_df['tgt'].tolist())
    res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(n:_Entity_ {nodeId:nodeId})
        RETURN n.nodeId AS nodeId, n.name AS name,
               n.textEmbedding AS textEmbedding, n.details AS details
        """,
        parameters_={"nodeIds": list(all_ids)})
    return pd.DataFrame([rec.data() for rec in res.records])


# ---------------------------------------------------------------------------
# Build one Data object
# ---------------------------------------------------------------------------

def build_data(query: str, answer_ids: List[int], driver: Driver, debug: bool = False) -> Data:
    """Retrieve subgraph from Neo4j and build a PyG Data object."""

    # 1. Vector search → seed nodes
    init_node_ids = get_nodes_by_vector_search(query, driver)
    if debug:
        print(f"  DEBUG init_node_ids: {init_node_ids}")

    # 2. Subgraph relations
    rel_df = get_subgraph_rels(init_node_ids, driver)
    if debug:
        print(f"  DEBUG rel_df shape: {rel_df.shape}")
    if rel_df.shape[0] == 0:
        return None

    # 3. All nodes in subgraph
    node_df = get_node_df(init_node_ids, rel_df, driver)
    if node_df.shape[0] == 0:
        return None

    # 4. Embed edges
    rel_df['textEmbedding'] = embed(rel_df['text'].tolist())

    # 5. Re-index nodes for edge_index
    node_df = node_df.reset_index(drop=True)
    id_to_idx = {row['nodeId']: i for i, row in node_df.iterrows()}

    src_idx = [id_to_idx[s] for s in rel_df['src'] if s in id_to_idx]
    tgt_idx = [id_to_idx[t] for t in rel_df['tgt'] if t in id_to_idx]

    # Filter rel_df to only edges where both nodes are in node_df
    valid = [(s, t, e) for s, t, e in zip(
        rel_df['src'], rel_df['tgt'], rel_df['textEmbedding'])
        if s in id_to_idx and t in id_to_idx]

    if not valid:
        return None

    src_idx = [id_to_idx[s] for s, t, e in valid]
    tgt_idx = [id_to_idx[t] for s, t, e in valid]
    edge_embs = [e for s, t, e in valid]

    # 6. Tensors
    x = torch.tensor(
        np.stack(node_df['textEmbedding'].tolist()), dtype=torch.float)
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_embs), dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        question=query,
        answer=answer_ids,   # list of integer node IDs
        desc='',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end',   type=int, default=None)
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip questions whose .pt file already exists')
    args = parser.parse_args()

    # Load STaRK-QA dataset
    print("Loading STaRK-QA Prime dataset...")
    from stark_qa import load_qa
    qa_dataset = load_qa('prime')
    df = qa_dataset.data

    end = args.end if args.end else len(df)
    df = df.iloc[args.start:end].reset_index(drop=True)
    print(f"Processing {len(df)} questions (indices {args.start} to {end})")

    already = set(os.listdir(OUT_DIR))
    skipped = 0
    saved = 0
    failed = 0

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            global_idx = args.start + i
            fname = f"subgraph_{global_idx:05d}.pt"

            if args.skip_existing and fname in already:
                skipped += 1
                continue

            try:
                query      = row['query']
                answer_ids = list(row['answer_ids'])

                data = build_data(query, answer_ids, driver)

                if data is None:
                    print(f"  [{global_idx}] Skipped — empty subgraph")
                    failed += 1
                    continue

                torch.save(data, os.path.join(OUT_DIR, fname))
                saved += 1

            except Exception as e:
                print(f"  [{global_idx}] Error: {e}")
                failed += 1
                continue

            if (i + 1) % 100 == 0:
                total_pt = len([f for f in os.listdir(OUT_DIR) if f.endswith('.pt')])
                print(f"\nProgress: {i+1}/{len(df)} | Saved: {saved} | "
                      f"Skipped: {skipped} | Failed: {failed} | "
                      f"Total .pt: {total_pt}")

    print(f"\nDone!")
    print(f"  Saved:   {saved}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Total .pt files: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.pt')])}")


if __name__ == '__main__':
    main()
