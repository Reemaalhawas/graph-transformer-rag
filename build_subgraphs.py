"""
Build and save pre-computed subgraphs from Neo4j + OpenAI embeddings.
Uses PCST (Prize-Collecting Steiner Tree) to prune subgraphs, following
the NVIDIA G-Retriever reference implementation.
Saves each Q&A pair as an individual .pt file ready for train.py.

Usage:
    python build_subgraphs.py
    python build_subgraphs.py --start 0 --end 1000   # process a slice
    python build_subgraphs.py --skip_existing         # resume interrupted run
"""

import os
import ast
import argparse
import torch
import numpy as np
import pandas as pd
import pcst_fast
from typing import List, Tuple
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

def get_nodes_by_vector_search(prompt: str, driver: Driver, k: int = 5) -> Tuple[List, List]:
    """Returns (node_ids, query_embedding)."""
    query_embedding = embedding_model.embed_query(prompt)
    res = driver.execute_query("""
        CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
        RETURN node.nodeId AS nodeId
        """,
        parameters_={
            "index": "textembeddings",
            "k": k,
            "query_embedding": query_embedding,
        })
    node_ids = [rec.data()['nodeId'] for rec in res.records]
    return node_ids, query_embedding


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
# PCST pruning
# ---------------------------------------------------------------------------

def apply_pcst(node_df: pd.DataFrame, rel_df: pd.DataFrame,
               query_embedding: List[float],
               topk_node_ids: List) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prune the candidate subgraph using Prize-Collecting Steiner Tree.

    Node prizes  = cosine similarity between node embedding and query embedding.
                   Seed nodes (from vector search) get a bonus prize of 1.0.
    Edge costs   = uniform 1.0 per edge.
    """
    node_df = node_df.reset_index(drop=True)
    id_to_idx = {row['nodeId']: i for i, row in node_df.iterrows()}

    # Node prizes
    node_embs = np.stack(node_df['textEmbedding'].tolist()).astype(np.float64)
    q_emb = np.array(query_embedding, dtype=np.float64)
    norms = np.linalg.norm(node_embs, axis=1)
    q_norm = np.linalg.norm(q_emb)
    cos_sim = (node_embs @ q_emb) / (norms * q_norm + 1e-8)
    prizes = np.clip(cos_sim, 0, None)

    # Bonus for seed nodes from vector search
    for nid in topk_node_ids:
        if nid in id_to_idx:
            prizes[id_to_idx[nid]] += 1.0

    # Build edge arrays (only edges where both endpoints are in node_df)
    edges = []
    rel_indices = []
    for i, row in rel_df.iterrows():
        s, t = row['src'], row['tgt']
        if s in id_to_idx and t in id_to_idx:
            edges.append([id_to_idx[s], id_to_idx[t]])
            rel_indices.append(i)

    if not edges:
        return node_df, rel_df

    edges_arr = np.array(edges, dtype=np.int64)
    costs = np.ones(len(edges_arr), dtype=np.float64)

    # Run PCST
    selected_nodes, selected_edges = pcst_fast.pcst_fast(
        edges_arr, prizes, costs, -1, 1, 'strong', 0
    )

    pruned_node_df = node_df.iloc[selected_nodes].reset_index(drop=True)
    selected_rel_indices = [rel_indices[i] for i in selected_edges]
    pruned_rel_df = rel_df.iloc[selected_rel_indices].reset_index(drop=True)

    return pruned_node_df, pruned_rel_df


# ---------------------------------------------------------------------------
# STaRK SKB helpers
# ---------------------------------------------------------------------------

def get_entity_name_from_skb(skb, entity_id: int) -> str:
    """Get entity name from STaRK SKB by entity ID."""
    try:
        attrs = skb.node_attr_dict[entity_id]
        if isinstance(attrs, dict):
            name = attrs.get('name') or attrs.get('title') or attrs.get('id', '')
            if name:
                return str(name)
    except Exception:
        pass
    try:
        info = skb.node_info[entity_id]
        if isinstance(info, dict):
            name = info.get('name') or info.get('title') or ''
            if name:
                return str(name)
    except Exception:
        pass
    try:
        cand_list = list(skb.candidate_ids)
        if entity_id in cand_list:
            idx = cand_list.index(entity_id)
            doc = skb.get_doc_info(idx, add_rel=False, compact=True)
            first = str(doc).strip().split('\n')[0]
            if ':' in first:
                return first.split(':', 1)[1].strip()
            if first and not first.startswith('--'):
                return first
    except Exception:
        pass
    return ''


def get_answer_names_from_skb(answer_ids: List[int], skb) -> str:
    """Get '|'-separated entity names from STaRK SKB.

    IMPORTANT: STaRK answer_ids are entity IDs in the STaRK/PrimeKG space,
    NOT Neo4j nodeIds. Always use this function for label generation.
    """
    names = [get_entity_name_from_skb(skb, aid) for aid in answer_ids]
    names = [n for n in names if n]
    return ' | '.join(names)


# ---------------------------------------------------------------------------
# Build one Data object
# ---------------------------------------------------------------------------

def build_data(query: str, answer_ids: List[int], driver: Driver,
               skb=None, debug: bool = False) -> Data:
    """Retrieve subgraph from Neo4j, apply PCST, and build a PyG Data object."""

    # 1. Vector search → seed nodes + query embedding
    init_node_ids, query_embedding = get_nodes_by_vector_search(query, driver)
    if debug:
        print(f"  DEBUG init_node_ids: {init_node_ids}")

    # 2. Expand to 2-hop subgraph
    rel_df = get_subgraph_rels(init_node_ids, driver)
    if debug:
        print(f"  DEBUG rel_df shape: {rel_df.shape}")
    if rel_df.shape[0] == 0:
        return None

    # 3. Fetch all nodes in candidate subgraph
    node_df = get_node_df(init_node_ids, rel_df, driver)
    if node_df.shape[0] == 0:
        return None

    # 4. Embed edges
    rel_df = rel_df.copy()
    rel_df['textEmbedding'] = embed(rel_df['text'].tolist())

    # 5. Apply PCST pruning
    node_df, rel_df = apply_pcst(node_df, rel_df, query_embedding, init_node_ids)
    if node_df.shape[0] == 0 or rel_df.shape[0] == 0:
        return None

    # 6. Re-index nodes for edge_index
    node_df = node_df.reset_index(drop=True)
    id_to_idx = {row['nodeId']: i for i, row in node_df.iterrows()}

    valid = [(s, t, e) for s, t, e in zip(
        rel_df['src'], rel_df['tgt'], rel_df['textEmbedding'])
        if s in id_to_idx and t in id_to_idx]

    if not valid:
        return None

    src_idx  = [id_to_idx[s] for s, t, e in valid]
    tgt_idx  = [id_to_idx[t] for s, t, e in valid]
    edge_embs = [e for s, t, e in valid]

    # 7. Tensors
    x = torch.tensor(
        np.stack(node_df['textEmbedding'].tolist()), dtype=torch.float)
    edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)
    edge_attr  = torch.tensor(np.array(edge_embs), dtype=torch.float)

    # Label: entity names from STaRK SKB (not Neo4j — different ID spaces!)
    if skb is None:
        raise ValueError("skb (STaRK knowledge base) must be provided for label generation")
    label = get_answer_names_from_skb(answer_ids, skb)
    if not label:
        return None

    # Desc: node names as context string for the LLM prompt
    desc_parts = []
    for _, row in node_df.iterrows():
        name = row.get('name') or ''
        if name:
            details = row.get('details') or ''
            desc_parts.append(f"{name}: {details[:80]}" if details else name)
    desc = '; '.join(desc_parts[:30])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        question=query,
        label=label,
        desc=desc,
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

    # Load STaRK-QA dataset and SKB
    print("Loading STaRK-QA Prime dataset...")
    from stark_qa import load_qa, load_skb
    qa_dataset = load_qa('prime')
    df = qa_dataset.data

    print("Loading STaRK SKB for label lookup...")
    skb = load_skb('prime', download_processed=True)

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
                query = row['query']
                raw_ids = row['answer_ids']
                if isinstance(raw_ids, str):
                    answer_ids = [int(x) for x in ast.literal_eval(raw_ids)]
                else:
                    answer_ids = [int(x) for x in raw_ids]

                data = build_data(query, answer_ids, driver, skb=skb)

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
