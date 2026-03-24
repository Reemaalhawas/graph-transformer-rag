"""
Build and save pre-computed subgraphs from Neo4j + OpenAI embeddings.
Uses PCST (Prize-Collecting Steiner Tree) to prune subgraphs, following
the NVIDIA G-Retriever reference implementation exactly.
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
# PCST prize assignment  (matches NVIDIA reference exactly)
# ---------------------------------------------------------------------------

def assign_prizes(
    node_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    query_embedding: List[float],
    topk_node_ids: List,
    topk: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Node prizes : linspace(4, 0) for the top-k seed nodes (by vector search rank).
    Edge prizes : top-k edges by cosine similarity to query get descending int prizes.
    Matches assign_prizes_topk() in the NVIDIA reference.
    """
    id_to_idx = {row['nodeId']: i for i, row in node_df.iterrows()}

    # --- node prizes ---
    n_prizes = torch.zeros(len(node_df))
    top_local = [id_to_idx[nid] for nid in topk_node_ids if nid in id_to_idx]
    if top_local:
        n_prizes[top_local] = torch.linspace(4, 0, steps=len(top_local)).float()

    # --- edge prizes ---
    q_emb = np.array(query_embedding, dtype=np.float64)
    edge_embs = np.stack(rel_df['textEmbedding'].tolist()).astype(np.float64)
    q_norm = np.linalg.norm(q_emb)
    e_norms = np.linalg.norm(edge_embs, axis=1)
    cos_sims = (edge_embs @ q_emb) / (e_norms * q_norm + 1e-8)

    k_edges = min(topk, len(cos_sims))
    top_edge_idx = np.argsort(cos_sims)[::-1][:k_edges]

    e_prizes = torch.zeros(len(rel_df))
    for rank, eidx in enumerate(top_edge_idx):
        e_prizes[int(eidx)] = float(k_edges - rank)

    return n_prizes, e_prizes


# ---------------------------------------------------------------------------
# PCST algorithm  (matches compute_pcst() in the NVIDIA reference exactly)
# ---------------------------------------------------------------------------

def compute_pcst(
    base_edge_index: torch.Tensor,
    num_nodes: int,
    n_prizes: torch.Tensor,
    e_prizes: torch.Tensor,
    cost_e: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prize-Collecting Steiner Tree with virtual-node trick for high-prize edges.
    Uses pruning='gw' and cost_e=0.5, exactly as in the NVIDIA reference.
    """
    root = -1
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}

    for i, (src, dst) in enumerate(base_edge_index.t().tolist()):
        prize_e = float(e_prizes[i])
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes.numpy(), np.array(virtual_n_prizes)])
    num_real_edges = len(edges)

    if len(virtual_costs) > 0:
        all_costs = np.array(costs + virtual_costs)
        all_edges = np.array(edges + virtual_edges, dtype=np.int64)
    else:
        all_costs = np.array(costs)
        all_edges = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)

    if len(all_edges) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    vertices, sel_edges = pcst_fast.pcst_fast(
        all_edges, prizes, all_costs, root, num_clusters, pruning, verbosity_level
    )

    selected_nodes = vertices[vertices < num_nodes]
    selected_edges = [mapping_e[e] for e in sel_edges if e < num_real_edges]

    virtual_vertices = vertices[vertices >= num_nodes]
    if len(virtual_vertices) > 0:
        virtual_edge_ids = [mapping_n[v] for v in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edge_ids)
    else:
        selected_edges = np.array(selected_edges, dtype=np.int64)

    # Add any nodes implied by selected edges
    if len(selected_edges) > 0:
        edge_nodes = base_edge_index[:, selected_edges].numpy().flatten()
        selected_nodes = np.unique(np.concatenate([selected_nodes, edge_nodes]))

    return selected_nodes, selected_edges


# ---------------------------------------------------------------------------
# STaRK SKB helpers
# ---------------------------------------------------------------------------

def get_entity_name_from_skb(skb, entity_id: int) -> str:
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
    """Get '|'-separated entity names from STaRK SKB."""
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

    # Drop nodes with missing embeddings
    node_df = node_df[node_df['textEmbedding'].apply(
        lambda e: e is not None and len(e) > 0)].reset_index(drop=True)
    if node_df.shape[0] == 0:
        return None

    # 4. Embed edges
    rel_df = rel_df.copy()
    rel_df['textEmbedding'] = embed(rel_df['text'].tolist())

    # 5. Build base edge_index (local integer indices)
    node_df = node_df.reset_index(drop=True)
    id_to_idx = {row['nodeId']: i for i, row in node_df.iterrows()}

    valid_mask = rel_df.apply(
        lambda r: r['src'] in id_to_idx and r['tgt'] in id_to_idx, axis=1)
    rel_df = rel_df[valid_mask].reset_index(drop=True)
    if rel_df.shape[0] == 0:
        return None

    src_idx = [id_to_idx[s] for s in rel_df['src']]
    tgt_idx = [id_to_idx[t] for t in rel_df['tgt']]
    base_edge_index = torch.tensor([src_idx, tgt_idx], dtype=torch.long)

    # 6. Assign prizes (NVIDIA reference logic)
    n_prizes, e_prizes = assign_prizes(
        node_df, rel_df, query_embedding, init_node_ids)

    # 7. Run PCST (NVIDIA reference logic)
    selected_nodes, selected_edges = compute_pcst(
        base_edge_index, len(node_df), n_prizes, e_prizes)

    if len(selected_nodes) == 0:
        return None

    # 8. Build pruned tensors
    pruned_node_df = node_df.iloc[selected_nodes].reset_index(drop=True)

    mapping = {int(n): i for i, n in enumerate(selected_nodes)}
    if len(selected_edges) > 0:
        pruned_ei = base_edge_index[:, selected_edges]
        src_pruned = [mapping[int(s)] for s in pruned_ei[0].tolist()]
        tgt_pruned = [mapping[int(t)] for t in pruned_ei[1].tolist()]
        edge_embs = [rel_df.iloc[int(e)]['textEmbedding'] for e in selected_edges]
        edge_index = torch.tensor([src_pruned, tgt_pruned], dtype=torch.long)
        edge_attr  = torch.tensor(np.array(edge_embs), dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1536), dtype=torch.float)

    x = torch.tensor(
        np.stack(pruned_node_df['textEmbedding'].tolist()), dtype=torch.float)

    # Label
    if skb is None:
        raise ValueError("skb must be provided")
    label = get_answer_names_from_skb(answer_ids, skb)
    if not label:
        return None

    # Desc
    desc_parts = []
    for _, row in pruned_node_df.iterrows():
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
