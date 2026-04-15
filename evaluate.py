"""
Standalone evaluation script.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import ast
import os

import yaml
import torch
import torch.nn.functional as F

from src.loaders.prime_skb_loader import load_prime_skb
from src.loaders.prime_qa_loader import load_prime_qa
from src.models import GraphTransformerGCN
from src.utils.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--k",          type=int, default=10)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ data
    graph_raw       = load_prime_skb(cfg["data"]["data_root"])
    qa_df, splits   = load_prime_qa(cfg["data"]["data_root"])

    node_emb_raw    = torch.load(
        os.path.join(cfg["data"]["data_root"], cfg["data"]["node_emb_file"]),
        weights_only=False,
    )
    node_ids_tensor = torch.load(
        os.path.join(cfg["data"]["data_root"], cfg["data"]["node_ids_file"]),
        weights_only=False,
    )
    node_ids        = node_ids_tensor.tolist()
    node_id_to_row  = {nid: i for i, nid in enumerate(node_ids)}
    node_feat_d     = node_emb_raw.size(1)

    query_emb_path = os.path.join(
        cfg["data"]["emb_root"],
        cfg["data"]["emb_model"],
        "query",
        "query_emb_dict.pt",
    )
    query_emb_dict = torch.load(query_emb_path, weights_only=False) if os.path.exists(query_emb_path) else {}
    query_dim      = cfg["model"]["query_feat_dim"]

    num_edge_types = int(graph_raw["edge_types"].max().item() + 1) if graph_raw.get("edge_types") is not None else 0

    # ----------------------------------------------------------------- model
    mcfg = cfg["model"]
    model = GraphTransformerGCN(
        node_feat_dim=node_feat_d,
        query_feat_dim=query_dim,
        hidden_dim=mcfg["hidden_dim"],
        out_dim=mcfg["output_dim"],
        num_gcn_layers=mcfg["num_gcn_layers"],
        num_transformer_layers=mcfg["num_transformer_layers"],
        num_heads=mcfg["num_heads"],
        num_edge_types=num_edge_types,
        edge_dim=mcfg["edge_dim"],
        dropout=mcfg["dropout"],
    ).to(device)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"[WARN] Checkpoint not found: {args.checkpoint}. Using random weights.")

    model.eval()

    # --------------------------------------------------------- encode graph
    x          = node_emb_raw.to(device)
    edge_index = graph_raw["edge_index"].to(device)
    edge_types = graph_raw["edge_types"].to(device) if graph_raw.get("edge_types") is not None else None

    with torch.no_grad():
        node_emb = model.encode_graph(x, edge_index, edge_types)  # [N, out_dim]

    # ------------------------------------------------------- rank + metrics
    split_ids     = splits[args.split]
    rows          = qa_df[qa_df["id"].isin(split_ids)]
    ranked_lists  = []
    ground_truths = []

    for _, row in rows.iterrows():
        qid   = str(row["id"])
        q_emb = query_emb_dict.get(qid, torch.randn(query_dim)).float().to(device)
        with torch.no_grad():
            q_enc = model.encode_query(q_emb.unsqueeze(0)).squeeze(0)
            q_enc = F.normalize(q_enc, dim=-1)
            scores = (node_emb @ q_enc).cpu()

        ranked = torch.argsort(scores, descending=True).tolist()

        raw = row.get("answer_ids", row.get("answer", "[]"))
        try:
            ans_ids = ast.literal_eval(str(raw))
        except Exception:
            ans_ids = []
        pos_rows = [node_id_to_row[nid] for nid in ans_ids if nid in node_id_to_row]

        if not pos_rows:
            continue
        ranked_lists.append(ranked)
        ground_truths.append(pos_rows)

    metrics = compute_metrics(ranked_lists, ground_truths, k_values=[1, 5, args.k, 20])

    print(f"\n{'='*40}")
    print(f"Split: {args.split}  |  Queries evaluated: {len(ranked_lists)}")
    print(f"{'='*40}")
    for name, val in metrics.items():
        print(f"  {name:<18}: {val:.4f}")


if __name__ == "__main__":
    main()
