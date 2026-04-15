import os
import pickle
from typing import Dict, Any

import torch


def load_prime_skb(data_root: str = "data/prime") -> Dict[str, Any]:
    """
    Load the STaRK-Prime SKB (graph) from the local processed/ folder.

    Returns a dict with:
      - edge_index        : LongTensor [2, num_edges]
      - edge_types        : LongTensor [num_edges]
      - node_types        : LongTensor [num_nodes]
      - node_info         : dict with metadata per node
      - node_type_dict    : dict[int -> str]
      - edge_type_dict    : dict[int -> str]
    """
    processed_dir = os.path.join(data_root, "processed")

    # Tensors
    edge_index = torch.load(os.path.join(processed_dir, "edge_index.pt"))
    edge_types = torch.load(os.path.join(processed_dir, "edge_types.pt"))
    node_types = torch.load(os.path.join(processed_dir, "node_types.pt"))

    # Dicts
    with open(os.path.join(processed_dir, "node_info.pkl"), "rb") as f:
        node_info = pickle.load(f)

    with open(os.path.join(processed_dir, "node_type_dict.pkl"), "rb") as f:
        node_type_dict = pickle.load(f)

    with open(os.path.join(processed_dir, "edge_type_dict.pkl"), "rb") as f:
        edge_type_dict = pickle.load(f)

    return {
        "edge_index": edge_index,
        "edge_types": edge_types,
        "node_types": node_types,
        "node_info": node_info,
        "node_type_dict": node_type_dict,
        "edge_type_dict": edge_type_dict,
    }


if __name__ == "__main__":
    # Simple sanity check when you run this file directly
    data = load_prime_skb()

    edge_index = data["edge_index"]
    edge_types = data["edge_types"]
    node_types = data["node_types"]
    node_info = data["node_info"]

    print("Number of nodes:", len(node_info))
    print("edge_index shape:", edge_index.shape)
    print("edge_types shape:", edge_types.shape)
    print("node_types shape:", node_types.shape)

    # Show a couple of nodes
    print("\nFirst 3 node_info entries:")
    for i, (node_id, info) in enumerate(node_info.items()):
        print("node_id:", node_id)
        print("  name:", info.get("name"))
        print("  type:", info.get("type"))
        print()
        if i == 2:
            break
