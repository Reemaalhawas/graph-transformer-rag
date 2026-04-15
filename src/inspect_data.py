import os
import pickle
import json
from pprint import pprint

import torch
import pandas as pd


BASE = os.path.join("data", "prime")
PROCESSED_DIR = os.path.join(BASE, "processed")
SPLIT_DIR = os.path.join(BASE, "split")
QA_DIR = os.path.join(BASE, "stark_qa")

# If you want a full dump of node_info to a JSON file, set this to True
DUMP_NODE_INFO_TO_JSON = False
NODE_INFO_JSON_PATH = os.path.join(PROCESSED_DIR, "node_info_dump.json")


def inspect_qa():
    print("=" * 80)
    print("1) QA DATA (stark_qa/)")
    print("=" * 80)

    qa_path = os.path.join(QA_DIR, "stark_qa.csv")
    if not os.path.exists(qa_path):
        print(f"stark_qa.csv NOT FOUND at {qa_path}")
        return

    df = pd.read_csv(qa_path)

    print("Columns in stark_qa.csv:")
    print(list(df.columns))
    print()

    print("First 3 rows:")
    print(df.head(3).to_string(index=False))
    print()

    row = df.iloc[0]
    print("Example question (synthetic):")
    print("  id         :", row["id"])
    print("  query      :", row["query"])
    print("  answer_ids :", row["answer_ids"])
    print()

    # Human-generated eval file (if exists)
    eval_path = os.path.join(QA_DIR, "stark_qa_human_generated_eval.csv")
    if os.path.exists(eval_path):
        df_eval = pd.read_csv(eval_path)
        print("Human eval file found: stark_qa_human_generated_eval.csv")
        print("  Columns:", list(df_eval.columns))
        print("  Rows   :", len(df_eval))
        print()

        row_eval = df_eval.iloc[0]
        print("Example human-generated question:")
        print("  id         :", row_eval.get("id"))
        print("  query      :", row_eval.get("query"))
        print("  answer_ids :", row_eval.get("answer_ids"))
        print()
    else:
        print("No stark_qa_human_generated_eval.csv found.")
        print()


def inspect_splits():
    print("=" * 80)
    print("2) SPLITS (train/val/test indices)")
    print("=" * 80)

    def read_index(name):
        path = os.path.join(SPLIT_DIR, f"{name}.index")
        if not os.path.exists(path):
            print(f"  {name}.index not found at {path}")
            return []
        with open(path) as f:
            return [int(x) for x in f.read().splitlines()]

    train_ids = read_index("train")
    val_ids = read_index("val")
    test_ids = read_index("test")

    print(f"Train size: {len(train_ids)}")
    print(f"Val size  : {len(val_ids)}")
    print(f"Test size : {len(test_ids)}")
    print()

    print("First 5 train IDs:", train_ids[:5])
    print("First 5 val IDs  :", val_ids[:5])
    print("First 5 test IDs :", test_ids[:5])
    print()


def inspect_processed():
    print("=" * 80)
    print("3) GRAPH (processed/)")
    print("=" * 80)

    # --- Load tensors ---
    edge_index = torch.load(os.path.join(PROCESSED_DIR, "edge_index.pt"))
    edge_types = torch.load(os.path.join(PROCESSED_DIR, "edge_types.pt"))
    node_types = torch.load(os.path.join(PROCESSED_DIR, "node_types.pt"))

    print("edge_index shape:", edge_index.shape)   # [2, num_edges]
    print("edge_types shape:", edge_types.shape)   # [num_edges]
    print("node_types shape:", node_types.shape)   # [num_nodes]
    print()

    # --- Load dicts (metadata) ---
    with open(os.path.join(PROCESSED_DIR, "node_info.pkl"), "rb") as f:
        node_info = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "node_type_dict.pkl"), "rb") as f:
        node_type_dict = pickle.load(f)

    with open(os.path.join(PROCESSED_DIR, "edge_type_dict.pkl"), "rb") as f:
        edge_type_dict = pickle.load(f)

    print(f"Number of nodes in node_info: {len(node_info)}")
    print()

    print("Node type dict (id -> type_name):")
    pprint(node_type_dict)
    print()

    print("Edge type dict (id -> rel_type):")
    pprint(edge_type_dict)
    print()

    # --- Show keys and full info for a few example nodes ---
    print("Example nodes from node_info (showing ALL fields):")
    print("-" * 80)
    for i, (node_id, info) in enumerate(node_info.items()):
        print(f"node_id: {node_id}")
        print("  keys:", list(info.keys()))
        print("  full info dict:")
        pprint(info)  # prints all fields for this node
        print("-" * 80)
        if i == 2:  # first 3 nodes
            break

    # --- Optional: dump full node_info to JSON for browsing in VS Code ---
    if DUMP_NODE_INFO_TO_JSON:
        # node_info keys might be int, convert to str for JSON
        node_info_str_keys = {str(k): v for k, v in node_info.items()}
        with open(NODE_INFO_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(node_info_str_keys, f, ensure_ascii=False, indent=2)
        print()
        print("Full node_info has been dumped to JSON:")
        print("  ", NODE_INFO_JSON_PATH)


def main():
    inspect_qa()
    inspect_splits()
    inspect_processed()


if __name__ == "__main__":
    main()
