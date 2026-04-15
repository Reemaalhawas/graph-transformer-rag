import os
from typing import Dict, List, Tuple

import pandas as pd


def load_prime_qa(data_root: str = "data/prime") -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """
    Load STaRK-Prime QA data and train/val/test splits.

    Returns:
      - df: DataFrame with columns like ["id", "query", "answer", ...]
      - splits: dict with keys "train", "val", "test" mapping to lists of ids
    """
    qa_path = os.path.join(data_root, "stark_qa", "stark_qa.csv")
    df = pd.read_csv(qa_path)

    split_dir = os.path.join(data_root, "split")

    def read_index(name: str):
        path = os.path.join(split_dir, f"{name}.index")
        with open(path) as f:
            return [int(x) for x in f.read().splitlines()]

    splits = {
        "train": read_index("train"),
        "val": read_index("val"),
        "test": read_index("test"),
    }

    return df, splits


if __name__ == "__main__":
    df, splits = load_prime_qa()
    print("Total questions:", len(df))
    print("Train/Val/Test sizes:",
          len(splits["train"]), len(splits["val"]), len(splits["test"]))

    print("\nExample question:")
    row = df.iloc[0]
    print("id:", row["id"])
    print("query:", row["query"])
    print("answer (raw):", row["answer"])
