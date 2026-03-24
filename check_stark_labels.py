"""
Diagnose the label mismatch: compare STaRK ground truth entity names
against what's currently stored in .pt files.

Usage:
    python check_stark_labels.py
"""

import os
import torch
from stark_qa import load_qa, load_skb

print("Loading STaRK QA dataset...")
qa_dataset = load_qa('prime')
df = qa_dataset.data
print(f"QA rows: {len(df)}")

print("\nLoading STaRK SKB...")
skb = load_skb('prime', download_processed=True)
print(f"SKB type: {type(skb).__name__}")

# Show available methods on the SKB
methods = [x for x in dir(skb) if not x.startswith('_')]
print(f"SKB methods: {methods[:30]}\n")

# ── Try to get entity names for the first 5 questions ──────────────────────
DATA_DIR = 'data/processed'

for i in range(5):
    row = df.iloc[i]
    answer_ids = list(row['answer_ids'])

    print(f"=== STaRK index {i} ===")
    print(f"Query:      {row['query'][:90]}")
    print(f"answer_ids: {answer_ids}")

    # Try multiple SKB API approaches
    names_found = []
    for aid in answer_ids[:3]:
        name = None
        # Approach 1: get_doc_info
        try:
            doc = skb.get_doc_info(aid, add_rel=False, compact=True)
            name = str(doc).strip().split('\n')[0][:80]
        except Exception:
            pass
        # Approach 2: direct indexing
        if name is None:
            try:
                entity = skb[aid]
                name = str(entity)[:80]
            except Exception:
                pass
        # Approach 3: node_to_str
        if name is None:
            try:
                name = skb.node_to_str(aid)[:80]
            except Exception:
                pass
        names_found.append(f"{aid}→{name}" if name else f"{aid}→??")

    print(f"SKB names:  {names_found}")

    # What's in the .pt file?
    fname = f"subgraph_{i:05d}.pt"
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        data = torch.load(fpath, weights_only=False)
        print(f".pt label:  {getattr(data, 'label', 'NO LABEL')}")
    else:
        print(f".pt label:  (file not found)")
    print()
