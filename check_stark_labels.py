"""
Diagnose label mismatch: show STaRK ground truth entity names
vs what's currently stored in .pt files.

Usage:
    python check_stark_labels.py
"""

import os
import ast
import torch
from stark_qa import load_qa, load_skb

print("Loading STaRK QA dataset...")
qa_dataset = load_qa('prime')
df = qa_dataset.data
print(f"QA rows: {len(df)}")

print("\nLoading STaRK SKB...")
skb = load_skb('prime', download_processed=True)
print(f"SKB type: {type(skb).__name__}\n")

DATA_DIR = 'data/processed'


def parse_answer_ids(raw):
    """Parse answer_ids regardless of how pandas stored them."""
    if isinstance(raw, str):
        return [int(x) for x in ast.literal_eval(raw)]
    try:
        return [int(x) for x in raw]
    except Exception:
        return []


def get_entity_name(skb, entity_id: int) -> str:
    """Get entity name from STaRK SKB by entity ID."""
    # Method 1: node_attr_dict (direct dict lookup by entity ID)
    try:
        attrs = skb.node_attr_dict.get(entity_id) or skb.node_attr_dict[entity_id]
        if isinstance(attrs, dict):
            name = attrs.get('name') or attrs.get('title') or attrs.get('id', '')
            if name:
                return str(name)
    except Exception:
        pass

    # Method 2: node_info (alternative attribute store)
    try:
        info = skb.node_info.get(entity_id) or skb.node_info[entity_id]
        if isinstance(info, dict):
            name = info.get('name') or info.get('title') or ''
            if name:
                return str(name)
    except Exception:
        pass

    # Method 3: get_doc_info by sequential index in candidate_ids
    try:
        cand_list = list(skb.candidate_ids)
        if entity_id in cand_list:
            idx = cand_list.index(entity_id)
            doc = skb.get_doc_info(idx, add_rel=False, compact=True)
            first = str(doc).strip().split('\n')[0]
            # Format is usually "Name: <name>" or just the name
            if ':' in first:
                return first.split(':', 1)[1].strip()
            return first
    except Exception:
        pass

    return ''


# Show first 5 examples
for i in range(5):
    row = df.iloc[i]
    answer_ids = parse_answer_ids(row['answer_ids'])

    print(f"=== STaRK index {i} ===")
    print(f"Query:      {row['query'][:90]}")
    print(f"answer_ids: {answer_ids}")

    # Try to get entity names
    names = []
    for aid in answer_ids[:5]:
        name = get_entity_name(skb, aid)
        names.append(f"{aid}→{name!r}" if name else f"{aid}→??")
    print(f"SKB names:  {names}")

    # Inspect the raw node_attr_dict entry for first answer ID
    if answer_ids:
        aid = answer_ids[0]
        try:
            raw_attrs = skb.node_attr_dict.get(aid, skb.node_attr_dict[aid] if aid in skb.node_attr_dict else 'NOT FOUND')
            print(f"node_attr_dict[{aid}]: {str(raw_attrs)[:200]}")
        except Exception as e:
            print(f"node_attr_dict[{aid}]: ERROR {e}")

    # .pt file label
    fname = f"subgraph_{i:05d}.pt"
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        data = torch.load(fpath, weights_only=False)
        print(f".pt label:  {getattr(data, 'label', 'NO LABEL')}")
    print()
