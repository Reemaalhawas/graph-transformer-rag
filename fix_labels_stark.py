"""
Fix all .pt labels using STaRK SKB entity names.

The previous labels were wrong because:
1. STaRK answer_ids were used as Neo4j nodeIds (different ID spaces)
2. answer_ids were stored as strings and incorrectly parsed character-by-character

This script re-reads answer_ids from the STaRK QA dataset (by matching filename
index to QA row index) and looks up correct entity names from the STaRK SKB.

Usage:
    python fix_labels_stark.py --dry_run   # preview without saving
    python fix_labels_stark.py             # apply fixes
"""

import os
import re
import ast
import argparse
import torch
from tqdm import tqdm
from stark_qa import load_qa, load_skb


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
    # Method 1: node_attr_dict
    try:
        attrs = skb.node_attr_dict[entity_id]
        if isinstance(attrs, dict):
            name = attrs.get('name') or attrs.get('title') or attrs.get('id', '')
            if name:
                return str(name)
    except Exception:
        pass

    # Method 2: node_info
    try:
        info = skb.node_info[entity_id]
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
            if ':' in first:
                return first.split(':', 1)[1].strip()
            if first and not first.startswith('--'):
                return first
    except Exception:
        pass

    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--dry_run',  action='store_true')
    args = parser.parse_args()

    print("Loading STaRK QA dataset...")
    qa_dataset = load_qa('prime')
    df = qa_dataset.data
    print(f"QA rows: {len(df)}")

    print("Loading STaRK SKB...")
    skb = load_skb('prime', download_processed=True)
    print(f"SKB loaded: {type(skb).__name__}\n")

    # Quick sanity check on the SKB API using the first real example
    print("=== API SANITY CHECK ===")
    row0 = df.iloc[0]
    ids0 = parse_answer_ids(row0['answer_ids'])
    print(f"Parsed answer_ids for row 0: {ids0}")
    if ids0:
        aid = ids0[0]
        name = get_entity_name(skb, aid)
        print(f"Entity name for {aid}: {repr(name)}")
        # Also show raw attrs for debugging
        try:
            print(f"node_attr_dict[{aid}]: {str(skb.node_attr_dict[aid])[:200]}")
        except Exception as e:
            print(f"node_attr_dict error: {e}")
    print("========================\n")

    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.pt')])
    print(f"Found {len(files)} .pt files\n")

    fixed = 0
    skipped = 0
    errors = 0
    empty_names = 0

    for fname in tqdm(files):
        m = re.match(r'subgraph_(\d+)\.pt', fname)
        if not m:
            skipped += 1
            continue
        stark_idx = int(m.group(1))

        if stark_idx >= len(df):
            skipped += 1
            continue

        row = df.iloc[stark_idx]
        answer_ids = parse_answer_ids(row['answer_ids'])

        if not answer_ids:
            skipped += 1
            continue

        names = [get_entity_name(skb, aid) for aid in answer_ids]
        names = [n for n in names if n]  # drop empty

        if not names:
            empty_names += 1
            skipped += 1
            continue

        label = ' | '.join(names)

        fpath = os.path.join(args.data_dir, fname)
        try:
            data = torch.load(fpath, weights_only=False)
            old_label = getattr(data, 'label', '')

            if args.dry_run:
                print(f"{fname}: old={repr(old_label[:60])}  →  new={repr(label[:60])}")
                fixed += 1
                continue

            data.label = label
            if not hasattr(data, 'desc') or data.desc is None:
                data.desc = ''
            torch.save(data, fpath)
            fixed += 1

        except Exception as e:
            print(f"Error on {fname}: {e}")
            errors += 1

    print(f"\nDone!")
    print(f"  Fixed:        {fixed}")
    print(f"  Skipped:      {skipped}  (empty_names: {empty_names})")
    print(f"  Errors:       {errors}")
    if args.dry_run:
        print("  (dry run — nothing was saved)")


if __name__ == '__main__':
    main()
