"""
Fix all .pt labels using STaRK SKB entity names.

The previous labels were wrong because build_subgraphs.py used STaRK answer_ids
as Neo4j nodeIds — but those are different ID spaces. The correct entity names
must come from the STaRK SKB directly.

Each subgraph_XXXXX.pt corresponds to STaRK QA index XXXXX.
We look up answer_ids from STaRK and resolve names via the SKB.

Usage:
    python fix_labels_stark.py
    python fix_labels_stark.py --dry_run   # preview without saving
"""

import os
import re
import argparse
import torch
from tqdm import tqdm
from stark_qa import load_qa, load_skb


def get_entity_name(skb, entity_id: int) -> str:
    """Try multiple SKB approaches to get entity name from index."""
    # Approach 1: get_doc_info compact
    try:
        doc = skb.get_doc_info(entity_id, add_rel=False, compact=True)
        text = str(doc).strip()
        # First line is usually the entity name
        first_line = text.split('\n')[0].strip()
        if first_line:
            return first_line
    except Exception:
        pass
    # Approach 2: direct indexing
    try:
        entity = skb[entity_id]
        if isinstance(entity, dict):
            return entity.get('name', entity.get('title', str(entity_id)))
        return str(entity).split('\n')[0].strip()
    except Exception:
        pass
    # Approach 3: node_to_str
    try:
        return skb.node_to_str(entity_id).strip()
    except Exception:
        pass
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Print what would change without saving')
    args = parser.parse_args()

    print("Loading STaRK QA dataset...")
    qa_dataset = load_qa('prime')
    df = qa_dataset.data
    print(f"QA rows: {len(df)}")

    print("Loading STaRK SKB...")
    skb = load_skb('prime', download_processed=True)
    print(f"SKB loaded: {type(skb).__name__}\n")

    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.pt')])
    print(f"Found {len(files)} .pt files\n")

    fixed = 0
    skipped = 0
    errors = 0

    for fname in tqdm(files):
        # Extract STaRK index from filename: subgraph_XXXXX.pt
        m = re.match(r'subgraph_(\d+)\.pt', fname)
        if not m:
            skipped += 1
            continue
        stark_idx = int(m.group(1))

        if stark_idx >= len(df):
            skipped += 1
            continue

        row = df.iloc[stark_idx]
        answer_ids = list(row['answer_ids'])

        if not answer_ids:
            skipped += 1
            continue

        # Get entity names from SKB
        names = []
        for aid in answer_ids:
            name = get_entity_name(skb, aid)
            if name:
                names.append(name)

        if not names:
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
    print(f"  Fixed:   {fixed}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors:  {errors}")
    if args.dry_run:
        print("  (dry run — nothing was saved)")


if __name__ == '__main__':
    main()
