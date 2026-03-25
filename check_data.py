"""
Quick sanity check: verify subgraphs are compatible with train.py.
Run before training to catch any data consistency issues.

Usage:
    python check_data.py
    python check_data.py --n 200   # check more files
"""

import os
import argparse
import torch

DATA_DIR = 'data/processed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50, help='Number of files to check')
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])
    print(f"Total .pt files: {len(files)}")

    # ── Per-file checks ───────────────────────────────────────────────────────
    errors = []
    for fname in files[:args.n]:
        try:
            d = torch.load(os.path.join(DATA_DIR, fname), weights_only=False)
            n = d.x.shape[0]

            assert d.x.shape[1] == 1536,                        f"x dim wrong: {d.x.shape}"
            assert d.edge_attr.shape[1] == 1536,                f"edge_attr dim wrong: {d.edge_attr.shape}"
            assert d.edge_attr.shape[0] == d.edge_index.shape[1], f"edge_attr/edge_index mismatch: {d.edge_attr.shape[0]} vs {d.edge_index.shape[1]}"
            if d.edge_index.shape[1] > 0:
                assert int(d.edge_index.max()) < n,             f"edge_index out of bounds: max={d.edge_index.max()} >= num_nodes={n}"
            assert hasattr(d, 'label')    and d.label,          f"missing or empty label"
            assert hasattr(d, 'question') and d.question,       f"missing or empty question"
            assert hasattr(d, 'desc'),                          f"missing desc field"

        except Exception as e:
            errors.append(f"{fname}: {e}")

    print(f"\nChecked {min(args.n, len(files))} files — Errors: {len(errors)}")
    for e in errors:
        print(f"  {e}")

    # ── STaRK split checks ────────────────────────────────────────────────────
    print("\nChecking STaRK official splits...")
    from stark_qa import load_qa
    qa_dataset = load_qa('prime')
    df = qa_dataset.data
    all_files = set(os.listdir(DATA_DIR))

    total_found = 0
    for split in ['train', 'val', 'test']:
        if 'split' in df.columns:
            indices = df[df['split'] == split].index.tolist()
        else:
            indices = list(qa_dataset.get_subset(split).data.index)
        found = sum(1 for i in indices if f"subgraph_{i:05d}.pt" in all_files)
        total_found += found
        print(f"  {split:5s}: {len(indices):5d} questions  →  {found:5d} subgraphs found  ({100*found/max(len(indices),1):.1f}%)")

    print(f"  {'total':5s}: {len(df):5d} questions  →  {total_found:5d} subgraphs found")

    # ── Sample inspection ─────────────────────────────────────────────────────
    print("\nSample subgraph inspection:")
    sample = torch.load(os.path.join(DATA_DIR, files[0]), weights_only=False)
    print(f"  x:          {sample.x.shape}")
    print(f"  edge_index: {sample.edge_index.shape}")
    print(f"  edge_attr:  {sample.edge_attr.shape}")
    print(f"  question:   {sample.question[:80]}")
    print(f"  label:      {repr(sample.label)}")
    print(f"  desc:       {sample.desc[:100]}")

    # ── Label format check ────────────────────────────────────────────────────
    print("\nLabel format check (first 5 files):")
    for fname in files[:5]:
        d = torch.load(os.path.join(DATA_DIR, fname), weights_only=False)
        parts = [p.strip() for p in d.label.split('|') if p.strip()]
        print(f"  {fname}: {parts}")

    print("\n" + ("=" * 50))
    if errors:
        print(f"FAILED — {len(errors)} corrupted subgraph(s) found.")
    else:
        print("ALL CHECKS PASSED — ready to train.")
    print("=" * 50)


if __name__ == '__main__':
    main()
