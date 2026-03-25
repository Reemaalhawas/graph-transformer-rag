"""
Diagnose STaRK split structure to fix train/val/test separation.

Usage:
    python check_stark_splits.py
"""

from stark_qa import load_qa

qa = load_qa('prime')
df = qa.data
print('columns:', list(df.columns))
print('shape:', df.shape)

if 'split' in df.columns:
    print('\nsplit column value counts:')
    print(df['split'].value_counts())
else:
    print('\nNo split column found — checking get_subset()')
    for split in ['train', 'val', 'test']:
        sub = qa.get_subset(split)
        print(f'\n{split}:')
        print(f'  size: {len(sub.data)}')
        print(f'  index[:5]: {list(sub.data.index[:5])}')
        print(f'  columns: {list(sub.data.columns)}')
