"""
Migrate existing .pt subgraph files to add 'label' field.
Looks up entity names from Neo4j using the 'answer' node IDs already stored.
No OpenAI API calls needed.

Usage:
    python migrate_labels.py
"""

import os
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv('db.env', override=True)

NEO4J_URI      = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')


def get_names(node_ids, driver):
    if not node_ids:
        return ''
    res = driver.execute_query(
        "UNWIND $ids AS nodeId MATCH (n:_Entity_ {nodeId: nodeId}) RETURN n.name AS name",
        parameters_={"ids": list(node_ids)}
    )
    names = [r.data()['name'] for r in res.records if r.data().get('name')]
    return ' | '.join(names)


def main():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.pt')])
    print(f"Found {len(files)} .pt files in {DATA_DIR}")

    already_migrated = 0
    migrated = 0
    skipped = 0
    errors = 0

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        for fname in tqdm(files):
            path = os.path.join(DATA_DIR, fname)
            try:
                data = torch.load(path, weights_only=False)

                # Skip if already has a proper label (entity names, not just integers)
                if hasattr(data, 'label') and data.label and not data.label.replace(' ', '').replace('|', '').isdigit():
                    already_migrated += 1
                    continue

                # Get answer node IDs
                if hasattr(data, 'answer') and data.answer:
                    answer_ids = data.answer if isinstance(data.answer, list) else [data.answer]
                    answer_ids = [int(a) for a in answer_ids]
                else:
                    skipped += 1
                    continue

                # Look up entity names from Neo4j
                label = get_names(answer_ids, driver)
                if not label:
                    skipped += 1
                    continue

                # Add label field and ensure desc exists
                data.label = label
                if not hasattr(data, 'desc') or data.desc is None:
                    data.desc = ''

                torch.save(data, path)
                migrated += 1

            except Exception as e:
                print(f"  Error on {fname}: {e}")
                errors += 1

    print(f"\nDone!")
    print(f"  Migrated:         {migrated}")
    print(f"  Already migrated: {already_migrated}")
    print(f"  Skipped (no ans): {skipped}")
    print(f"  Errors:           {errors}")


if __name__ == '__main__':
    main()
