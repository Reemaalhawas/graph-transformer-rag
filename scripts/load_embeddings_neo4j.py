"""
Load pre-computed Ada-002 node embeddings into Neo4j as textEmbedding properties.

This is a required one-time setup step before running train.py.
It reads candidate_emb_dict.pt and writes each 1536-dim vector
onto the matching Neo4j node as a float-list property.

Usage:
    python scripts/load_embeddings_neo4j.py
    python scripts/load_embeddings_neo4j.py --batch_size 500 --dry_run
"""

import argparse
import os
import sys
import time

import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

# -----------------------------------------------------------------------
BATCH_SIZE_DEFAULT = 200   # nodes per transaction — tune for memory/speed
EMB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "emb", "prime", "text-embedding-ada-002", "doc", "candidate_emb_dict.pt",
)
# -----------------------------------------------------------------------


def load_batch(session, batch: list[dict]):
    """
    Write one batch of {nodeId, textEmbedding} into Neo4j.
    Uses UNWIND for a single round-trip per batch.
    """
    session.run(
        """
        UNWIND $rows AS row
        MATCH (n:_Entity_ {nodeId: row.nodeId})
        SET n.textEmbedding = row.emb
        """,
        rows=batch,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print first batch only, do not write to Neo4j")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Skip node IDs below this value (resume interrupted run)")
    args = parser.parse_args()

    # ---------------------------------------------------------------- env
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "db.env"), override=True)
    NEO4J_URI      = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    if not NEO4J_URI:
        sys.exit("ERROR: NEO4J_URI not set in db.env")

    # ----------------------------------------------------------- load embs
    if not os.path.exists(EMB_PATH):
        sys.exit(f"ERROR: Embedding file not found: {EMB_PATH}")

    print(f"Loading embeddings from: {EMB_PATH}")
    t0 = time.time()
    emb_dict = torch.load(EMB_PATH, weights_only=False)
    print(f"  Loaded {len(emb_dict):,} embeddings in {time.time()-t0:.1f}s")

    # Peek at first entry to understand shape
    first_key = next(iter(emb_dict))
    first_val = emb_dict[first_key]
    print(f"  Key type: {type(first_key).__name__},  Value shape: {first_val.shape},  dtype: {first_val.dtype}")

    # Sort keys so we can resume cleanly with --start_from
    all_node_ids = sorted(emb_dict.keys())
    if args.start_from > 0:
        all_node_ids = [k for k in all_node_ids if k >= args.start_from]
        print(f"  Resuming from node_id {args.start_from}: {len(all_node_ids):,} remaining")

    if args.dry_run:
        print("\nDRY RUN — showing first 3 entries, not writing to Neo4j:")
        for k in all_node_ids[:3]:
            v = emb_dict[k]
            emb_list = v.squeeze().tolist() if hasattr(v, "squeeze") else list(v)
            print(f"  nodeId={k}  emb[:4]={emb_list[:4]}  len={len(emb_list)}")
        return

    # ---------------------------------------------------------------- write
    print(f"\nConnecting to Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    # Verify connection
    with driver.session() as session:
        result = session.run("MATCH (n:_Entity_) RETURN count(n) AS cnt")
        cnt = result.single()["cnt"]
        print(f"  Connected — {cnt:,} _Entity_ nodes in graph")
        if cnt == 0:
            sys.exit("ERROR: No _Entity_ nodes found. Load the graph first with scripts/load_full_optimized.py")

    total = len(all_node_ids)
    written = 0
    errors = 0
    t_start = time.time()

    batch = []
    with driver.session() as session:
        for node_id in tqdm(all_node_ids, desc="Writing embeddings", unit="nodes"):
            v = emb_dict[node_id]
            emb_list = v.squeeze().tolist() if hasattr(v, "squeeze") else list(v)
            batch.append({"nodeId": int(node_id), "emb": emb_list})

            if len(batch) >= args.batch_size:
                try:
                    load_batch(session, batch)
                    written += len(batch)
                except Exception as e:
                    print(f"\nBatch error at node_id={node_id}: {e}")
                    errors += len(batch)
                batch = []

        # Flush remainder
        if batch:
            try:
                load_batch(session, batch)
                written += len(batch)
            except Exception as e:
                print(f"\nFinal batch error: {e}")
                errors += len(batch)

    driver.close()

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Written : {written:,} / {total:,}")
    if errors:
        print(f"  Errors  : {errors:,}  (re-run with --start_from to resume)")
    else:
        print("  No errors — run scripts/create_vector_index.py next")


if __name__ == "__main__":
    main()
