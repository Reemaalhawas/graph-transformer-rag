# Graph Transformer + GCN for Knowledge Graph Retrieval (GraphRAG)

Replication of [neo4j-product-examples/neo4j-gnn-llm-example](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example)
with the GAT encoder replaced by three alternative GNN encoders:

| Encoder | Flag | Description |
|---|---|---|
| GCN | `--encoder gcn` | Graph Convolutional Network (Kipf & Welling, 2017) |
| Graph Transformer | `--encoder graph_transformer` | Multi-head attention over graph (TransformerConv) |
| Combined (default) | `--encoder combined` | GCN layers + GT layers with learned fusion gate |

The full pipeline (Neo4j retrieval → PCST subgraph pruning → GNN encoding → LLM fine-tuning → metrics) is identical to the original repo.

---

## Architecture

```
STaRK-Prime QA pair
        │
Neo4j vector search  ──► 1-hop Cypher retrieval ──► base subgraph
        │
PCST pruning (via Neo4j GDS)  ──► compact subgraph
        │
Node embeddings (Ada-002, 1536-dim) from Neo4j
        │
  ┌─────┴───────────────────────────────────────────┐
  │  GNN Encoder (choose one):                       │
  │                                                   │
  │  gcn             → GCNConv × num_layers           │
  │                                                   │
  │  graph_transformer → TransformerConv × num_layers │
  │                      + pre-LN + FFN + residuals   │
  │                                                   │
  │  combined (default):                              │
  │    GCNConv × (num_layers//2)                      │
  │    + TransformerConv × (num_layers - num_layers//2)│
  │    + learned fusion gate α                        │
  └─────────────────────────┬───────────────────────┘
                            │  node embeddings [N, 1536]
                            │
                   GRetriever (PyG)
                     global mean pool
                     MLP projection
                            │
                   LLM (TinyLlama / Llama2 / Llama3)
                   fine-tuned on question+graph → answer
```

---

## Project Structure

```
graph-transformer-rag/
├── STaRKQADatasetGDS.py          # Neo4j + GDS dataset (main pipeline)
├── STaRKQAVectorSearchDataset.py # Vector-only baseline dataset
├── compute_metrics.py            # F1 / Precision / Recall / Hits / MRR
├── compute_pcst.py               # PCST subgraph pruning utilities
├── train.py                      # Training script (add --encoder flag)
├── db.env                        # Neo4j credentials (fill in before running)
├── configs/
│   ├── retrieval_config_v0.yaml  # k_nodes for Cypher retrieval
│   └── algo_config_v0.yaml       # PCST prize/topk settings
├── src/
│   └── models/
│       ├── gcn.py                # GCN encoder (wraps PyG GCN)
│       ├── graph_transformer.py  # Graph Transformer encoder
│       └── combined.py           # Combined GCN + GT encoder
├── scripts/
│   ├── import_neo4j.sh           # Load graph into Neo4j
│   ├── load_full_optimized.py    # Fast data loading
│   └── ...
├── emb/                          # Pre-computed embeddings
│   └── prime/text-embedding-ada-002/
│       ├── query/query_emb_dict.pt
│       └── doc/candidate_emb_dict.pt
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Install PyG extensions matching your CUDA version — example for CUDA 12.1:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.9.1+cu121.html
```

### 2. Configure Neo4j credentials

Edit `db.env`:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Load graph data into Neo4j (if not already done)

```bash
bash scripts/import_neo4j.sh
python scripts/load_full_optimized.py
```

---

## Training

```bash
# Combined GCN + Graph Transformer (default, recommended)
python train.py --encoder combined --llama_version tiny_llama \
    --retrieval_config_version 0 --algo_config_version 0 \
    --g_retriever_config_version 0 --checkpointing

# GCN only
python train.py --encoder gcn --llama_version tiny_llama \
    --retrieval_config_version 0 --algo_config_version 0 \
    --g_retriever_config_version 0

# Graph Transformer only
python train.py --encoder graph_transformer --llama_version tiny_llama \
    --retrieval_config_version 0 --algo_config_version 0 \
    --g_retriever_config_version 0

# LLM-only baseline (no GNN)
python train.py --num_gnn_layers 0 --llama_version tiny_llama \
    --retrieval_config_version 0 --algo_config_version 0 \
    --g_retriever_config_version 0
```

### Lambda Labs 1-GPU recommended settings

```bash
python train.py \
    --encoder combined \
    --llama_version tiny_llama \
    --retrieval_config_version 0 \
    --algo_config_version 0 \
    --g_retriever_config_version 0 \
    --gnn_hidden_channels 1536 \
    --num_gnn_layers 4 \
    --batch_size 4 \
    --eval_batch_size 16 \
    --lr 1e-5 \
    --epochs 2 \
    --checkpointing
```

> For Llama2-7B or Llama3.1-8B on an A100-80GB, increase `--batch_size` to 8.

---

## Metrics

The evaluation (identical to original repo) computes from the LLM's text output:

| Metric | Description |
|---|---|
| F1 | Token-level F1 between predicted and gold answers |
| Precision | Fraction of predicted nodes that are correct |
| Recall | Fraction of correct nodes that were predicted |
| Substring hit@1 | Regex match of top prediction in gold label |
| Exact hit@1 | Exact string match at rank 1 |
| Exact hit@5 | Any of top-5 predictions matches |
| Recall@20 | Recall within top-20 predictions |
| MRR | Mean Reciprocal Rank |

---

## References

- Kipf & Welling (2017) — Semi-Supervised Classification with GCNs
- Shi et al. (2021) — Masked Label Prediction (Graph Transformer / TransformerConv)
- He et al. (2024) — G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding
- Neo4j GNN+LLM example — https://github.com/neo4j-product-examples/neo4j-gnn-llm-example
