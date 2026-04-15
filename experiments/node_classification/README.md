# Node Classification Experiment

Tests whether GNN architecture advantage on the QA task generalises to node type
classification on the same STaRK-Prime knowledge graph.

## Node Types (7 classes)

| Class | Type | Expected frequency |
|-------|------|--------------------|
| 0 | Drug | Common |
| 1 | GeneOrProtein | Most common |
| 2 | Disease | Common |
| 3 | BiologicalProcess | Moderate |
| 4 | MolecularFunction | Moderate |
| 5 | Pathway | Less common |
| 6 | Anatomy | Less common |

## Two Modes

- **text** — 1536-d ada-002 embeddings as node features. Expected ~90%+ accuracy.
  Differences between GNNs will be small (embeddings alone classify well).
- **structure** — 7-d one-hot node type encoding. Forces GNN to use graph structure only.
  Harder task. Architecture differences more informative here.

## Usage

```bash
# Step 1: Build the full graph from Neo4j (run once)
python experiments/node_classification/build_graph.py

# Step 2: Train all 6 combinations (3 GNNs × 2 modes)
python experiments/node_classification/train_node_clf.py
```

Results saved to `/home/ubuntu/stark-graphrag-storage/node_clf_results/results.json`
