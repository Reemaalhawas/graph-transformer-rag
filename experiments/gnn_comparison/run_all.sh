#!/bin/bash
set -e
echo "================================================"
echo " GNN Architecture Comparison — QA Experiments"
echo "================================================"

conda activate graphrag

# Verify environment
python verify_env.py

# Verify subgraphs are enriched before starting
python3 -c "
import torch, os
FS = '/home/ubuntu/stark-graphrag-storage'
for split in ['train', 'val', 'test']:
    for suffix in ['_enriched', '']:
        path = f'{FS}/stark_{split}_subgraphs{suffix}.pt'
        if os.path.exists(path):
            d = torch.load(path, map_location='cpu', weights_only=False)
            assert d[0].desc, f'desc empty in {path} — run enrich_subgraphs.py'
            print(f'OK: {path} ({len(d)} samples)')
            break
"

echo "--- Experiment 1/3: GAT (original baseline) ---"
python main.py \
  --gnn_type gat --num_epochs 3 --num_gnn_layers 4 \
  --batch_size 4 --eval_batch_size 4 --lr 1e-5 \
  --llama_version llama3.1-8b \
  --retrieval_config_version 0 --algo_config_version 0 \
  --g_retriever_config_version 0 --checkpointing \
  --max_txt_len 4096

echo "--- Experiment 2/3: GCN (structural baseline) ---"
python main.py \
  --gnn_type gcn --num_epochs 3 --num_gnn_layers 4 \
  --batch_size 4 --eval_batch_size 4 --lr 1e-5 \
  --llama_version llama3.1-8b \
  --retrieval_config_version 0 --algo_config_version 0 \
  --g_retriever_config_version 0 --checkpointing \
  --max_txt_len 4096

echo "--- Experiment 3/3: Graph Transformer (edge-aware) ---"
python main.py \
  --gnn_type graph_transformer --num_epochs 3 --num_gnn_layers 4 \
  --batch_size 2 --eval_batch_size 2 --lr 1e-5 \
  --llama_version llama3.1-8b \
  --retrieval_config_version 0 --algo_config_version 0 \
  --g_retriever_config_version 0 --checkpointing \
  --max_txt_len 4096

echo "All QA experiments complete."
echo "Results: /home/ubuntu/stark-graphrag-storage/qa_results/"
