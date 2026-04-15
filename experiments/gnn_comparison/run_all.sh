#!/bin/bash
# =============================================================================
# Full experiment suite — 8 runs
# Run from: /home/ubuntu/graph-transformer-rag/
#
# After every run, train.py saves a JSON to qa_results/ and prints the
# full comparison table so far. Completed runs are safe even if killed.
#
# Usage:
#   cd /home/ubuntu/graph-transformer-rag
#   chmod +x experiments/gnn_comparison/run_all.sh
#   nohup bash experiments/gnn_comparison/run_all.sh > run_all.log 2>&1 &
# =============================================================================
set -e
cd /home/ubuntu/graph-transformer-rag
source .venv/bin/activate

LLM="llama3.1-8b"
EPOCHS=3
LAYERS=4
HIDDEN=1024
LR=1e-5
BS=4
EBS=4
RCFG=0; ACFG=0; GCFG=0

echo "========================================================"
echo " Graph-RAG Experiment Suite  ($(date))"
echo " 8 runs: baseline · pcst_only · gretriever×3 · pipeline×3"
echo "========================================================"

# ── [1/8] Baseline: pure vector search, no GNN, no LLM ───────────────────────
echo ""; echo "--- [1/8] Baseline: pure vector search ---"
python train.py \
  --mode baseline \
  --retrieval_config_version $RCFG

# ── [2/8] PCST only: pruned subgraph text → LLM, no GNN ──────────────────────
echo ""; echo "--- [2/8] PCST Only (LLM reads pruned subgraph, no GNN) ---"
python train.py \
  --mode pcst_only \
  --llama_version $LLM --epochs $EPOCHS \
  --batch_size $BS --eval_batch_size $EBS --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [3/8] G-Retriever: GAT ───────────────────────────────────────────────────
echo ""; echo "--- [3/8] G-Retriever: GAT ---"
python train.py \
  --mode gretriever --encoder gat \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size $BS --eval_batch_size $EBS --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [4/8] G-Retriever: GCN ───────────────────────────────────────────────────
echo ""; echo "--- [4/8] G-Retriever: GCN ---"
python train.py \
  --mode gretriever --encoder gcn \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size $BS --eval_batch_size $EBS --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [5/8] G-Retriever: Graph Transformer ─────────────────────────────────────
echo ""; echo "--- [5/8] G-Retriever: Graph Transformer ---"
python train.py \
  --mode gretriever --encoder graph_transformer \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size 2 --eval_batch_size 2 --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [6/8] Pipeline: GAT (PCST pruned graph for GNN + extra nodes for LLM) ────
echo ""; echo "--- [6/8] Pipeline: GAT ---"
python train.py \
  --mode pipeline --encoder gat \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size $BS --eval_batch_size $EBS --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [7/8] Pipeline: GCN ──────────────────────────────────────────────────────
echo ""; echo "--- [7/8] Pipeline: GCN ---"
python train.py \
  --mode pipeline --encoder gcn \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size $BS --eval_batch_size $EBS --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

# ── [8/8] Pipeline: Graph Transformer ────────────────────────────────────────
echo ""; echo "--- [8/8] Pipeline: Graph Transformer ---"
python train.py \
  --mode pipeline --encoder graph_transformer \
  --llama_version $LLM --epochs $EPOCHS --num_gnn_layers $LAYERS \
  --gnn_hidden_channels $HIDDEN --batch_size 2 --eval_batch_size 2 --lr $LR \
  --retrieval_config_version $RCFG --algo_config_version $ACFG \
  --g_retriever_config_version $GCFG --checkpointing

echo ""
echo "========================================================"
echo " All 8 experiments complete  ($(date))"
echo "========================================================"
