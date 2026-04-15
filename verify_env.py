#!/usr/bin/env python3
"""Run after install.sh to confirm the environment is correct."""
import sys
print(f"Python: {sys.version}")

import torch
assert torch.__version__.startswith("2.1"), f"Wrong torch: {torch.__version__}"
assert torch.cuda.is_available(), "CUDA not available"
print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

import torch_geometric as pyg
assert pyg.__version__ == "2.6.0", f"Wrong PyG: {pyg.__version__}"
print(f"PyG: {pyg.__version__}")

try:
    from torch_geometric.nn import GRetriever
    from torch_geometric.nn.nlp import LLM
    print("GRetriever: OK (PyG native)")
    GRETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"GRetriever: NOT available — {e}")
    print("Will use custom implementation")
    GRETRIEVER_AVAILABLE = False

from torch_geometric.nn import GAT
print("GAT: OK")

import bitsandbytes as bnb
print(f"bitsandbytes: {bnb.__version__}")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
print("peft: OK")

from neo4j import GraphDatabase
print("neo4j: OK")

import graphdatascience
print(f"graphdatascience: {graphdatascience.__version__}")

print("\n=== All checks passed ===")
print(f"GRetriever available: {GRETRIEVER_AVAILABLE}")
