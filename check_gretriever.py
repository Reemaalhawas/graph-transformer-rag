#!/usr/bin/env python3
"""Run before training to confirm which GRetriever path will be used."""
try:
    from torch_geometric.nn import GRetriever
    from torch_geometric.nn.nlp import LLM
    print("USE_PYGS_GRETRIEVER=True")
except ImportError as e:
    print(f"USE_PYGS_GRETRIEVER=False — {e}")
    print("Custom fallback (_build_custom_model) will be used in train.py")
