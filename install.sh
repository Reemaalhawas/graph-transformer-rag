#!/bin/bash
set -e
echo "=== Installing GraphRAG environment ==="

# Check Python and CUDA
python3 --version
nvidia-smi | head -3

# PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# PyG core + C++ extensions
pip install torch_geometric==2.6.0
pip install pyg_lib torch_scatter torch_sparse \
    torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# LLM stack in exact order
pip install transformers==4.40.0
pip install accelerate==0.29.3
pip install peft==0.10.0
pip install bitsandbytes==0.43.1

# Everything else
pip install -r requirements.txt

echo "=== Installation complete. Run: python3 verify_env.py ==="
