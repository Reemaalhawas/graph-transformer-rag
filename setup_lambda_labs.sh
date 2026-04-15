#!/bin/bash
# =============================================================================
# Lambda Labs Setup Script — graph-transformer-rag
# Run this once after SSH-ing into a fresh Lambda Labs instance.
#
# Assumptions:
#   - Ubuntu 22.04
#   - CUDA 12.1 (Lambda Labs default)
#   - Python 3.10
#
# Usage:
#   chmod +x setup_lambda_labs.sh
#   ./setup_lambda_labs.sh
# =============================================================================

set -e  # exit on first error

echo "========================================================"
echo " Graph Transformer + GCN — Lambda Labs Setup"
echo "========================================================"

CUDA_VERSION="cu121"   # change to cu118 if your instance has CUDA 11.8
TORCH_VERSION="2.4.1"

# -----------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------
echo ""
echo "[1/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    wget curl git unzip \
    openjdk-21-jdk \
    build-essential g++ \
    screen htop

# -----------------------------------------------------------------------
# 2. Python environment
# -----------------------------------------------------------------------
echo ""
echo "[2/7] Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools -q

# -----------------------------------------------------------------------
# 3. PyTorch (CUDA-matched)
# -----------------------------------------------------------------------
echo ""
echo "[3/7] Installing PyTorch ${TORCH_VERSION} for ${CUDA_VERSION}..."
pip install torch==${TORCH_VERSION} torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION} -q

# -----------------------------------------------------------------------
# 4. PyG + extensions (must match torch + CUDA exactly)
# -----------------------------------------------------------------------
echo ""
echo "[4/7] Installing PyTorch Geometric and extensions..."
pip install torch-geometric==2.6.1 -q
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html -q

# -----------------------------------------------------------------------
# 5. Python project requirements
# -----------------------------------------------------------------------
echo ""
echo "[5/7] Installing project requirements..."
pip install -r requirements.txt -q

# -----------------------------------------------------------------------
# 6. Neo4j 5.x Community Edition
# -----------------------------------------------------------------------
echo ""
echo "[6/7] Installing Neo4j 5.x..."

# Add Neo4j APT repository
wget -qO- https://debian.neo4j.com/neotechnology.gpg.key \
    | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/neo4j.gpg

echo "deb https://debian.neo4j.com stable 5" \
    | sudo tee /etc/apt/sources.list.d/neo4j.list

sudo apt-get update -qq
sudo apt-get install -y -qq neo4j=1:5.28.1

# Configure Neo4j:
#   - Allow remote connections (needed if you want to connect from your laptop)
#   - Set initial password
sudo neo4j-admin dbms set-initial-password "your_password_here"

# Start Neo4j
sudo systemctl enable neo4j
sudo systemctl start neo4j

echo "Waiting for Neo4j to start..."
sleep 15

# Verify
sudo systemctl status neo4j --no-pager | head -5

# -----------------------------------------------------------------------
# 7. Fill db.env
# -----------------------------------------------------------------------
echo ""
echo "[7/7] Configuring db.env..."
cat > db.env << 'EOF'
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
EOF

echo ""
echo "========================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Edit db.env if you changed the Neo4j password"
echo "   2. Run: python check_setup.py"
echo "   3. Run the data loading notebook:"
echo "      jupyter notebook data-loading/stark_prime_neo4j_loading.ipynb"
echo "   4. After data is loaded, run:"
echo "      python train.py --encoder combined --llama_version tiny_llama \\"
echo "          --retrieval_config_version 0 --algo_config_version 0 \\"
echo "          --g_retriever_config_version 0 --checkpointing"
echo "========================================================"
