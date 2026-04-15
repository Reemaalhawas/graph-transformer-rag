#!/bin/bash
set -e
echo '============================================================================'
echo 'SETTING UP NEO4J ON HOSTINGER VPS'
echo '============================================================================'

# Update system (Hostinger usually uses Ubuntu)
apt-get update
apt-get install -y docker.io docker-compose htop curl ufw

# Start Docker
systemctl start docker
systemctl enable docker

# Create Neo4j directories
mkdir -p /home/neo4j-data
mkdir -p /home/neo4j-logs
mkdir -p /home/neo4j-import
mkdir -p /home/embeddings

# Copy CSV files to import directory
cp /tmp/neo4j-import/*.csv /home/neo4j-import/
chown -R 7474:7474 /home/neo4j-data /home/neo4j-logs /home/neo4j-import

# Install Python, PyTorch for loading embeddings
echo 'Installing Python and PyTorch for GPU...'
apt-get install -y python3 python3-pip python3-venv
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
pip3 install numpy scipy

echo 'GPU check:'
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo 'GPU driver check - install NVIDIA drivers if needed'
fi

# Configure firewall for Hostinger
ufw allow 22/tcp      # SSH
ufw allow 7474/tcp    # Neo4j HTTP
ufw allow 7687/tcp    # Neo4j Bolt
ufw --force enable

# Show system resources
echo 'Hostinger VPS Resources:'
free -h
df -h
lscpu | head -10

echo 'Hostinger VPS setup complete!'
