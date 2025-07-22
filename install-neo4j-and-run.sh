# After launching into container

# Step 1: Install neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | tee -a /etc/apt/sources.list.d/neo4j.list
apt-get update

# All installable versions of neo4j
# apt list -a neo4j

# Latest version
apt-get install -y neo4j=1:2025.06.2
neo4j start

# If neo4j starts correctly
pushd /var/lib/neo4j
mv products/* plugins/
neo4j-admin dbms set-initial-password mypassword
echo "dbms.security.procedures.allowlist=gds.*" >> /etc/neo4j/neo4j.conf
popd

neo4j restart

#
# Add these credentials to your db.env file
#
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=mypassword
# 
# To manually select python on system
# update-alternatives --config python 
popd

#######
# Step 3: Get dependencies and load data
# in neo4j-benchmark repo directory 
pip install -r requirements.txt
git clone https://github.com/snap-stanford/stark.git
pushd stark
# frozen commit after last update to support python3.12
git checkout e98459b
pip install .
popd

pushd data-loading
python emb_download.py --dataset prime --emb_dir emb/
python load_data.py
popd

######
# Step 4: Run benchmark
# Make sure you have the correct PyG branch installed in your environment
set -x
python train.py \
    --checkpointing \
    --llama_version llama3.1-8b \
    --retrieval_config_version 0 \
    --algo_config_version 0 \
    --g_retriever_config_version 0 \
    --eval_batch_size 4 \
    --num_gpus 8 \
    2>&1 | tee result.out
set +x
