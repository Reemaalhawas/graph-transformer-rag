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


#######
# Step 2: Get Python 3.11
pushd /usr/src
apt-get install libffi-dev
curl -O https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xzf Python-3.11.9.tgz
cd Python-3.11.9
./configure --enable-optimizations
make -j$(nproc)
make altinstall

update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1
echo "Location of Python: $(which python) and version $(python --version)"

# To manually select python on system
# update-alternatives --config python 
popd

#######
# Step 3: Get dependencies and load data
# in neo4j-benchmark repo directory 
pip install -r requirements.txt
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