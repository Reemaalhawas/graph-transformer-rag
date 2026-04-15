#!/bin/bash
set -e
echo 'Starting Neo4j server optimized for Hostinger VPS...'

# Get available memory and set heap/pagecache
TOTAL_MEM=$(free -g | awk 'NR==2{print $2}')
if [ $TOTAL_MEM -le 4 ]; then
    HEAP_SIZE="1G"
    PAGECACHE_SIZE="2G"
elif [ $TOTAL_MEM -le 8 ]; then
    HEAP_SIZE="2G"
    PAGECACHE_SIZE="4G"
elif [ $TOTAL_MEM -le 16 ]; then
    HEAP_SIZE="4G"
    PAGECACHE_SIZE="8G"
else
    HEAP_SIZE="8G"
    PAGECACHE_SIZE="16G"
fi

echo "Total RAM: ${TOTAL_MEM}GB"
echo "Heap Size: $HEAP_SIZE"
echo "PageCache Size: $PAGECACHE_SIZE"

# Remove any existing container
docker rm -f neo4j-prod 2>/dev/null || true

# Start Neo4j with optimized settings for Hostinger VPS
docker run -d \
  --name neo4j-prod \
  --restart unless-stopped \
  -p 7474:7474 -p 7687:7687 \
  -v /home/neo4j-data:/data \
  -v /home/neo4j-logs:/logs \
  -e NEO4J_AUTH=neo4j/HostingerNeo4j2024! \
  -e NEO4J_server_memory_heap_initial__size=$HEAP_SIZE \
  -e NEO4J_server_memory_heap_max__size=$HEAP_SIZE \
  -e NEO4J_server_memory_pagecache_size=$PAGECACHE_SIZE \
  -e NEO4J_server_default__listen__address=0.0.0.0 \
  -e NEO4J_server_default__advertised__address=VPS_IP_PLACEHOLDER \
  -e NEO4J_server_logs_debug_level=INFO \
  neo4j:5

echo 'Waiting for Neo4j to start...'
sleep 30

echo 'Neo4j server started successfully on Hostinger VPS!'
echo "Access Neo4j Browser at: http://VPS_IP_PLACEHOLDER:7474"
echo 'Username: neo4j'
echo 'Password: HostingerNeo4j2024!'

# Show container status
docker ps --filter name=neo4j-prod
docker logs neo4j-prod | tail -10
