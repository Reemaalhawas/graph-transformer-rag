#!/bin/bash
set -e
echo 'Starting Neo4j ultra-fast bulk import on Hostinger...'

# Pull Neo4j image
docker pull neo4j:5

# Run neo4j-admin import using Docker (fastest method)
echo 'Running bulk import...'
time docker run --rm \
  -v /home/neo4j-import:/import \
  -v /home/neo4j-data:/data \
  neo4j:5 \
  neo4j-admin database import full \
  --nodes=/import/nodes.csv \
  --relationships=/import/edges.csv \
  --overwrite-destination=true \
  neo4j

echo 'Neo4j bulk import completed!'
ls -lah /home/neo4j-data/
