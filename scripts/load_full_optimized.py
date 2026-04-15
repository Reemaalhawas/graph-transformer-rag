import os
import csv
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load client configuration
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "client.env"))
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "")

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prime", "neo4j_export")
NODES = os.path.join(BASE, "nodes.csv")
EDGES = os.path.join(BASE, "edges.csv")

print("=" * 80)
print("OPTIMIZED FULL DATASET LOADING INTO NEO4J")
print("=" * 80)
print(f"Database: {URI}")
print(f"User: {USER}")
print()

# Show what files we're loading
print("FILES BEING LOADED:")
if os.path.exists(NODES):
    size_mb = os.path.getsize(NODES) / (1024*1024)
    print(f"✅ Nodes: {NODES}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"❌ Missing: {NODES}")
    exit(1)

if os.path.exists(EDGES):
    size_mb = os.path.getsize(EDGES) / (1024*1024)
    print(f"✅ Edges: {EDGES}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"❌ Missing: {EDGES}")
    exit(1)

# Count rows in files
print("\nCOUNTING ROWS...")
with open(NODES, 'r', encoding='utf-8') as f:
    node_count = sum(1 for line in f) - 1  # subtract header
print(f"Nodes to load: {node_count:,}")

with open(EDGES, 'r', encoding='utf-8') as f:
    edge_count = sum(1 for line in f) - 1  # subtract header
print(f"Edges to load: {edge_count:,}")

print("\nOPTIMIZATIONS ENABLED:")
print("🚀 Large batch sizes (10,000 items)")
print("🚀 Index-based edge loading")
print("🚀 MERGE strategy for data safety")
print("🚀 Progress tracking every 50,000 items")
print(f"\nEstimated time: ~{(node_count + edge_count) // 50000} minutes (50x faster!)")
print("=" * 80)

# Ask for confirmation
response = input("This will CLEAR existing data and load the full dataset. Continue? (y/N): ")
if response.lower() != 'y':
    print("Cancelled.")
    exit(0)

print("\nConnecting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))

def create_indexes(session):
    """Create indexes for faster loading"""
    print("Creating indexes for faster loading...")
    try:
        session.run("CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.node_id)")
        # Wait a moment for index to be ready
        time.sleep(2)
        print("✅ Index created on node_id")
    except Exception as e:
        print(f"Index creation warning: {e}")

def load_nodes_batch(session, rows):
    """Load nodes with MERGE for safety"""
    query = """
    UNWIND $batch AS row
    MERGE (n:Node {node_id: toInteger(row.node_id)})
    SET n.name = row.name,
        n.node_type_id = toInteger(row.node_type_id),
        n.node_type_name = row.node_type_name,
        n.description = row.description
    """
    session.run(query, batch=rows)

def load_edges_batch(session, rows):
    """Load edges with optimized query using index"""
    query = """
    UNWIND $batch AS row
    MATCH (s:Node {node_id: toInteger(row.src_id)})
    MATCH (d:Node {node_id: toInteger(row.dst_id)})
    MERGE (s)-[rel:RELATED_TO {rel_type_id: toInteger(row.rel_type_id)}]->(d)
    SET rel.rel_type = row.rel_type
    """
    session.run(query, batch=rows)

# Much larger batch sizes for better performance
NODE_BATCH_SIZE = 10000
EDGE_BATCH_SIZE = 5000  # Smaller for edges due to complexity
PROGRESS_INTERVAL = 50000  # Report progress less frequently

start_time = time.time()

with driver.session() as session:
    # Clear existing data
    print("Clearing existing data...")
    session.run("MATCH (n) DETACH DELETE n")
    
    # Create indexes first
    create_indexes(session)
    
    # Load nodes
    print(f"Loading {node_count:,} nodes in batches of {NODE_BATCH_SIZE:,}...")
    with open(NODES, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        loaded_nodes = 0
        
        for row in reader:
            batch.append(row)
            if len(batch) >= NODE_BATCH_SIZE:
                load_nodes_batch(session, batch)
                loaded_nodes += len(batch)
                
                # Progress reporting every PROGRESS_INTERVAL
                if loaded_nodes % PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = loaded_nodes / elapsed if elapsed > 0 else 0
                    eta = (node_count - loaded_nodes) / rate if rate > 0 else 0
                    print(f"  Loaded {loaded_nodes:,}/{node_count:,} nodes ({loaded_nodes/node_count*100:.1f}%) - {rate:.0f}/sec - ETA: {eta/60:.1f}m")
                batch = []
        
        if batch:
            load_nodes_batch(session, batch)
            loaded_nodes += len(batch)
    
    print(f"✅ Loaded {loaded_nodes:,} nodes total")
    
    # Load edges
    print(f"Loading {edge_count:,} edges in batches of {EDGE_BATCH_SIZE:,}...")
    with open(EDGES, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        batch = []
        loaded_edges = 0
        
        for row in reader:
            batch.append(row)
            if len(batch) >= EDGE_BATCH_SIZE:
                load_edges_batch(session, batch)
                loaded_edges += len(batch)
                
                # Progress reporting every PROGRESS_INTERVAL
                if loaded_edges % PROGRESS_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    rate = loaded_edges / elapsed if elapsed > 0 else 0
                    eta = (edge_count - loaded_edges) / rate if rate > 0 else 0
                    print(f"  Loaded {loaded_edges:,}/{edge_count:,} edges ({loaded_edges/edge_count*100:.1f}%) - {rate:.0f}/sec - ETA: {eta/60:.1f}m")
                batch = []
        
        if batch:
            load_edges_batch(session, batch)
            loaded_edges += len(batch)
    
    print(f"✅ Loaded {loaded_edges:,} edges total")
    
    # Final verification
    print("\nFINAL VERIFICATION:")
    result = session.run("MATCH (n:Node) RETURN count(n) AS node_count")
    nodes_in_db = result.single()["node_count"]
    
    result = session.run("MATCH ()-[r]->() RETURN count(r) AS edge_count")
    edges_in_db = result.single()["edge_count"]
    
    total_time = time.time() - start_time
    
    print(f"Nodes in database: {nodes_in_db:,}")
    print(f"Edges in database: {edges_in_db:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Overall rate: {(nodes_in_db + edges_in_db) / total_time:.0f} items/second")

driver.close()
print("\n🎉 OPTIMIZED FULL DATA LOAD COMPLETE!")
print("Open http://localhost:7474 and try: MATCH (n:Node) RETURN count(n);")
print("\nSample queries to try:")
print("• MATCH (n:Node) WHERE n.node_type_name = 'gene' RETURN count(n);")
print("• MATCH (n:Node)-[r]->(m:Node) RETURN n.name, r.rel_type, m.name LIMIT 10;")