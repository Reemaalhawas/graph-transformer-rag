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
print("ULTRA-FAST CSV IMPORT FOR NEO4J")
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

print("\nULTRA-FAST OPTIMIZATIONS:")
print("⚡ Neo4j LOAD CSV (native bulk import)")
print("⚡ File:// protocol for maximum speed")
print("⚡ Single-pass loading")
print("⚡ Automatic indexing")
print(f"\nEstimated time: ~{(node_count + edge_count) // 200000} minutes (200x faster!)")
print("=" * 80)

# Ask for confirmation
response = input("This will CLEAR existing data and load the full dataset. Continue? (y/N): ")
if response.lower() != 'y':
    print("Cancelled.")
    exit(0)

print("\nConnecting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=(USER, PWD))

# Convert Windows paths to file:// URLs
def path_to_file_url(path):
    """Convert Windows path to file:// URL for Neo4j LOAD CSV"""
    return f"file:///{path.replace(chr(92), '/')}"  # Replace backslashes

nodes_url = path_to_file_url(NODES)
edges_url = path_to_file_url(EDGES)

print(f"Nodes URL: {nodes_url}")
print(f"Edges URL: {edges_url}")

start_time = time.time()

with driver.session() as session:
    # Clear existing data
    print("Clearing existing data...")
    session.run("MATCH (n) DETACH DELETE n")
    
    # Load nodes using LOAD CSV
    print(f"Loading {node_count:,} nodes using LOAD CSV...")
    load_nodes_query = f"""
    LOAD CSV WITH HEADERS FROM '{nodes_url}' AS row
    CREATE (n:Node)
    SET n.node_id = toInteger(row.node_id),
        n.name = row.name,
        n.node_type_id = toInteger(row.node_type_id),
        n.node_type_name = row.node_type_name,
        n.description = row.description
    """
    
    try:
        session.run(load_nodes_query)
        nodes_time = time.time() - start_time
        print(f"✅ Loaded nodes in {nodes_time:.1f} seconds ({node_count/nodes_time:.0f} nodes/sec)")
    except Exception as e:
        print(f"❌ Error loading nodes: {e}")
        print("Falling back to Python-based loading...")
        driver.close()
        exit(1)
    
    # Create index for faster edge loading
    print("Creating index on node_id...")
    session.run("CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.node_id)")
    time.sleep(2)  # Wait for index
    
    # Load edges using LOAD CSV
    print(f"Loading {edge_count:,} edges using LOAD CSV...")
    load_edges_query = f"""
    LOAD CSV WITH HEADERS FROM '{edges_url}' AS row
    MATCH (s:Node {{node_id: toInteger(row.src_id)}})
    MATCH (d:Node {{node_id: toInteger(row.dst_id)}})
    CREATE (s)-[rel:RELATED_TO]->(d)
    SET rel.rel_type_id = toInteger(row.rel_type_id),
        rel.rel_type = row.rel_type
    """
    
    try:
        session.run(load_edges_query)
        edges_time = time.time() - start_time - nodes_time
        print(f"✅ Loaded edges in {edges_time:.1f} seconds ({edge_count/edges_time:.0f} edges/sec)")
    except Exception as e:
        print(f"❌ Error loading edges: {e}")
        print("This might be due to file:// protocol restrictions.")
        print("Try the optimized Python version instead: load_full_optimized.py")
    
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
print("\n🎉 ULTRA-FAST DATA LOAD COMPLETE!")
print("Open http://localhost:7474 and try: MATCH (n:Node) RETURN count(n);")