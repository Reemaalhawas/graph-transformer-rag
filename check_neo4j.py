from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://72.61.201.127:7687', auth=('neo4j', 'neo4j123456'))

# List all indexes
print("=== INDEXES ===")
res = driver.execute_query("SHOW INDEXES")
for rec in res.records:
    print(dict(rec))

# Check node count and labels
print("\n=== NODE LABELS ===")
res = driver.execute_query("CALL db.labels()")
for rec in res.records:
    print(rec.data())

# Check if _Entity_ nodes exist and have textEmbedding
print("\n=== SAMPLE _Entity_ NODE ===")
res = driver.execute_query("MATCH (n:_Entity_) RETURN n LIMIT 1")
for rec in res.records:
    node = rec['n']
    print("Keys:", list(node.keys()))
    print("nodeId:", node.get('nodeId'))
    emb = node.get('textEmbedding')
    print("textEmbedding type:", type(emb), "len:", len(emb) if emb else None)

driver.close()
