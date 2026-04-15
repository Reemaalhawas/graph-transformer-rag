# Demo

End-to-end inference demo for the GraphRAG pipeline.

## Best demo question

```
"What drugs target the CYP3A4 enzyme and treat strongyloidiasis?"
```

Expected answer: **Ivermectin**. Multi-hop reasoning required: drug→enzyme + drug→disease.

## Usage

```bash
# Interactive mode
python demo/demo.py --gnn_type gat

# Direct question
python demo/demo.py --gnn_type gat --question "What drugs target the CYP3A4 enzyme and treat strongyloidiasis?"
```

## Neo4j Browser — subgraph visualisation

After the demo runs, paste this Cypher into Neo4j Browser to show the retrieved subgraph:

```cypher
MATCH (n)-[r]->(m)
WHERE n.name IN ["Ivermectin", "CYP3A4", "Strongyloidiasis", "Albendazole"]
RETURN n, r, m LIMIT 50
```

## Presentation strategy

1. Play pre-recorded terminal video (model loads + answers question)
2. Switch to live Neo4j Browser showing the force-directed subgraph
3. Explain the PCST pruning step: why these specific nodes were selected
