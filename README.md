# Neo4j GNN LLM Example


# Installing Neo4j GenAI plugin
Copy over from plugins to products according to Neo4j guide.


# Installing LlamaTokenizer that's required by PyG GRetriever

LlamaTokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.


In DB:
- SKB graph with node embeddings from candidate_emb_dict.pt, downloaded from https://github.com/snap-stanford/stark
In emb/prime/text-embedding-ada-002:
- reltype_emb_dict.pt is reltype embeddings generated ourselves using openAI (18 relTypes)
- query_emb_dict.pt is query(prompt) embeddings downloaded from https://github.com/snap-stanford/stark

Dataset preparation:
- Find top 4 nodes for each query in stark-qa using Neo4j `index.vector.queryNodes`.
- Construct 2-hop subgraph from 4 nodes using Zach's cypher query (base-subgraph). Some of the 4 nodes may be dropped.
- Find top 4 edges for each query in this base-subgraph using Sklearn cos-sim. There may be <= 4 edges.
- Compute pcst on base-subgraph, top (<= 4) nodes and top 4 edges are assigned high prizes.
- Retrieve nodes textual-name, textual-description and embedding using Cypher. 
This includes the answer nodes for the query.
- Retrieve edges textual-description (edge-attr) using Cypher.
- Final data for each stark-qa is:
  - x: node embeddings of pcst nodes
  - edge_index: edge_index of pcst graph
  - edge_attr: None
  - question: f"Question: {prompt}\nAnswer: "
  - labels: textual answer nodes concatenated with |
  - desc: csv-style textual nodes and edges description

Dataset sizes:
train_data.pt: 5783
val_data.pt: 2120
test_data.pt: 2633
Total: 10536
(Smaller than stark-qa 11,204 because some are dropped due to empty PCST graphs)

# v0
Epochs = 2, k_nodes = 4

# v1
Epochs = 10, k_nodes = 10

# v2
Epochs = 10, k_nodes = 4

# v3
Epochs = 10, k_nodes = 4, edge_embedding "nodeType relType nodeType".

# v4
Epochs = 10, k_nodes = 4, same triplet edge embedding. Base graph sampling = 1-hop neighbourhood.

# v5
Epochs = 10, k_nodes = 4, k_edges = 4,
Given 4 most similar nodes and edges. Do 2-hop expansion for 4 k_nodes along edges of the top 4 types.
Pass the resulting graph to GNN & LLM.
