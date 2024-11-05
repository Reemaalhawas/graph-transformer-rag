# Neo4j GraphRAG with GNN+LLM

__Knowledge graph retrieval to improve multi-hop Q&A performance, optimized with GNN + LLM models.__

This repo contains experiments for combining Knowledge Graph Retrieval with GNN+LLM models to improve RAG.  Currently leveraging [Neo4j](https://neo4j.com/generativeai/), [G-Retriever](https://arxiv.org/abs/2402.07630), and the [STaRK-Prime dataset](https://stark.stanford.edu/dataset_prime.html) for benchmarking. 

## Architecture Overview

![Architecture](architecture.png)

- RAG on large knowledge graphs that require multi-hop retrieval and reasoning, beyond node classification and link prediction.
- General, extensible 2-part architecture: KG Retrieval & GNN+LLM.
- Efficient, stable inference time and output for real-world use cases.

## Installing Neo4j GenAI plugin
Copy over from plugins to products according to Neo4j guide.


## Installing LlamaTokenizer that's required by PyG GRetriever

LlamaTokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.


## Reproduce results
1. G-retriever:
`python train.py --checkpointing --llama_version llama3.1-8b --retrieval_config_version 0 --algo_config_version 0 --g_retriever_config_version 0 --eval_batch_size 4`
2. Pipeline:
Run `eval_pcst_ordering.ipynb` with Neo4j DB.
   
![Table Description](finalmetric.png)


In DB:
- SKB graph with node embeddings from candidate_emb_dict.pt, downloaded from https://github.com/snap-stanford/stark
In emb/prime/text-embedding-ada-002:
- reltype_emb_dict.pt is reltype embeddings generated ourselves using openAI text-embedding-ada-002 (18 relTypes)
- query_emb_dict.pt is query(prompt) embeddings downloaded from https://github.com/snap-stanford/stark

## Additional Neo4j GraphRAG Resources
- For a high-level overview of Neo4j & GenAI, have a look at [neo4j.com/genai](http://neo4j.com/genai).
- To learn how to get started using LLMs with Neo4j see [this online Graph Academy course](https://graphacademy.neo4j.com/courses/llm-fundamentals/) which is one of many [Neo4j-GenAI courses](https://graphacademy.neo4j.com/categories/llms/) covering multiple topics ranging from KG construction, to graph+vector search, and building GenAI chatbot applications.
- Pick your [GenAI framework of choice](https://neo4j.com/developer/genai-ecosystem/genai-frameworks/) to start building your own GenAI applications with Neo4j.
- Check out [Neo4j GenAI technical blogs](https://neo4j.com/developer-blog/tagged/genai/) for other worked examples and integrations.

