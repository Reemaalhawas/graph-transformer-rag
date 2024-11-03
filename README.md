# Neo4j GNN LLM Example


# Installing Neo4j GenAI plugin
Copy over from plugins to products according to Neo4j guide.


# Installing LlamaTokenizer that's required by PyG GRetriever

LlamaTokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.


# Reproduce results
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

