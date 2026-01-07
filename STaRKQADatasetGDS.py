import os
import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class STaRKQADataset(InMemoryDataset):
    def __init__(self, root, raw_dataset, retrieval_config_version, algo_config_version, split="train", force_reload=False):
        self.split = split
        self.raw_dataset = raw_dataset
        self.retrieval_config_version = retrieval_config_version
        self.algo_config_version = algo_config_version
        
        emb_path = os.path.join(os.path.dirname(__file__), 'data-loading', 'emb', 'prime', 'text-embedding-ada-002', 'query', 'query_emb_dict.pt')
        self.query_embedding_dict = torch.load(emb_path, weights_only=False)
        
        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.split}_data.pt"]

    def process_single_subgraph(self, driver, question_id, prompt, answer_label, k_nodes, query_text):
        try:
            query_emb = self.query_embedding_dict[question_id].numpy()[0].tolist()
            # Use a short fetch_size to prevent the driver from buffering too much data at once
            with driver.session(fetch_size=1000) as session:
                result = session.run(query_text, {"k": k_nodes, "query_embedding": query_emb})
                records = list(result)

            if not records: return None

            node_data, id_to_idx, edges = [], {}, []
            for i, rec in enumerate(records):
                n_id = rec['nodeId']
                id_to_idx[n_id] = i
                # Ensure we convert to float32 immediately to save memory
                node_data.append(np.array(rec['embedding'], dtype=np.float32))
            
            for rec in records:
                for rel in rec['rels']:
                    s, t = rel['s'], rel['t']
                    if s in id_to_idx and t in id_to_idx:
                        edges.append([id_to_idx[s], id_to_idx[t]])

            if node_data:
                x = torch.from_numpy(np.array(node_data))
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
                             else torch.empty((2, 0), dtype=torch.long)

                data = Data(x=x, edge_index=edge_index)
                data.question = str(prompt)
                data.label = answer_label
                return data
        except Exception as e:
            # Silently fail for corrupted records to keep the bar moving
            return None

    def process(self):
        load_dotenv('db.env', override=True)
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

        with open(f"configs/retrieval_config_v{self.retrieval_config_version}.yaml", "r") as f:
            cypher_config = yaml.safe_load(f)

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        retrieval_data = []

        # LEAN QUERY: Strictly limits neighbor expansion to prevent data bloat
        optimized_query = """
        CALL db.index.vector.queryNodes('textembeddings', $k, $query_embedding) YIELD node AS seed
        WITH seed
        OPTIONAL MATCH (seed)-[r]->(neighbor:_Entity_)
        WITH seed, neighbor LIMIT 30
        WITH seed, neighbor, seed.nodeId AS sID, neighbor.nodeId AS nID
        UNWIND [seed, neighbor] AS n
        WITH n, sID, nID WHERE n IS NOT NULL
        RETURN DISTINCT n.nodeId AS nodeId, n.textEmbedding AS embedding, collect({s: sID, t: nID}) AS rels
        """

        print(f"Processing {self.split} subgraphs (Optimized Bolt Connection)...")
        
        # INCREASED max_connection_pool_size for Windows parallelism
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), 
                                  max_connection_pool_size=50) as driver:
            
            # Use max_workers=10 if your CPU has 8+ cores, otherwise stay at 4-6
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [
                    executor.submit(self.process_single_subgraph, driver, q_id, p, a, cypher_config['k_nodes'], optimized_query)
                    for _, (q_id, p, a) in dataframe.iterrows()
                ]
                
                for future in tqdm(futures, total=len(futures)):
                    res = future.result()
                    if res:
                        retrieval_data.append(res)

        if len(retrieval_data) > 0:
            print(f"Saving {len(retrieval_data)} samples...")
            self.save(retrieval_data, self.processed_paths[0])
        else:
            raise ValueError("Data retrieval failed totally.")