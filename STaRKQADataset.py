import os
import time
from typing import List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import Driver, GraphDatabase
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset

from compute_pcst import compute_pcst


class STaRKQADataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.embedding_dimension = 1536

        super().__init__(root, force_reload=force_reload)

        path = self.processed_paths[0]
        self.load(path)

    @property
    def processed_file_names(self) -> List[str]:
        return [self.split + '_data.pt']

    def process(self) -> None:
        load_dotenv('db.env', override=True)
        NEO4J_URI = os.getenv('NEO4J_URI')
        NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
        NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

        retrieval_data = []

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices].head(5)
        skipped_queries = 0
        for index, row in dataframe.iterrows():
            t = time.time()
            print(f"Retrieving and constructing base subgraph for row {index}...")
            prompt = row[1]
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                print(f"Retrieving relevant nodes and edges for prompt...")
                topk_node_ids = self.get_nodes_by_vector_search(prompt, driver, OPENAI_API_KEY)
                subgraph_rels = self.get_subgraph_rels(topk_node_ids, driver)
                if len(subgraph_rels) < 1:
                    # topk_node_ids can't form a small connected graph, skip this query
                    skipped_queries += 1
                    continue
                subgraph_rels['textEmbedding'] = self._embed(subgraph_rels['text'])
                topk_edge_ids = self.get_edges_by_vector_search(prompt, subgraph_rels)

            # process ids to consecutive tensor
            src = subgraph_rels['src'].values
            tgt = subgraph_rels['tgt'].values
            unique_nodes = np.unique(np.concatenate([src, tgt]))
            id_map = {node: i for i, node in enumerate(unique_nodes)}
            src_consecutive = [id_map[node] for node in src]
            tgt_consecutive = [id_map[node] for node in tgt]
            pcst_base_graph_topology = Data(edge_index=torch.tensor([src_consecutive, tgt_consecutive], dtype=torch.long))

            # Some topk_node_ids may not be in subgraph_rels. Drop them for now.
            mapped_topk_node_ids = [id_map[node] for node in topk_node_ids if node in id_map.keys()]

            print("Computing PCST...")
            pcst = compute_pcst(pcst_base_graph_topology, mapped_topk_node_ids, topk_edge_ids)

            # Retrive textual node and edge data
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                textual_nodes = self.get_textual_nodes(unique_nodes, driver)


            enriched_data = Data(
                x=pcst.x,
                edge_index=pcst.edge_index,
                edge_attr=None,
                question=f"Question: {prompt}\nAnswer: ",
                label="Answer",
                desc="Description",
            )

            retrieval_data.append(enriched_data)
            print(f"Finished PCST for row {index} in {time.time() - t} seconds.")
            print(f"Skipped {skipped_queries} queries due to insufficient subgraph data.")

        self.save(retrieval_data, self.processed_paths[0])


    def get_nodes_by_vector_search(self, prompt: str, driver: Driver, OPENAI_API_KEY: str) -> List:
        """
        Given a prompt, encode it with OpenAI's API and search for similar nodes in the SKB graph in Neo4j

        :param driver:
        :return: A list of 4 node-ids that are most similar to the prompt
        """
        res = driver.execute_query("""
        WITH genai.vector.encode(
          $searchPrompt,
          "OpenAI",
          {token:$token}) AS queryVector
        CALL db.index.vector.queryNodes($index, $k, queryVector) YIELD node
        RETURN node.nodeId AS nodeId
        """,
                                   parameters_={
                                       "searchPrompt": prompt,
                                       "token": OPENAI_API_KEY,
                                       "index": "text_embeddings",
                                       "k": 4})
        return [rec.data()['nodeId'] for rec in res.records]

    def get_subgraph_rels(self, node_ids: List, driver: Driver):
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        // create filtered cartesian product
        WITH collect(node) AS sources, collect(node) AS targets
        UNWIND sources as source
        UNWIND targets as target
        WITH source, target
        WHERE source > target //how is this calculated? on element id?...it works

        // find connecting paths
        MATCH (source)-[rl]->{0,2}(target)

        //get rels
        UNWIND rl AS r
        WITH DISTINCT r
        MATCH (m)-[r]->(n)
        RETURN
        m.nodeId AS src,
        n.nodeId AS tgt,
        n.name + ' - ' + type(r) +  ' -> ' + m.name AS text
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])

    def get_edges_by_vector_search(self, prompt: str, subgraph_rels: DataFrame) -> np.ndarray:
        """
        Given a prompt, encode it with OpenAI's API and search for similar edges in the SKB graph in Neo4j

        :param driver:
        :return: A list of 4 edges (node pairs) that are most similar to the prompt
        """
        prompt_emb = self.embedding_model.embed_query(prompt)
        sims = cosine_similarity([prompt_emb], np.vstack(subgraph_rels["textEmbedding"].values))
        k = min(4, len(subgraph_rels))
        indices = np.argpartition(sims[0], -k)[-k:]

        return indices[np.argsort(sims[0][indices])[::-1]]

    def _chunks(self, xs, n=500):
        n = max(1, n)
        return [xs[i:i + n] for i in range(0, len(xs), n)]

    def _embed(self, doc_list, chunk_size=500):
        embeddings = []
        for docs in self._chunks(doc_list, chunk_size):
            embeddings.extend(self.embedding_model.embed_documents(docs))
        return embeddings

    def get_textual_nodes(self, node_ids: List, driver: Driver) -> List:
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        RETURN node.nodeId AS nodeId, node.name AS name, node.description AS description
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])