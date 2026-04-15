import os
from typing import List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from pandas import DataFrame
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class STaRKQAVectorSearchDataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        # Path adjusted for this repo: embeddings live in emb/ not data-loading/emb/
        self.query_embedding_dict = torch.load(
            os.path.join(os.path.dirname(__file__), 'emb/prime/text-embedding-ada-002/query/query_emb_dict.pt'),
            weights_only=False,
        )

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

        retrieval_data = []

        dataframe = self.raw_dataset.data.loc[self.raw_dataset.indices]
        for index, qa_row in tqdm(dataframe.iterrows()):
            prompt = qa_row[1]
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                query_emb = self.query_embedding_dict[qa_row[0]].numpy()[0]
                topk_node_ids = self.get_nodes_by_vector_search(query_emb, driver)

            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                textual_nodes_df = self.get_textual_nodes(topk_node_ids, driver)

                textual_nodes_df.description.fillna("")
                textual_nodes_df['node_attr'] = textual_nodes_df.apply(
                    lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
                textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
                nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)

                answer_ids = eval(qa_row[2])
                answers = self.get_textual_nodes(answer_ids, driver)['name'].tolist()

            enriched_data = Data(
                x=None,
                edge_index=None,
                edge_attr=None,
                question=f"Question: {prompt}\nAnswer: ",
                label=('|').join(answers).lower(),
                desc=nodes_desc,
            )

            retrieval_data.append(enriched_data)

        self.save(retrieval_data, self.processed_paths[0])

    def get_nodes_by_vector_search(self, query_embedding: np.ndarray, driver: Driver) -> List:
        res = driver.execute_query("""
        CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
        RETURN node.nodeId AS nodeId
        """,
                                   parameters_={
                                       "index": "text_embeddings",
                                       "k": 4,
                                       "query_embedding": query_embedding})
        return [rec.data()['nodeId'] for rec in res.records]

    def get_textual_nodes(self, node_ids: List, driver: Driver) -> DataFrame:
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])
