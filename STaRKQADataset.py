import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from neo4j import Driver, GraphDatabase
from pandas import DataFrame
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from compute_pcst import compute_pcst, assign_prizes_topk, assign_prizes_modified
from compute_metrics import compute_intermediate_metrics


class STaRKQADataset(InMemoryDataset):

    def __init__(
        self,
        root: str,
        raw_dataset: Dataset,
        dataset_version: str,
        split: str = "train",
        force_reload: bool = False,
    ) -> None:
        self.split = split
        self.raw_dataset = raw_dataset
        self.dataset_version = dataset_version
        #self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        #embedding_dimension is fixed as 1536
        # load from parent directory of this file
        self.reltype_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/doc/reltype_emb_dict.pt'))
        self.query_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/query/query_emb_dict.pt'))
        self.triplet_embedding_dict = torch.load(os.path.join(os.path.dirname(__file__), 'data-loading/emb/prime/text-embedding-ada-002/doc/triplet_sentence_emb_dict.pt'))

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
        skipped_queries = 0

        config_path = f"configs/star_qa_{self.dataset_version}.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)


        correct_nodes = {}; topk_nodes = {}; subgraph_nodes = {}; pcst_nodes = {}
        pcst_edges = []

        for index, qa_row in tqdm(dataframe.iterrows()):
            prompt = qa_row[1]
            query_emb = self.query_embedding_dict[qa_row[0]].numpy()[0]
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                topk_node_ids = self.get_nodes_by_vector_search(query_emb, config["k_nodes"], driver)
                subgraph_rels = self.get_subgraph_rels(topk_node_ids, config["cypher_query_type"], driver)

            correct_ids = eval(qa_row[2])
            topk_nodes[index] = topk_node_ids
            correct_nodes[index] = correct_ids

            if len(subgraph_rels) < 1:
                # topk_node_ids can't form a small connected graph, skip this query
                skipped_queries += 1
                continue

            if config["edge_embedding_method"] == 'triplet':
                subgraph_rels['textEmbedding'] = self._embed_triplet(subgraph_rels['srcType'], subgraph_rels['relType'], subgraph_rels['tgtType'])
            else:
                subgraph_rels['textEmbedding'] = self._embed(subgraph_rels['relType'])


            # process ids to consecutive tensor
            src = subgraph_rels['src'].values
            tgt = subgraph_rels['tgt'].values
            unique_nodes = np.unique(np.concatenate([src, tgt]))

            subgraph_nodes[index] = unique_nodes

            id_map = {node: i for i, node in enumerate(unique_nodes)}
            src_consecutive = [id_map[node] for node in src]
            tgt_consecutive = [id_map[node] for node in tgt]
            pcst_base_graph_topology = Data(edge_index=torch.tensor([src_consecutive, tgt_consecutive], dtype=torch.long))


            if self.dataset_version in ["v9", "v10"]:
                top_edges, second_top_edges = self.get_edges_by_reltype_vector_search(qa_row[0], subgraph_rels)
                with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                    topn_nodes = self.get_topn_similar_nodes(query_emb, unique_nodes.tolist(), driver, 25)
                mapped_topn_node_ids = [id_map[node] for node in topn_nodes if node in id_map.keys()]
                if self.dataset_version == "v9":
                    node_prizes, edge_prizes = assign_prizes_modified(pcst_base_graph_topology, mapped_topn_node_ids, top_edges, second_top_edges)
                else:
                    node_prizes, edge_prizes = assign_prizes_modified(pcst_base_graph_topology, mapped_topn_node_ids, top_edges, second_top_edges, 0.5, 0.2)

            else:
                topk_edge_ids = self.get_edges_by_vector_search(qa_row[0], subgraph_rels, config["k_edges"])
                mapped_topk_node_ids = [id_map[node] for node in topk_node_ids if node in id_map.keys()]
                node_prizes, edge_prizes = assign_prizes_topk(pcst_base_graph_topology, mapped_topk_node_ids, topk_edge_ids)

            pcst, inner_id_mapping, selected_nodes, selected_edges = compute_pcst(pcst_base_graph_topology,
                                                                                  node_prizes, edge_prizes)
            pcst_edges.append(selected_edges.shape[1])

            reverse_id_map = {v: k for k, v in id_map.items()}
            pcst_nodes_original_ids = [reverse_id_map[intermediate_id] for intermediate_id in selected_nodes]
            pcst_nodes[index] = pcst_nodes_original_ids

            # Retrieve node embedding, label and textual graph description
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                textual_nodes_df = self.get_textual_nodes(pcst_nodes_original_ids, driver)
                original_edges = [(reverse_id_map[src.item()], reverse_id_map[tgt.item()]) for src, tgt in selected_edges.t()]
                textual_edges_df = self.get_textual_edges(original_edges, driver)

            node_embedding = torch.tensor(textual_nodes_df['textEmbedding'].tolist())

            textual_nodes_df.description.fillna("")
            textual_nodes_df['node_attr'] = textual_nodes_df.apply(lambda row: f"name: {row['name']}, description: {row['description']}", axis=1)
            textual_nodes_df.rename(columns={'nodeId': 'node_id'}, inplace=True)
            nodes_desc = textual_nodes_df.drop(['name', 'description', 'textEmbedding'], axis=1).to_csv(index=False)

            edges_desc = textual_edges_df.to_csv(index=False)

            desc = nodes_desc + '\n' + edges_desc

            answer_ids = eval(qa_row[2])
            with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
                answers = self.get_textual_nodes(answer_ids, driver)['name'].tolist()

            enriched_data = Data(
                x=node_embedding,
                edge_index=pcst.edge_index,
                edge_attr=None,  # add edge_attr if needed
                question=f"Question: {prompt}\nAnswer: ",
                label=('|').join(answers).lower(),
                desc=desc,
            )

            retrieval_data.append(enriched_data)

        print(f"Skipped {skipped_queries} queries due to insufficient subgraph data.")
        t = {'correct_nodes': correct_nodes, 'topk_nodes': topk_nodes, 'subgraph1_nodes': subgraph_nodes, 'pcst_nodes': pcst_nodes}
        print(f"Evaluate {self.processed_paths[0][:-3]}...")
        compute_intermediate_metrics(t)
        print(f"Average number of edges in PCST graph is: {np.mean(pcst_edges)}")
        torch.save(t, self.processed_paths[0][:-3] + '_nodes.pt')
        self.save(retrieval_data, self.processed_paths[0])


    def get_nodes_by_vector_search(self, query_embedding: np.ndarray, k_nodes: int, driver: Driver) -> List:
        """
        Given a prompt, encode it with OpenAI's API and search for similar nodes in the SKB graph in Neo4j

        :param driver:
        :return: A list of 4 node-ids that are most similar to the prompt
        """
        res = driver.execute_query("""
        CALL db.index.vector.queryNodes($index, $k, $query_embedding) YIELD node
        RETURN node.nodeId AS nodeId
        """,
                                   parameters_={
                                       "index": "text_embeddings",
                                       "k": k_nodes,
                                       "query_embedding": query_embedding})
        return [rec.data()['nodeId'] for rec in res.records]


    def get_topn_similar_nodes(self, query_emb: np.ndarray, node_ids: List, driver: Driver, top_nodes: int) -> List:
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH (node:_Entity_ {nodeId:nodeId}) RETURN node.nodeId as nodeId, node.textEmbedding AS textEmbedding
        """,
                                   parameters_={
                                       "nodeIds": node_ids})
        node_embs = pd.DataFrame([rec.data() for rec in res.records])
        embeddings = np.vstack(node_embs['textEmbedding'].values)
        cos_sim = cosine_similarity(embeddings, query_emb.reshape(1,-1)).ravel()
        top_n_indices = np.argsort(cos_sim)[-top_nodes:][::-1]
        top_n_nodeIds = node_embs.iloc[top_n_indices]['nodeId'].to_numpy()

        return top_n_nodeIds.tolist()


    def get_subgraph_rels(self, node_ids: List, cypher_query: str, driver: Driver):
        if cypher_query == "1hop":
            res = driver.execute_query("""
                UNWIND $nodeIds AS nodeId
                MATCH (source:_Entity_ {nodeId:nodeId})-[rl]->{0,1}(target)

                UNWIND rl as r
                WITH DISTINCT r
                MATCH (m)-[r]-(n)
                RETURN
                m.nodeId as src,
                n.nodeId as tgt,
                type(r) as relType,
                labels(m)[0] as srcType,
                labels(n)[0] as tgtType
            """,
                                       parameters_={'nodeIds': node_ids})

        if cypher_query == "2path":
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
            type(r) AS relType,
            labels(m)[0] as srcType,
            labels(n)[0] as tgtType
            """,
                                       parameters_={"nodeIds": node_ids})

        if cypher_query == "2hop":
            res = driver.execute_query("""
            MATCH (source)-[r1]->(middle)-[r2]->(target)
            WHERE source.nodeId IN $nodeIds
            CALL (source, middle, target, r1, r2) {
                RETURN
                source.nodeId as src,
                middle.nodeId as tgt,
                type(r1) as relType,
                labels(source)[0] as srcType,
                labels(middle)[0] as tgtType
                UNION
                RETURN
                middle.nodeId as src,
                target.nodeId as tgt,
                type(r2) as relType,
                labels(middle)[0] as srcType,
                labels(target)[0] as tgtType
            }   
                RETURN src, tgt, relType, srcType, tgtType
            """,
                                       parameters_={"nodeIds": node_ids})

        return pd.DataFrame([rec.data() for rec in res.records])

    def get_edges_by_vector_search(self, qa_row_id: int, subgraph_rels: DataFrame, k=4) -> np.ndarray:
        """
        Given a prompt find the most similar edges in the subgraph

        :param driver:
        :return: A list of 4 edges (node pairs) that are most similar to the prompt
        """

        prompt_emb = self.query_embedding_dict[qa_row_id]
        sims = cosine_similarity(prompt_emb, np.vstack(subgraph_rels["textEmbedding"].values))
        k = min(k, len(subgraph_rels))
        indices = np.argpartition(sims[0], -k)[-k:]

        return indices[np.argsort(sims[0][indices])[::-1]]

    def get_edges_by_reltype_vector_search(self, qa_row_id: int, subgraph_rels: DataFrame):
        subgraph_df = subgraph_rels.copy()
        subgraph_df['textEmbedding'] = subgraph_df['textEmbedding'].apply(np.array)
        grouped = subgraph_df.groupby(['srcType', 'relType', 'tgtType']).first().reset_index()

        prompt_emb = self.query_embedding_dict[qa_row_id]
        query_embedding = np.array(prompt_emb)
        grouped['cosine_sim'] = grouped['textEmbedding'].apply(lambda emb: cosine_similarity(query_embedding, emb.reshape(1,-1))[0][0])
        grouped['cosine_sim'] = grouped['cosine_sim'].astype(float)

        top_2_groups = grouped.nlargest(2, 'cosine_sim')[['srcType', 'relType', 'tgtType']]
        first_group = top_2_groups.iloc[0]
        top_edges = subgraph_rels[subgraph_rels[['srcType', 'relType', 'tgtType']].apply(tuple, axis=1) == tuple(first_group)].index.to_numpy()

        if len(top_2_groups) > 1:
            second_group = top_2_groups.iloc[1]
            second_edges = subgraph_rels[subgraph_rels[['srcType', 'relType', 'tgtType']].apply(tuple, axis=1) == tuple(
                second_group)].index.to_numpy()

        else:
            second_edges = np.array([])

        return top_edges, second_edges

    def _chunks(self, xs, n=500):
        n = max(1, n)
        return [xs[i:i + n] for i in range(0, len(xs), n)]

    # def _embed(self, doc_list, chunk_size=500):
    #     embeddings = []
    #     for docs in self._chunks(doc_list, chunk_size):
    #         try:
    #             embeddings.extend(self.embedding_model.embed_documents(docs))
    #         except Exception as e:
    #             print(f"Error while embedding the documents: {e}")
    #             exit()
    #         time.sleep(1)
    #     return embeddings

    def _embed(self, reltype_list: List[str]):
        return [self.reltype_embedding_dict[reltype] for reltype in reltype_list]

    def _embed_triplet(self, src_type_list: List[str], rel_type_list: List[str], tgt_type_list: List[str]):
        return [self.triplet_embedding_dict[(src_type, rel_type, tgt_type)]['embedding'] for src_type, rel_type, tgt_type in zip(src_type_list, rel_type_list, tgt_type_list)]

    def get_textual_nodes(self, node_ids: List, driver: Driver) -> DataFrame:
        res = driver.execute_query("""
        UNWIND $nodeIds AS nodeId
        MATCH(node:_Entity_ {nodeId:nodeId})
        RETURN node.nodeId AS nodeId, node.name AS name, node.details AS description, node.textEmbedding AS textEmbedding
        """,
                                   parameters_={"nodeIds": node_ids})
        return pd.DataFrame([rec.data() for rec in res.records])

    def get_textual_edges(self, node_pairs: List[Tuple[int, int]], driver: Driver) -> DataFrame:
        res = driver.execute_query("""
        UNWIND $node_pairs AS pair
        MATCH(src:_Entity_ {nodeId:pair[0]})-[e]->(tgt:_Entity_ {nodeId:pair[1]})
        RETURN src.nodeId AS src, type(e) AS edge_attr, tgt.nodeId AS dst
        """,
                                   parameters_={"node_pairs": node_pairs})
        return pd.DataFrame([rec.data() for rec in res.records])