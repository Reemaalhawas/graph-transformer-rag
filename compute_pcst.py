import time
from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

def assign_prizes_topk(
    base_subgraph_data: Data,
    top_node_ids: List,
    top_edge_ids: np.ndarray,
) -> tuple[torch.tensor, torch.tensor]:
    # Assign prizes to nodes based on their similarity to the question
    top_node_count = len(top_node_ids)
    n_prizes = torch.zeros(base_subgraph_data.num_nodes)
    n_prizes[top_node_ids] = torch.linspace(4, 0, steps=len(top_node_ids)).float()

    # Assign prizes to edges based on their similarity to the question
    top_edges_count = len(top_edge_ids)
    e_prizes = torch.zeros(base_subgraph_data.num_edges)
    for i, top_edge_id in enumerate(top_edge_ids):
        e_prizes[top_edge_id] = top_edges_count - i

    return n_prizes, e_prizes


def assign_prizes_modified(
    base_subgraph_data: Data,
    top_node_ids: List,
    top_edge_ids: np.ndarray,
    second_edge_ids: np.ndarray,
    top_prize: float = 0.8,
    second_prize: float = 0.5,
) -> tuple[torch.tensor, torch.tensor]:
    # Assign prizes to nodes based on their similarity to the question
    interval = 4/len(top_node_ids)
    n_prizes = torch.zeros(base_subgraph_data.num_nodes)
    n_prizes[top_node_ids] = torch.linspace(4, 0, steps=len(top_node_ids)).float()

    # Assign prizes to edges based on their similarity to the question
    e_prizes = torch.zeros(base_subgraph_data.num_edges)
    for _, top_edge_id in enumerate(top_edge_ids):
        e_prizes[top_edge_id] = top_prize
    for _, second_edge_id in enumerate(second_edge_ids):
        e_prizes[second_edge_id] = second_prize

    return n_prizes, e_prizes


# def assign_prizes(
#     base_subgraph_data: Data,
#     prized_node_ids: List,
#     prized_edge_ids: List[np.ndarray],
#     max_node_prize: float = 4,
#     edge_prizes: List[float] = [0.5],
# ) -> tuple[torch.tensor, torch.tensor]:
#
#     n_prizes = torch.zeros(base_subgraph_data.num_nodes)
#     n_prizes[prized_node_ids] = torch.linspace(max_node_prize, 0, steps=len(prized_node_ids)).float()
#
#     assert len(prized_edge_ids) == len(edge_prizes)
#     e_prizes = torch.zeros(base_subgraph_data.num_edges)
#     for group, edge_id_group in enumerate(prized_edge_ids):
#         for edge_id in edge_id_group:
#             e_prizes[edge_id] = edge_prizes[group]
#
#     return n_prizes, e_prizes


def compute_pcst(
    base_subgraph_data: Data,
    n_prizes: torch.tensor,
    e_prizes: torch.tensor,
    cost_e: float = 0.5,
) -> Data:

    from pcst_fast import pcst_fast

    root = -1
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0

    t = time.time()
    # Preparing data for pcst_fast
    costs = []
    edges = []
    virtual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(base_subgraph_data.edge_index.t().numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = base_subgraph_data.num_nodes + len(virtual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            virtual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(virtual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs + virtual_costs)
        edges = np.array(edges + virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters,
                                pruning, verbosity_level)

    # Parsing result from pcst_fast
    selected_nodes = vertices[vertices < base_subgraph_data.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= base_subgraph_data.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= base_subgraph_data.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges + virtual_edges)

    edge_index = base_subgraph_data.edge_index[:, selected_edges]
    selected_nodes = np.unique(
        np.concatenate(
            [selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]

    data = Data(
        edge_index=torch.tensor([src, dst]),
    )

    return data, mapping, selected_nodes, edge_index