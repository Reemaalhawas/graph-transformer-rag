import torch
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

class GRetriever(torch.nn.Module):
    def __init__(self, llm, gnn):
        super().__init__()
        self.llm = llm
        self.gnn = gnn
        # Project GNN hidden size (e.g. 128) to LLM hidden size (e.g. 4096)
        self.projector = Linear(gnn.out_channels, llm.model.config.hidden_size)

    def forward(self, question, x, edge_index, batch, label, edge_attr=None, desc=None):
        # x are your node embeddings
        node_embeddings = self.gnn(x, edge_index, edge_attr)
        # Pool to get one graph vector
        graph_embedding = global_mean_pool(node_embeddings, batch)
        # Align with LLM space
        projected_embedding = self.projector(graph_embedding)
        # LLM processes text + graph
        return self.llm(question, projected_embedding, label, desc)

    def inference(self, question, x, edge_index, batch, edge_attr=None, desc=None):
        node_embeddings = self.gnn(x, edge_index, edge_attr)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        projected_embedding = self.projector(graph_embedding)
        return self.llm.inference(question, projected_embedding, desc)