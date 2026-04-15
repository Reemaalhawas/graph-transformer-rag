# models/gnn_variants.py
"""
GNN architecture variants — all PyG official implementations.
All expose out_channels and accept:
    forward(x, edge_index, edge_attr=None, batch=None)
so they are drop-in compatible with PyG GRetriever.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool


class GCN(nn.Module):
    """
    Graph Convolutional Network — PyG GCNConv.
    Equal-weight message passing. Ignores edge_attr entirely.
    Structural baseline: lowest expressiveness, fastest training.
    Paper: Kipf & Welling (2017) — Semi-Supervised Classification
           with Graph Convolutional Networks.
    """
    def __init__(self, in_channels=1536, hidden_channels=1024,
                 num_layers=4, out_channels=1024):
        super().__init__()
        self.out_channels = out_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # edge_attr intentionally ignored — GCN does not use edge features
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        x = self.convs[-1](x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x


class GraphTransformer(nn.Module):
    """
    Graph Transformer — PyG TransformerConv.
    Full multi-head attention. Natively uses edge_attr in attention.
    Most expressive. Uses edge_dim=1536 to match OpenAI ada-002 embeddings.
    beta=True adds a learnable skip connection.
    Paper: Shi et al. (2021) — Masked Label Prediction:
           Unified Message Passing Model for Semi-Supervised Classification.
    """
    def __init__(self, in_channels=1536, hidden_channels=1024,
                 num_layers=4, heads=4, out_channels=1024, edge_dim=1536):
        super().__init__()
        self.out_channels = out_channels
        head_dim = hidden_channels // heads
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        # Input layer
        self.convs.append(TransformerConv(
            in_channels, head_dim, heads=heads,
            edge_dim=edge_dim, dropout=0.1, beta=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                hidden_channels, head_dim, heads=heads,
                edge_dim=edge_dim, dropout=0.1, beta=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        # Output layer — single head
        self.convs.append(TransformerConv(
            hidden_channels, out_channels, heads=1,
            edge_dim=edge_dim, dropout=0.1, beta=True))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index, edge_attr)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x
