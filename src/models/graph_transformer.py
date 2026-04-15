"""
Graph Transformer encoder — drop-in replacement for PyG's GAT inside GRetriever.

Implements a multi-layer Graph Transformer using PyG's TransformerConv
with multi-head attention, optional edge features, pre-LN, and FFN sublayers.

Interface matches PyG's BasicGNN so it works seamlessly with GRetriever:
    forward(x, edge_index, batch=None, edge_attr=None) -> Tensor [N, out_channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import TransformerConv


class _GraphTransformerLayer(nn.Module):
    """One Graph Transformer block: pre-LN attention + pre-LN FFN + residuals."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = TransformerConv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True,  # heads concatenated → stays at dim
            beta=True,    # gating between self and neighbour
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), edge_index)
        x = x + self.ffn(self.norm2(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """
    Stack of Graph Transformer layers.

    Compatible with GRetriever — same call signature as PyG's GAT:
        forward(x, edge_index, batch=None, edge_attr=None) -> [N, out_channels]
    """

    def __init__(
        self,
        in_channels: int = 1536,
        hidden_channels: int = 1536,
        out_channels: int = 1536,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_proj = (
            nn.Linear(in_channels, hidden_channels)
            if in_channels != hidden_channels
            else nn.Identity()
        )

        # All layers operate at hidden_channels; last layer projects to out_channels
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dim = hidden_channels if i < num_layers - 1 else out_channels
            # Adjust hidden if last layer changes dimension
            in_dim = hidden_channels if i == 0 else (hidden_channels if i < num_layers - 1 else hidden_channels)
            self.layers.append(_GraphTransformerLayer(dim=hidden_channels, heads=heads, dropout=dropout))

        self.out_proj = (
            nn.Linear(hidden_channels, out_channels)
            if hidden_channels != out_channels
            else nn.Identity()
        )
        self.out_norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,   # accepted but not used (GRetriever passes it)
        edge_attr: Tensor = None,  # accepted but not used by TransformerConv here
    ) -> Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.out_norm(self.out_proj(x))
        return x
