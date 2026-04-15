"""
Combined GCN + Graph Transformer encoder — drop-in replacement for GAT.

Architecture:
  First half of layers → GCNConv  (local neighbourhood aggregation)
  Second half          → TransformerConv  (global attention)
  Learned fusion gate  → α·GCN_out + (1-α)·GT_out  (per-layer residual)

Compatible with GRetriever:
    forward(x, edge_index, batch=None, edge_attr=None) -> [N, out_channels]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, TransformerConv


class _GCNLayer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = GCNConv(dim, dim, add_self_loops=True, normalize=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        out = F.relu(self.norm(self.conv(x, edge_index)))
        return F.dropout(out, p=self.dropout, training=self.training) + x


class _GTLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TransformerConv(dim, dim // heads, heads=heads, dropout=dropout, concat=True, beta=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), edge_index)
        x = x + self.ffn(self.norm2(x))
        return x


class CombinedEncoder(nn.Module):
    """
    GCN layers followed by Graph Transformer layers with a learned fusion gate.

    num_layers is split evenly between GCN and GT (if odd, GT gets the extra layer).
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

        n_gcn = num_layers // 2
        n_gt  = num_layers - n_gcn

        self.gcn_layers = nn.ModuleList(
            [_GCNLayer(hidden_channels, dropout) for _ in range(n_gcn)]
        )
        self.gt_layers = nn.ModuleList(
            [_GTLayer(hidden_channels, heads, dropout) for _ in range(n_gt)]
        )

        # Learned scalar gate — fuses GCN branch and GT branch outputs
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

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
        batch: Tensor = None,      # accepted but not used
        edge_attr: Tensor = None,  # accepted but not used
    ) -> Tensor:
        x = self.input_proj(x)

        # GCN branch
        gcn_out = x
        for layer in self.gcn_layers:
            gcn_out = layer(gcn_out, edge_index)

        # Graph Transformer branch
        gt_out = x
        for layer in self.gt_layers:
            gt_out = layer(gt_out, edge_index)

        # Fuse
        alpha = torch.sigmoid(self.fusion_gate)
        x = alpha * gcn_out + (1.0 - alpha) * gt_out

        return self.out_norm(self.out_proj(x))
