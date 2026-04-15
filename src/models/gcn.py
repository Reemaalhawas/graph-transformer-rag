"""
GCN encoder — drop-in replacement for PyG's GAT inside GRetriever.

Uses PyG's built-in GCN (BasicGNN subclass) which already has the
correct forward(x, edge_index, batch, edge_attr) signature that
GRetriever expects.
"""

from torch_geometric.nn import GCN


def build_gcn(
    in_channels: int = 1536,
    hidden_channels: int = 1536,
    out_channels: int = 1536,
    num_layers: int = 4,
    dropout: float = 0.0,
    **kwargs,
) -> GCN:
    """
    Returns a PyG GCN model compatible with GRetriever.

    Args:
        in_channels     : Input feature dimension  (default 1536 = Ada-002)
        hidden_channels : Hidden layer dimension
        out_channels    : Output node embedding dimension
        num_layers      : Number of GCN layers
        dropout         : Dropout rate

    Returns:
        torch_geometric.nn.GCN  — call as gnn(x, edge_index, batch=batch)
    """
    return GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
    )
