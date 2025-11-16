import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    """Graph Attention Network for node regression"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension
            num_layers: Number of GAT layers
            heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout
                )
            )
        
        # Last layer (use concat=False to average attention heads)
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout
            )
        )
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (num_nodes x in_channels)
            edge_index: Edge list (2 x num_edges)
            
        Returns:
            Node predictions (num_nodes x out_channels)
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def __repr__(self):
        return f"GAT(layers={self.num_layers}, heads={self.heads}, dropout={self.dropout})"