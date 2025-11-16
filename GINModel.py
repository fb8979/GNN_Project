from torch_geometric.nn import GINConv
import torch.nn as nn
import torch.nn.functional as F

class GINModel(nn.Module):
    """Graph Isomorphism Network for node regression"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        """
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output dimension
            num_layers: Number of GIN layers
            dropout: Dropout probability
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        
        for _ in range(num_layers - 2):
            nn_mid = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_mid))
        
        nn_last = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.convs.append(GINConv(nn_last))
    
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
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x
    
    def __repr__(self):
        return f"GIN(layers={self.num_layers}, dropout={self.dropout})"