import torch
import torch.nn.functional as F 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class GCN_Net(torch.nn.Module):
    r""" GCN model from the "Semi-supervised Classification with Graph
    Convolutional Networks" paper, in ICLR'17.

    Arguments:
        in_channels (int): dimension of input.
        out_channels (int): dimension of output.
        hidden (int): dimension of hidden units, default=64.

    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=32):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        elif isinstance(data, dict):
            x, edge_index = data['x'], data['edge_index']
        else:
            raise TypeError('Unsupported data type!')

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x