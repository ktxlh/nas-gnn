# Update Z, E, P (NGE Eq. 5, 9, and 10)
import torch
from torch import sigmoid
from torch.nn import Embedding, Linear, Module, ReLU, Sequential
from torch.nn.functional import softmax
from torch_geometric.nn import GCNConv
from torch_geometric.nn import Sequential as GeoSequential

from operations import OPS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NGE(Module):
    # NGE Eq 5-8
    def __init__(self, num_nodes, weighted=False):
        super(NGE, self).__init__()
        self.num_nodes = num_nodes
        self.num_ops = len(OPS)
        self.weighted = weighted
        
        D, F, K = 64, 64, 64
        H_NG = 128
        H_EE = 64
        
        self.node_embed = Embedding(num_nodes, D)
        self.node_gcn = GeoSequential('x, edge_index, weights', [
            (GCNConv(D, H_NG), 'x, edge_index, weights -> x'),
            ReLU(inplace=True),
            (GCNConv(H_NG, H_NG), 'x, edge_index, weights -> x'),
            ReLU(inplace=True),
            (GCNConv(H_NG, F), 'x, edge_index, weights -> x'),
            ReLU(inplace=True),
        ])
        self.edge_embed = Sequential(
            Linear(F * 2, H_EE),
            ReLU(),
            Linear(H_EE, K),
            ReLU(),
        )
        self.op_prob = Linear(K, self.num_ops)
        
        # Constant input
        self.indices = torch.arange(num_nodes, requires_grad=False).long().to(device)
        num_edges = (2 + num_nodes - 1) * (num_nodes - 2) // 2
        self.edges = torch.zeros(2, num_edges, requires_grad=False).long().to(device)
        self.weights = torch.ones(num_edges, requires_grad=False).to(device)
        offset = 0
        for i in range(2, num_nodes):
            for j in range(0, i):
                self.edges[0, offset + j] = i
                self.edges[1, offset + j] = j
            offset += i

    def forward(self):
        # OUT:  num_nodes x num_nodes x num_ops
        h = self.node_embed(self.indices)
        h = self.node_gcn(h, self.edges, self.weights)

        p = torch.zeros(self.edges.shape[1], self.num_ops, requires_grad=True).to(device)
        offset = 0
        for i in range(2, self.num_nodes - 1):  # (0,1,)2,3,4,5(,6)
            for j in range(0, i):
                e = self.edge_embed(torch.cat([h[i], h[j]], dim=-1).unsqueeze(0))  # 1 x K
                o = self.op_prob(e)  # num_ops
                if self.weighted:
                    self.weights[offset + j] = sigmoid(o.detach().sum(dim=-1)).squeeze(0)
                p[offset + j] = softmax(o, dim=-1).squeeze(0)
            offset += i
        return p
        
