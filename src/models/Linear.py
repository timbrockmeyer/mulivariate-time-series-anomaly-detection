import torch
from torch import nn

from ..layers import SingleEmbedding

class LinearModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.device = args.device

        self.num_nodes = args.num_nodes
        self.horizon = args.horizon
        self.topk = args.topk
        self.embed_dim = args.embed_dim
        self.lags = args.window_size

        self.embedding = SingleEmbedding(self.num_nodes, self.embed_dim, topk=self.topk)
        
        self.lin = nn.Sequential(
            nn.Linear(self.lags, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.pred = nn.Sequential(
            nn.Linear(1024, self.horizon)
        )

        # initial graph
        self._edge_index, self.edge_attr, self.A = self.embedding()

    def get_graph(self):
        return self.embedding.get_A()
    
    def get_embedding(self):
        return self.embedding.get_E()

    def forward(self, x):
        # input sizes
        N = self.num_nodes
        B = x.size(0) // N # batch size
        
        x = self.lin(x)
        pred = self.pred(x).view(B*N, -1)

        return pred
        