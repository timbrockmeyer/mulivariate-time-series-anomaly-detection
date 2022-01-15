import torch
import torch.nn.functional as F
from torch import nn

from ..layers import SingleEmbedding

class RecurrentModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = config.device

        self.num_nodes = config.num_nodes
        self.horizon = config.horizon
        self.topk = config.topk
        self.embed_dim = config.embed_dim
        self.lags = config.window_size

        # dummy
        self.embedding = SingleEmbedding(1, 1, 1).to(self.device)

        # encoder lstm
        self.lstm = nn.LSTM(self.num_nodes, 512, 2, batch_first=True, dropout=0.25)
        
        # decoder lstm
        self.cell1 = nn.LSTMCell(self.num_nodes, 512)
        self.cell2 = nn.LSTMCell(512, 512)

        # linear prediction layer
        self.pred = nn.Linear(512, self.num_nodes)

    def get_graph(self):
        return self.embedding.get_A()
    
    def get_embedding(self):
        return self.embedding.get_E()


    def forward(self, window):
        # batch stacked window; input shape: [num_nodes*batch_size, lags]
        N = self.num_nodes # number of nodes
        T = self.lags # number of input time steps
        B = window.size(0) // N # batch size

        x = window.view(B, T, N)

        # encoder
        _, (h, c) = self.lstm(x) # -> (B, T, H), (2, B, H), (2, B, H)
        # get hidden and cell states for each layer
        h1 = h[0, ...].squeeze(0)
        h2 = h[1, ...].squeeze(0)
        c1 = c[0, ...].squeeze(0)
        c2 = c[1, ...].squeeze(0)

        # decoder
        predictions = []
        for _ in range(self.horizon-1):
            pred = self.pred(h2)
            predictions.append(pred.view(-1, 1))
            # layer 1
            h1, c1 = self.cell1(pred, (h1, c1))
            h1 = F.dropout(h1, 0.2)
            c1 = F.dropout(c1, 0.2)
            # layer 2
            h2, c2 = self.cell2(h1, (h2, c2))
        # final prediction
        pred = self.pred(h2).view(-1, 1)
        predictions.append(pred)

        return torch.cat(predictions, dim=1)


