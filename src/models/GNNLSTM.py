import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LSTM, LSTMCell
import math

from ..utils.device import get_device

from torch_geometric.nn import ARMAConv
from torch_geometric.nn import Sequential

from ..layers import DoubleEmbedding, SingleEmbedding


class GNNLSTM(torch.nn.Module):
    '''
    Anomaly detection neural network model for multivariate sensor time series.
    Graph structure is randomly initialized and learned during training.
    Uses an attention layer that scores the attention weights for the input
    time series window and the sensor embedding vector.

    Args:
        args (dict): Argparser with config information.
    '''
    def __init__(self, args):
        super().__init__()

        self.device = args.device

        self.num_nodes = args.num_nodes
        self.horizon = args.horizon
        self.topk = args.topk
        self.embed_dim = args.embed_dim
        self.lags = args.window_size

        # model parameters
        channels = 32 # channel == node embedding size because they are added
        hidden_size = 512
  
        # learned graph embeddings
        self.graph_embedding = SingleEmbedding(self.num_nodes, channels, topk=self.topk, warmup_epochs=10)
        
        # encoder
        self.tgconv = TGConv(1, channels)
        self.lstm = LSTM(channels*self.num_nodes, hidden_size, 2, batch_first=True, dropout=0.20)
        
        # decoder
        self.gnn = ARMAConv(
            in_channels=1,
            out_channels=channels,
            num_stacks=1,
            num_layers=1,
            act=nn.GELU(),
            dropout=0.2,
        )
        self.cell1 = LSTMCell(self.num_nodes*channels, hidden_size)
        self.cell2 = LSTMCell(hidden_size, hidden_size)

        # linear prediction layer
        self.pred = nn.Linear(hidden_size, self.num_nodes)

        # cached offsets for batch stacking for each batch_size and number of edges
        self.batch_edge_offset_cache = {}

        # initial graph
        self._edge_index, self.edge_attr, self.A = self.graph_embedding()
    
    def get_graph(self):
        return self.graph_embedding.get_A()
    
    def get_embedding(self):
        return self.graph_embedding.get_E()

    def forward(self, window):
        # batch stacked window; input shape: [num_nodes*batch_size, lags]
        N = self.num_nodes # number of nodes
        T = self.lags # number of input time steps
        B = window.size(0) // N # batch size
        
        # get learned graph representation
        edge_index, edge_attr, _ = self.graph_embedding()
        _, W = self.get_embedding()
        W = W.pop()

        # batching works by stacking graphs; creates a mega graph with disjointed subgraphs 
        # for each input sample. E.g. for a batch of B inputs with 51 nodes each;
        # samples i in {0, ..., B} => node indices [0...50], [51...101], [102...152], ... ,[51*B...50*51*B] 
        # => node indices for sample i = [0, ..., num_nodes-1] + (i*num_nodes)
        num_edges = len(edge_attr)
        try:
            batch_offset = self.batch_edge_offset_cache[(B, num_edges)]
        except:
            batch_offset = torch.arange(0, N * B, N).view(1, B, 1).expand(2, B, num_edges).flatten(1,-1).to(self.device)
            self.batch_edge_offset_cache[(B, num_edges)] = batch_offset
        # repeat edge indices B times and add i*num_nodes where i is the input index
        batched_edge_index = edge_index.unsqueeze(1).expand(2, B, -1).flatten(1, -1) + batch_offset
        # repeat edge weights B times
        batched_edge_attr = edge_attr.unsqueeze(0).expand(B, -1).flatten() 

        # add node feature dimension to input
        x = window.unsqueeze(-1) # (B*N, T, 1)

        ### ENCODER
        # GNN layer; batch stacked output with C feature channels for each time step
        x = self.tgconv(x, batched_edge_index, batched_edge_attr) # (B*N, T, C)
        x = x.view(B, N, T, -1).permute(0, 2, 1, 3).contiguous() # -> (B, T, N, C)
        # add node embeddings to feature vector as node positional embeddings
        x = x + W # (B, T, N, C) + (N, C)
        # concatenate node features for LSTM input
        x = x.view(B, T, -1) # -> (B, T, N*C)
        # LSTM layer
        h, (h_n, h_n) = self.lstm(x) # -> (B, T, H), (2, B, H), (2, B, H)
        # get hidden and cell states for each layer
        h1 = h_n[0, ...].squeeze(0)
        h2 = h_n[1, ...].squeeze(0)
        c1 = h_n[0, ...].squeeze(0)
        c2 = h_n[1, ...].squeeze(0)

        # TODO: try attention on h

        ### DECODER
        predictions = []
        # if prediction horizon > 1, iterate through decoder LSTM step by step
        for _ in range(self.horizon-1):
            # single decoder step per loop iteration
            pred = self.pred(h2).view(-1, 1)
            predictions.append(pred)

            # GNN layer analogous to encoder without time dimension
            x = self.gnn(pred, batched_edge_index, batched_edge_attr)
            x = x.view(B, N, -1) + W
            x = x.view(B, -1)
            # LSTM layer 1
            h1, c1 = self.cell1(x, (h1, c1))
            h1 = F.dropout(h1, 0.2)
            c1 = F.dropout(c1, 0.2)
            # LSTM layer 2
            h2, c2 = self.cell2(h1, (h2, c2))
        # final prediction
        pred = self.pred(h2).view(-1, 1)
        predictions.append(pred)

        return torch.cat(predictions, dim=1)


class TGConv(nn.Module):
    r'''
    Parallel graph convolution for multiple time steps.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        p (float): Dropout value between 0 and 1
    '''

    def __init__(self, in_channels: int, out_channels: int, p: float = 0.0):
        super(TGConv, self).__init__()

        self.device = get_device()

        self.graph_conv = ARMAConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_stacks=1,
            num_layers=1,
            act=nn.GELU(),
            dropout=p,
        )

        # cached offsets for temporal batch stacking for each batch_size and number of edges
        self.batch_edge_offset_cache = {}

    def forward(self, x: torch.FloatTensor, edge_index: torch.FloatTensor, edge_attr: torch.FloatTensor = None) -> torch.FloatTensor:
        '''
        Forward pass through temporal convolution block.

        Input data of shape: (batch, time_steps, in_channels).
        Output data of shape: (batch, time_steps, out_channels).
        '''
      
        # input dims
        BN, T, C = x.shape # (batch*nodes, time, in_channels)
        N = edge_index.max().item() + 1 # number of nodes in the batch stack
        
        # batch stacking the temporal dimension to create a mega giga graph consisting of batched temporally-stacked graphs
        # analogous to batch stacking in main GNN, see description there.
        x = x.contiguous().view(-1, C) # (B*N*T, C)

        # create temporal batch edge and weight lists
        num_edges = len(edge_attr)
        try:
            batch_offset = self.batch_edge_offset_cache[(BN, num_edges)]
        except:
            batch_offset = torch.arange(0, BN*T, N).view(1, T, 1).expand(2, T, num_edges).flatten(1,-1).to(x.device)
            self.batch_edge_offset_cache[(BN, num_edges)] = batch_offset
        # repeat edge indices T times and add offset for the edge indices
        temporal_batched_edge_index = edge_index.unsqueeze(1).expand(2, T, -1).flatten(1, -1) + batch_offset
        # repeat edge weights T times
        temporal_batched_edge_attr = edge_attr.unsqueeze(0).expand(T, -1).flatten() 

        # GNN with C output channels
        x = self.graph_conv(x, temporal_batched_edge_index, temporal_batched_edge_attr) # (B*N*T, C)
        x = x.view(BN, T, -1) # -> (B*N, T, C)
        
        return x
