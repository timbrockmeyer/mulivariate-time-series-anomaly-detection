import torch
import torch.nn.functional as F
from torch import nn
import math

from ..utils.device import get_device

from torch_geometric.nn import ARMAConv
from torch_geometric.nn import Sequential

from ..layers import DoubleEmbedding


class ConvSeqAttentionModel(torch.nn.Module):
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
  
        # learned graph embeddings
        self.graph_embedding = DoubleEmbedding(self.num_nodes, self.embed_dim, topk=self.topk, type='uni', warmup_epochs=50).to(self.device)

        # model parameters
        kernels = [5, 3]
        channels = [16, 32]
        hidden_dim = 64

        # GNN ENCODER ::: outputs one hidden state for each time step
        self.conv_encoder = Sequential('x, idx, attr', [ 
            (STConv(1, channels[0], channels[0], kernels[0], p=0.2, padding=True, residual=True), 'x, idx, attr -> x'), 
            (STConv(channels[0], channels[1], channels[1], kernels[1], p=0.2, padding=True, residual=True), 'x, idx, attr -> x'),
            (STConv(channels[1], hidden_dim, hidden_dim, kernels[1], p=0.2, padding=True, residual=True), 'x, idx, attr -> x')
        ]) 
               
        # linear transformation of encoder hidden states for alignment scores
        self.alignment_W = nn.Linear(hidden_dim, hidden_dim)

        # GNN DECODER ::: outputs single vector hidden state
        self.decoder_window_length = sum(kernels) - len(kernels) + 1
        self.conv_decoder = Sequential('x, idx, attr', [ 
            (nn.Sequential(
                nn.Conv1d(1, 2*channels[0], kernels[0]),
                nn.BatchNorm1d(2*channels[0]),
                nn.GLU(dim=1), 
                nn.Dropout(0.2),
                nn.Conv1d(channels[0], 2*channels[1], kernels[1]),
                nn.BatchNorm1d(2*channels[1]),
                nn.GLU(dim=1), 
                nn.Dropout(0.2),
                nn.Flatten(1, -1),), 'x, -> x'), 
            (ARMAConv(
                in_channels=channels[1],
                out_channels=hidden_dim,
                num_stacks=1,
                num_layers=1,
                act=nn.GELU(),
                dropout=0.2,), 'x, idx, attr -> x'),
            (nn.LayerNorm(hidden_dim), 'x -> x'),
        ])

        # prediction layer
        pred_channels = 2*hidden_dim
        self.pred = Sequential('x, idx, attr', [ 
            (nn.Linear(pred_channels, 1), 'x -> x'),
        ])

        # absolute positional embeddings based on sine and cosine functions
        position = torch.arange(1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(1e5) / hidden_dim))
        pe = torch.zeros(1000, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term) / 1e9
        pe[:, 1::2] = torch.cos(position * div_term) / 1e9
        self.positional_embedding = F.dropout(pe[:self.lags - self.decoder_window_length], 0.05).to(self.device) # vector with PEs for each input timestep

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
        window = window.unsqueeze(1) # (B, 1, T)
        encoder_window = window[..., :-self.decoder_window_length] # encoder takes beginning of input window
        decoder_window = window[..., -self.decoder_window_length:] # decoder takes the end

        # hidden states for all input time steps
        h_encoder = self.conv_encoder(encoder_window, batched_edge_index, batched_edge_attr) # (B, C, T)
        
        # add small positional encoding value
        h_encoder += self.positional_embedding.T
        
        # multistep prediction
        predictions = []
        for _ in range(self.horizon):
            # decoder hidden state
            h_decoder = self.conv_decoder(decoder_window, batched_edge_index, batched_edge_attr).unsqueeze(1) # -> (B, 1, C)
            # transformation of encoder states
            a = self.alignment_W(h_encoder.permute(0,2,1)).permute(0,2,1) # W @ H_encoder, shape -> (B, C, T)
            # compute alignment vector from decoder transformed encoder hidden states
            score = h_decoder @ a # (B, 1, C) @ (B, C, T) -> (B, 1, T)
            # attention weights for each time step
            alpha = F.softmax(score, dim=2) # -> (B, 1, T)
            # context vector
            context = torch.sum(alpha * h_encoder, dim=2)   # -> (B, C)            
            # concatination of context vector and decoder hidden state
            context = torch.cat([context, h_decoder.squeeze(1)], dim=1) # -> (B, 2C)   
            # layer normalization after adding all components
            context = F.layer_norm(context, tuple(context.shape[1:]))
            # single step prediction 
            y_pred = self.pred(context, batched_edge_index, batched_edge_attr).view(-1, 1) # column vector
            predictions.append(y_pred)
            # decoder input for the next step
            decoder_window = torch.cat([decoder_window[..., 1:], y_pred.detach().unsqueeze(1)], dim=-1)
        
        # full output prediction vector
        pred = torch.cat(predictions, dim=1) # row = node, column = time

        return pred

        # return window[..., -1].view(-1, 1).repeat(1, self.horizon)


class STConv(nn.Module):
    r'''Spatio-Temporal convolution block.

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        kernel_size (int): Convolutional kernel size.
    '''

    def __init__(self, in_channels: int, temporal_channels: int, spatial_channels: int, kernel_size: int = 3, padding: bool = True, residual: bool = True, p: float = 0.0):
        super(STConv, self).__init__()

        self.padding = padding
        self.residual = residual

        self.device = get_device()

        if residual:
            self.res = nn.Conv1d(in_channels, spatial_channels, 1)

        if padding:
            self.p1d = (kernel_size-1, 0)

        # absolute positional embeddings based on sine and cosine functions
        position = torch.arange(1000).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, temporal_channels, 2) * (-math.log(1e5) / temporal_channels))
        pe = torch.zeros(1000, temporal_channels)
        pe[:, 0::2] = torch.sin(position * div_term) / 100 
        pe[:, 1::2] = torch.cos(position * div_term) / 100
        self.positional_embedding = F.dropout(pe, 0.05).to(self.device) # vector with PEs for each input timestep
        
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, 2*temporal_channels, kernel_size),
            nn.BatchNorm1d(2*temporal_channels),
            nn.GLU(dim=1),
            nn.Dropout(p),
        )
        self.graph_conv = ARMAConv(
            in_channels=temporal_channels,
            out_channels=spatial_channels,
            num_stacks=1,
            num_layers=1,
            act=nn.GELU(),
            dropout=p,
        )

        # cached offsets for temporal batch stacking for each batch_size and number of edges
        self.batch_edge_offset_cache = {}

    def forward(self, x: torch.FloatTensor, edge_index: torch.FloatTensor, edge_attr: torch.FloatTensor = None) -> torch.FloatTensor:
        '''Forward pass through temporal convolution block.

        Input data of shape: (batch, in_channels, time_steps).
        Output data of shape: (batch, out_channels, time_steps).
        '''

        # input shape (batch*num_nodes, in_channels, time)
        
        if self.residual:
            res = self.res(x)

        if self.padding:
            x = F.pad(x, self.p1d, "constant", 0)
        
        # temporal aggregation
        x = self.temporal_conv(x)
        # dims after temporal convolution
        BN, C, T = x.shape # (batch*nodes, out_channels, time)
        N = edge_index.max().item() + 1 # number of nodes in the batch stack

        # positional encoding for every time step
        pe = self.positional_embedding[:T].T
        # print(x.mean().item(), pe.mean().item()) # balance layer output and embeddings
        x += pe 
        
        # batch stacking the temporal dimension to create a mega giga graph consisting of batched temporally-stacked graphs
        # analogous to batch stacking in main GNN, see description there.
        x = x.view(-1, C) # (B*N*T, C)

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

        x = self.graph_conv(x, temporal_batched_edge_index, temporal_batched_edge_attr)  
        x = x.view(BN, -1, T)  

        # add residual connection
        x = x + res if self.residual else x

        # layer normalization
        return F.layer_norm(x, tuple(x.shape[1:]))
