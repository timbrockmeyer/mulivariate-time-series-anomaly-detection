import math
import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch.autograd import Variable
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils.to_dense_adj import to_dense_adj

from torch_geometric.nn.inits import glorot, zeros


class EmbeddingAttention(MessagePassing):
    '''
    GATConv layer that computes concatinated attention scores for a graph time series window
    and corresponding node embedding values.
    Modification of the implementation of GATConv in pytorch geometric. 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATConv
    '''
 
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0,
                 add_self_loops=True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # transformations on source and target nodes (will be the same in sensor network)
        self.lin_src = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = self.lin_src

        # learnable parameters to compute attention coefficients
        # double number of parameters; for node features and sensor embedding
        self.att_src = Parameter(torch.Tensor(1, heads, 2*out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, 2*out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.lin_src.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, embedding, size=None, return_attention_weights=False):
       
        H, C = self.heads, self.out_channels

        # transform input node features
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)
        x = (x_src, x_dst)

        # shape [num_nodes*batch_size, embed_dim] -> [num_nodes*batch_size, heads, embed_dim]
        assert embedding.size(1) == C
        emb_src = emb_dst = embedding.unsqueeze(1).expand(-1, H, C)

        # combined representation of node features and embedding
        src = torch.cat([x_src, emb_src], dim=2)
        dst = torch.cat([x_dst, emb_dst], dim=2)
        # compute node-level attention coefficients
        alpha_src = (src * self.att_src).sum(dim=-1)
        alpha_dst = (dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            num_nodes = x_src.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        
        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        
        if return_attention_weights:
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_j * alpha.unsqueeze(-1)

        return msg

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)