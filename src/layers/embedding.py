from torch import nn
import torch.nn.functional as F
import torch
import math
from ..utils.device import get_device

class SingleEmbedding(nn.Module):
    r''' Layer for graph representation learning 
    using a linear embedding layer and cosine similarity 
    to produce an index list of edges for a fixed number of 
    neighbors for each node.

    Args:
        num_nodes (int): Number of nodes.
        embed_dim (int): Dimension of embedding.
        topk (int, optional): Number of neighbors per node.
    '''

    def __init__(self, num_nodes, embed_dim, topk=15, warmup_epochs = 20):
        super().__init__()

        self.device = get_device()

        self.topk = topk
        self.embed_dim = embed_dim
        self.num_nodes = num_nodes

        self.embedding = nn.Embedding(num_nodes, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

        self._A = None
        self._edges = None
        
        ### pre-computed index matrices 
        # square matrix for adjacency matrix indexing
        self._edge_indices = torch.arange(num_nodes).to(self.device).expand(num_nodes, num_nodes) # [[1,2,3,4,5], [1,2,3,4,5], ...] 
        # matrix containing column indices for the right side of a matrix - will be used to remove all but topk entries
        self._i = torch.arange(self.num_nodes).unsqueeze(1).expand(self.num_nodes, self.num_nodes - self.topk).flatten()

        # fully connected graph
        self._fc_edge_indices = torch.stack([self._edge_indices.T.flatten(), self._edge_indices.flatten()], dim=0)

        self.warmup_counter = 0
        self.warmup_durantion = warmup_epochs

    def get_A(self):
        if self._A is None:
            self.forward()
        return self._A
    
    def get_E(self):
        if self._edges is None:
            self.forward()
        return self._edges, [self.embedding.weight.clone()]

    def forward(self):
        W = self.embedding.weight.clone() # row vector represents sensor embedding

        eps = 1e-8 # avoid division by 0
        W_norm = W / torch.clamp(W.norm(dim=1)[:, None], min=eps)
        A = W_norm @ W_norm.t()

        # remove self loops
        A.fill_diagonal_(0)
        
        # remove negative scores
        A = A.clamp(0)

        if self.warmup_counter < self.warmup_durantion:
            edge_indices = self._fc_edge_indices
            edge_attr = A.flatten()

            self.warmup_counter += 1
        else:

            # topk entries 
            _, topk_idx = A.sort(descending=True)

            j = topk_idx[:, self.topk:].flatten()
            A[self._i, j] = 0
            
            # # row degree
            # row_degree = A.sum(1).view(-1, 1) + 1e-8 # column vector
            # col_degree = A.sum(0) + 1e-8 # row vector
            
            # # normalized adjacency matrix 
            # A /= torch.sqrt(row_degree) 
            # A /= torch.sqrt(col_degree) 

            msk = A > 0 # boolean mask

            edge_idx_src = self._edge_indices.T[msk] # source edge indices
            edge_idx_dst = self._edge_indices[msk] # target edge indices
            edge_attr = A[msk].flatten() # edge weights
        
            # shape [2, topk*num_nodes] tensor holding topk edge-index-pairs for each node 
            edge_indices = torch.stack([edge_idx_src, edge_idx_dst], dim=0)

        # save for later
        self._A = A
        self._edges = edge_indices

        return edge_indices, edge_attr, A


class DoubleEmbedding(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        num_nodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
        alpha (float, optional): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate
    """

    def __init__(self, num_nodes, embed_dim, topk=5, alpha=3, type='uni', warmup_epochs=20):

        super(DoubleEmbedding, self).__init__()

        self.device = get_device()

        assert type in ['bi', 'uni', 'sym']
        self.graph_type = type

        self.alpha = alpha

        self._embedding1 = nn.Embedding(num_nodes, embed_dim)
        self._embedding2 = nn.Embedding(num_nodes, embed_dim)
        self._linear1 = nn.Linear(embed_dim, embed_dim)
        self._linear2 = nn.Linear(embed_dim, embed_dim)

        nn.init.kaiming_uniform_(self._embedding1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._embedding2.weight, a=math.sqrt(5))

        self._topk = topk
        self._num_nodes = num_nodes

        # placeholders
        self._A = None
        self._edges = None
        self._M1 = self._embedding1.weight.clone()
        self._M2 = self._embedding2.weight.clone()

        ### pre-computed index matrices 
        # square matrix for adjacency matrix indexing
        self._edge_indices = torch.arange(num_nodes).to(self.device).expand(num_nodes, num_nodes) # [[1,2,3,4,5], [1,2,3,4,5], ...] 
        # row indices for entries that will be removed from adjacency matrix
        self._i = torch.arange(self._num_nodes).unsqueeze(1).expand(self._num_nodes, self._num_nodes - self._topk).flatten() 

        # fully connected graph
        self._fc_edge_indices = torch.stack([self._edge_indices.T.flatten(), self._edge_indices.flatten()], dim=0)

        self.warmup_counter = 0
        self.warmup_durantion = warmup_epochs
        
    def get_A(self):
        if self._A is None:
            self.forward()
        return self._A
    
    def get_E(self):
        if self._edges is None:
            self.forward()
        return self._edges, [self._M1, self._M2]

    def forward(self) -> torch.FloatTensor:
        """
        ...
        """

        M1 = self._embedding1.weight.clone()
        M2 = self._embedding2.weight.clone()

        self._M1 = M1.data.clone()
        self._M2 = M2.data.clone()

        M1 = torch.tanh(self.alpha * self._linear1(M1))
        M2 = torch.tanh(self.alpha * self._linear2(M2))

        if self.graph_type is 'uni':
            A = M1 @ M2.T - M2 @ M1.T # skew symmetric matrix (uni-directed)

        elif self.graph_type is 'bi': # unordered matrix (directed unconstraint)
            A = M1 @ M2.T

        elif self.graph_type is 'sym': # symmetric matrix (undirected)
            A = M1 @ M1.T - M2 @ M2.T
            # A = A.triu()
        
        # set negative values to zero
        A = F.relu(A)
        # no self loops
        A.fill_diagonal_(0)

        if self.warmup_counter < self.warmup_durantion:
            edge_indices = self._fc_edge_indices
            edge_attr = A.flatten()

            self.warmup_counter += 1
        else:
            # topk entries 
            _, idx = A.sort(descending=True)
            j = idx[:, self._topk:].flatten() # column indices of topk
            # remove all but topk 
            A[self._i, j] = 0
            
            # # node degrees (num incoming edges)
            # row_degree = A.sum(1).view(-1, 1) + 1e-8 # column vector
            # col_degree = A.sum(0) + 1e-8 # row vector
            
            # # normalized adjacency matrix 
            # A /= torch.sqrt(row_degree) 
            # A /= torch.sqrt(col_degree) 
        
            msk = A > 0 # boolean mask

            edge_idx_src = self._edge_indices.T[msk] # source edge indices
            edge_idx_dst = self._edge_indices[msk] # target edge indices
            edge_attr = A[msk].flatten() # edge weights
        
            # shape [2, topk*num_nodes] tensor holding topk edge-index-pairs for each node 
            edge_indices = torch.stack([edge_idx_src, edge_idx_dst], dim=0)

        # save for later
        self._A = A
        self._edges = edge_indices

        return edge_indices, edge_attr, A


class ProjectedEmbedding(nn.Module):
    r''' Layer for graph representation learning 
    using a linear embedding layer and cosine similarity 
    to produce an index list of edges for a fixed number of 
    neighbors for each node.

    Args:
        num_nodes (int): Number of nodes.
        embed_dim (int): Dimension of embedding.
        topk (int, optional): Number of neighbors per node.
    '''

    def __init__(self, num_nodes, num_node_features, embed_dim, topk=15):
        super().__init__()

        self.topk = topk
        self.embed_dim = embed_dim
        self.in_features = num_node_features

        self.device = get_device()
        
        self.embedding_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_node_features, 64),
                nn.ReLU(),
                nn.Linear(64, embed_dim)
            ) for _ in range(num_nodes)]
        )

        self.prev_embed = torch.empty((num_nodes, embed_dim), dtype=torch.float, requires_grad=True)
        nn.init.kaiming_uniform_(self.prev_embed, a=math.sqrt(5))

        self._A = None
        self._edges = None
        
        ### pre-computed index matrices 
        # square matrix for adjacency matrix indexing
        self._edge_indices = torch.arange(num_nodes).to(self.device).expand(num_nodes, num_nodes) # [[1,2,3,4,5], [1,2,3,4,5], ...] 
        # matrix containing column indices for the right side of a matrix - will be used to remove all but topk entries
        self._i = torch.arange(num_nodes).unsqueeze(1).expand(num_nodes, num_nodes - topk).flatten()

    def get_A(self):
        if self._A is None:
            self.forward()
        return self._A
    
    def get_E(self):
        if self._edges is None:
            self.forward()
        return self._edges, [self.embedding.weight.clone()]

    def forward(self, x):
        # x shape, B, N, F
        proj = []
        for i, func in enumerate(self.embedding_projection):
            proj.append(func(x[..., i, :]))
        M1 = self.prev_embed
        M2 = torch.stack(proj, dim=0)
             
        A = F.relu(M1 @ M2.T - M2 @ M1.T)
        
        self.prev_embed = M2

        # topk entries 
        _, topk_idx = A.sort(descending=True)

        j = topk_idx[:, self.topk:].flatten()
        A[self._i, j] = 0

        msk = A > 0 # boolean mask

        edge_idx_src = self._edge_indices.T[msk] # source edge indices
        edge_idx_dst = self._edge_indices[msk] # target edge indices
        edge_attr = A[msk].flatten() # edge weights
       
        # shape [2, topk*num_nodes] tensor holding topk edge-index-pairs for each node 
        edge_indices = torch.stack([edge_idx_src, edge_idx_dst], dim=0)

        # save for later
        self._A = A
        self._edges = edge_indices

        return edge_indices, edge_attr, A

class ConvEmbedding(nn.Module):
    r''' Layer for graph representation learning 
    using a linear embedding layer and cosine similarity 
    to produce an index list of edges for a fixed number of 
    neighbors for each node.

    Args:
        num_nodes (int): Number of nodes.
        embed_dim (int): Dimension of embedding.
        topk (int, optional): Number of neighbors per node.
    '''

    def __init__(self, num_nodes, num_node_features, embed_dim, topk=15):
        super().__init__()

        self.topk = topk
        self.embed_dim = embed_dim
        self.in_features = num_node_features

        self.device = get_device()
        
        # INPUT SIZE 25
        self.embedding_conv = nn.Sequential(
            nn.Conv1d(1, 8, 7),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, 5),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Flatten(1, -1),
            nn.Linear(32*11, 2*32),
            nn.ReLU(),
        )
        
        self._A = None
        self._edges = None
        
        ### pre-computed index matrices 
        # square matrix for adjacency matrix indexing
        self._edge_indices = torch.arange(num_nodes).to(self.device).expand(num_nodes, num_nodes) # [[1,2,3,4,5], [1,2,3,4,5], ...] 
        # matrix containing column indices for the right side of a matrix - will be used to remove all but topk entries
        self._i = torch.arange(num_nodes).unsqueeze(1).expand(num_nodes, num_nodes - topk).flatten()

    def get_A(self):
        if self._A is None:
            self.forward()
        return self._A
    
    def get_E(self):
        if self._edges is None:
            self.forward()
        return self._edges, [self.embedding.weight.clone()]

    def forward(self, x):
        # x shape, B, N, F
        
        M1, M2 = self.embedding_conv(x.unsqueeze(-2)).chunk()
             
        A = F.relu(M1 @ M2.T - M2 @ M1.T)
        
        self.prev_embed = M2

        # topk entries 
        _, topk_idx = A.sort(descending=True)

        j = topk_idx[:, self.topk:].flatten()
        A[self._i, j] = 0

        msk = A > 0 # boolean mask

        edge_idx_src = self._edge_indices.T[msk] # source edge indices
        edge_idx_dst = self._edge_indices[msk] # target edge indices
        edge_attr = A[msk].flatten() # edge weights
       
        # shape [2, topk*num_nodes] tensor holding topk edge-index-pairs for each node 
        edge_indices = torch.stack([edge_idx_src, edge_idx_dst], dim=0)

        # save for later
        self._A = A
        self._edges = edge_indices

        return edge_indices, edge_attr, A

        