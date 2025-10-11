from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch.nn import Linear,LayerNorm
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from torch_geometric.data import InMemoryDataset
# from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
import math
import numpy

def sparse_mincut_pool(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    s: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temp: float = 1.0,
    edge_weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Sparse MinCut pooling for a single graph without batch processing.
    
    Args:
        x (torch.Tensor): Node feature tensor [num_nodes, num_features]
        edge_index (torch.Tensor): Edge index tensor [2, num_edges]
        s (torch.Tensor): Assignment tensor [num_nodes, num_clusters]
        mask (torch.Tensor, optional): Node mask [num_nodes]
        temp (float, optional): Temperature for softmax
        edge_weight (torch.Tensor, optional): Edge weights [num_edges]
        
    Returns:
        Tuple containing:
        - pooled_x: Pooled features [num_clusters, num_features]
        - pooled_adj: Pooled adjacency [num_clusters, num_clusters]
        - mincut_loss: MinCut loss scalar
        - ortho_loss: Orthogonality loss scalar
    """
    x = x.squeeze(0) if x.dim() == 3 else x
    edge_index = edge_index.squeeze(0) if edge_index.dim() == 3 else edge_index
    s = s.squeeze(0) if s.dim() == 3 else s
    num_nodes, num_features = x.size()
    num_clusters = s.size(-1)
    
    # Apply softmax to assignment matrix with temperature
    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)
    
    # Apply mask if provided
    if mask is not None:
        mask = mask.to(x.dtype).view(-1, 1)
        x = x * mask
        s = s * mask
    
    # ====================== 1. Compute pooled features ======================
    # pooled_x = s^T * x
    pooled_x = torch.mm(s.t(), x)  # [num_clusters, num_features]
    
    # ====================== 2. Compute pooled adjacency matrix ======================
    row, col = edge_index
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=x.device)
    
    # Compute A * s using sparse operations
    # Step 1: Gather source node features
    s_col = s[col]  # [num_edges, num_clusters]
    
    # Step 2: Multiply by edge weights
    weighted_s = s_col * edge_weight.unsqueeze(-1)  # [num_edges, num_clusters]
    
    # Step 3: Scatter-add to destination nodes
    # Create index for scatter_add: expand row indices for each cluster
    # 创建全局索引：位置 = node_index * num_clusters + cluster_index
    node_indices = row.unsqueeze(1).expand(-1, num_clusters)
    cluster_indices = torch.arange(num_clusters, device=x.device).view(1, -1).expand_as(node_indices)
    global_indices = node_indices * num_clusters + cluster_indices
    # flatten global_indices
    global_indices = global_indices.contiguous().view(-1)
    # Create flattened values tensor
    values = weighted_s.contiguous().view(-1)
    
    # Compute A_s = A * s
    A_s = torch.zeros(num_nodes * num_clusters, device=x.device)
    A_s.scatter_add_(0, global_indices, values)
    A_s = A_s.view(num_nodes, num_clusters)
    
    # Step 4: Compute s^T * A_s
    pooled_adj = torch.mm(s.t(), A_s)  # [ num_clusters, num_clusters]
    
    # ====================== 3. Compute MinCut loss ======================
    # mincut_num = trace(s^T * A * s)
    mincut_num = torch.trace(pooled_adj)  # Scalar
    
    # mincut_den = trace(s^T * D * s)
    # Compute degree vector d = A * 1
    deg = torch.zeros(num_nodes, device=x.device)
    deg = deg.scatter_add_(0, row, edge_weight)  # [num_nodes]
    
    # Compute s^T * D * s = s^T * diag(deg) * s
    sT_D_s = torch.mm(s.t(), deg.unsqueeze(-1) * s)  # [num_clusters, num_clusters]
    mincut_den = torch.trace(sT_D_s)  # Scalar
    
    # Avoid division by zero
    mincut_den = mincut_den + 1e-10
    mincut_loss = -mincut_num / mincut_den
    
    # ====================== 4. Compute Orthogonality loss ======================
    # Compute s^T * s
    sT_s = torch.mm(s.t(), s)  # [num_clusters, num_clusters]
    
    # Normalize by Frobenius norm
    norm_sT_s = torch.norm(sT_s, p='fro')  # Scalar
    normalized_sT_s = sT_s / (norm_sT_s + 1e-10)
    
    # Create identity matrix normalized by sqrt(C)
    identity = torch.eye(num_clusters, device=x.device) / math.sqrt(num_clusters)
    
    # Compute Frobenius norm of difference
    ortho_loss = torch.norm(normalized_sT_s - identity, p='fro')
    
    # ====================== 5. Normalize pooled adjacency matrix ======================
    # Fix and normalize coarsened adjacency matrix
    # Set diagonal to zero
    pooled_adj_sq = pooled_adj # [num_clusters, num_clusters]
    pooled_adj_sq.fill_diagonal_(0)
    
    # Compute degree vector for pooled graph
    d_pooled = pooled_adj_sq.sum(dim=1)  # [num_clusters]
    
    # Symmetric normalization: D^{-1/2} * A * D^{-1/2}
    d_sqrt = torch.sqrt(d_pooled).unsqueeze(-1)  # [num_clusters, 1]
    d_sqrt_inv = 1.0 / (d_sqrt + 1e-15)
    pooled_adj_norm = pooled_adj_sq * d_sqrt_inv * d_sqrt_inv.t()
    
    return (pooled_x, pooled_adj_norm, 
            mincut_loss, ortho_loss)

def drop_node(feats,drop_rate,training:bool):
    n = feats.shape[0]
    drop_rates = torch.FloatTensor(numpy.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        feats = masks.to(feats.device) * feats
        
    else:
        feats = feats * (1. - drop_rate)

    return feats

class SparseNet_Grand(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 Num_TCN,
                 Num_TCN2,
                 S = 1,
                 drop_rate = 0.0,
                 edge_pruning_threshold = None, 
                 hidden_channels=128):
        """
        Parameters
        ----------
        in_channels : int, dimension of node attribute

        Num_TCN : int, number of fine-level TCNs

        Num_TCN2 : int, number of coarse-level TCNs

        S : int, number of running time during training

        drop_rate: float, the rate of node drop out

        edge_pruning_threshold : float, the edge pruning threshold

        hidden_channels: int, the dimension of hidden layer

        """
        super(SparseNet_Grand, self).__init__()

        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.pool1 = self.pool1 = torch.nn.Sequential(
            Linear(hidden_channels, Num_TCN),
            LayerNorm(Num_TCN)
        ) # fine level tcn
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = torch.nn.Sequential(
            Linear(hidden_channels, Num_TCN2),
            LayerNorm(Num_TCN2)
        ) # coarse level tcn
        if edge_pruning_threshold is None:
            self.edge_pruning_threshold = 1/(Num_TCN-1)
        else:
            self.edge_pruning_threshold = edge_pruning_threshold
        self.dropout = drop_rate
        self.S = S

    def forward(self, x, edge_index, mask=None,training = True):
        if training:
            output_list_s1 = []
            output_list_s2 = []
            mc1_loss = 0
            o1_loss = 0
            mc2_loss = 0
            o2_loss = 0
            for s in range(self.S):
                x_drop = drop_node(x,self.dropout,True)
                x1 = F.relu(self.conv1(x_drop, edge_index))
                s1 = self.pool1(x1)  #Here "s" is a non-softmax tensor.
                # Save important clustering results_1.
                output_list_s1.append(torch.log_softmax(s1,dim=-1))
                x_pool1, adj_pool1, mc1, o1 = sparse_mincut_pool(x1, edge_index, s1, mask)
                mc1_loss += mc1
                o1_loss += o1
                # batch_pool = torch.arange(adj_pool1.size(0)).repeat_interleave(adj_pool1.size(1)).to(x.device)
                # x_sparse = x_pool1.view(-1, x_pool1.size(-1))
                # cut off edge
                adj_pool1 = torch.where(adj_pool1 > self.edge_pruning_threshold, 1.0, 0.0)
                # prepare input data for second layers
                ## convert dense adj to sparse format
                # find non-zero elements
                rows, cols = torch.nonzero(adj_pool1, as_tuple=True)
                edge_index_pool = torch.stack([rows, cols], dim=0)
                # second convolution
                x2 = F.relu(self.conv2(x_pool1,edge_index_pool))
                s2 = self.pool2(x2)
                output_list_s2.append(torch.log_softmax(s2,dim=-1))
                x_pool2, adj_pool2, mc2, o2 = sparse_mincut_pool(x2, edge_index_pool, s2)
                mc2_loss += mc2
                o2_loss += o2

            mc1_loss /= self.S
            o1_loss /= self.S
            mc2_loss /= self.S
            o2_loss /= self.S
    
            return  mc1_loss, o1_loss, mc2_loss, o2_loss, output_list_s1, output_list_s2
        else:   # Inference Mode
            x_drop = drop_node(x,self.dropout,training=False)
            x1 = F.relu(self.conv1(x_drop, edge_index))
            s1 = self.pool1(x1)  #Here "s" is a non-softmax tensor.
            # Save important clustering results_1.
            ClusterAssignTensor_1 = s1
            x_pool1, adj_pool1, mc1, o1 = sparse_mincut_pool(x1, edge_index, s1, mask)
            # cut off edge
            adj_pool1 = torch.where(adj_pool1 > self.edge_pruning_threshold, 1.0, 0.0)
            ClusterAdjTensor_1 = adj_pool1
            # prepare input data for second layers
            ## convert dense adj to sparse format
            ### find non-zero elements
            rows, cols = torch.nonzero(adj_pool1, as_tuple=True)
            edge_index_pool = torch.stack([rows, cols], dim=0)
            # second convolution
            x2 = F.relu(self.conv2(x_pool1,edge_index_pool))
            s2 = self.pool2(x2)
            x_pool2, adj_pool2, mc2, o2 = sparse_mincut_pool(x2, edge_index_pool, s2)
            ClusterAssignTensor_2 = s2
            ClusterAdjTensor_2 = adj_pool2
            return mc1, o1, mc2, o2, ClusterAssignTensor_1, ClusterAdjTensor_1, ClusterAssignTensor_2, ClusterAdjTensor_2
    
class SparseNet(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 Num_TCN,
                 Num_TCN2,
                 edge_pruning_threshold = None,
                 hidden_channels=128):

        """
        Parameters
        ----------
        in_channels : int, dimension of node attribute

        Num_TCN : int, number of fine-level TCNs

        Num_TCN2 : int, number of coarse-level TCNs

        edge_pruning_threshold : float, the edge pruning threshold

        hidden_channels: int, the dimension of hidden layer

        """
        super(SparseNet, self).__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.pool1 = self.pool1 = torch.nn.Sequential(
            Linear(hidden_channels, Num_TCN),
            LayerNorm(Num_TCN)
        ) # fine level tcn
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = torch.nn.Sequential(
            Linear(hidden_channels, Num_TCN2),
            LayerNorm(Num_TCN2)
        ) # coarse level tcn
        if edge_pruning_threshold is None:
            self.edge_pruning_threshold = 1/(Num_TCN-1)
        else:
            self.edge_pruning_threshold = edge_pruning_threshold
    
    def forward(self, x, edge_index, mask=None):

        x1 = F.relu(self.conv1(x, edge_index))
        s1 = self.pool1(x1)  #Here "s" is a non-softmax tensor.
        # Save important clustering results_1.
        ClusterAssignTensor_1 = s1
        x_pool1, adj_pool1, mc1, o1 = sparse_mincut_pool(x1, edge_index, s1, mask)
        
        # batch_pool = torch.arange(adj_pool1.size(0)).repeat_interleave(adj_pool1.size(1)).to(x.device)
        # x_sparse = x_pool1.view(-1, x_pool1.size(-1))
        # cut off edge
        adj_pool1 = torch.where(adj_pool1 > self.edge_pruning_threshold, 1.0, 0.0)
        ClusterAdjTensor_1 = adj_pool1
        # prepare input data for second layers
        ## convert dense adj to sparse format
        # find non-zero elements
        rows, cols = torch.nonzero(adj_pool1, as_tuple=True)
        edge_index_pool = torch.stack([rows, cols], dim=0)

        # second convolution
        x2 = F.relu(self.conv2(x_pool1,edge_index_pool))
        s2 = self.pool2(x2)

        x_pool2, adj_pool2, mc2, o2 = sparse_mincut_pool(x2, edge_index_pool, s2)

        ClusterAssignTensor_2 = s2
        ClusterAdjTensor_2 = adj_pool2

        return  mc1, o1, mc2, o2, ClusterAssignTensor_1, ClusterAdjTensor_1, ClusterAssignTensor_2, ClusterAdjTensor_2

