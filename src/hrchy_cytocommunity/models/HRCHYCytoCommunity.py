import torch
# from torch.optim.lr_scheduler import StepLR
from .dataset import SpatialOmicsImageDataset
from .net import SparseNet_Grand,SparseNet
from ..visualization.visualization import vis_scatter_label
from sklearn.metrics import adjusted_mutual_info_score
import os
import numpy
import datetime
import csv
from tqdm import tqdm
import pandas as pd

def consis_loss(logps, temp):
    """
    Parameters
    -----------
    logps: log softmax prediction of different times of running

    temp: float temperature of sharpening 

    """
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim = 2)
    
    avg_p = torch.mean(ps, dim = 2)
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim = 1, keepdim=True))

    return loss

def entropy_loss_grand(logps):
    ps = [torch.exp(p) for p in logps]
    entro_loss_list = [entropy_loss(s) for s in ps]
    entro_loss = torch.mean(torch.stack(entro_loss_list, dim = 0))
    return entro_loss


def entropy(p, dim=-1, eps=1e-10):
    p = p.clamp(min=eps)
    return -(p * torch.log(p)).sum(dim=dim)

def entropy_loss(S):
    S = torch.softmax(S,dim=-1)
    cluster_probs = S.mean(dim = 0)
    cluster_entropy = entropy(cluster_probs)
    max_entropy = torch.log(torch.tensor(S.shape[1], dtype=torch.float))
    return 1- cluster_entropy / max_entropy

class HRCHYCytoCommunityGrand:
    """
    Hierarchical Community Detection Model (full model)

    This class implements a hierarchical tissue structure identification model
    that integrates a differentiable graph pooling mechanism (MinCut-based)
    with spatial feature learning on single-cell spatial omics data.
    It extends HRCHYCytoCommunity by introducing consitency and entropy-based balance regularization.

    Parameters
    ----------
    dataset : object
        Input dataset object, which must contain at least `num_features` attribute
        and provide node features and adjacency information for graph construction.
    num_tcn1 : int
        Number of fine-grained Cellular neighborhood (TCN1 level).
    num_tcn2 : int
        Number of coarse-grained tissue compartments (TCN2 level).
    cell_meta : pandas.DataFrame or dict
        Metadata for each cell, typically including cell type, position, or annotations(optimal).
    lr : float, default=1e-4
        Learning rate for the optimizer.
    alpha : float, default=0.9
        Initial balance coefficient between fine-grained and coarse-grained objectives, not recommend to change.
    num_epoch : int, default=1500
        Number of training epochs.
    lambda1 : float, default=1.0
        Weight of consistency regularization loss term.
    lambda2 : float, default=1.0
        Weight of orthogonality loss term. Not recommend to change.
    lambda_balance : float, default=1.0
        Weight balancing the hierarchy consistency regularization.
    edge_pruning_cutoff : float or None, optional
        Threshold for edge pruning. If None, defaults to 1 / (num_tcn1 - 1).
    temp : float, default=1.0
        Temperature coefficient for soft assignment scaling.
    s : int, default=10
        Scaling factor controlling graph coarsening strength.
    drop_rate : float, default=0.5
        Dropout probability used in GNN layers for regularization.
    device : str or torch.device or None, optional
        Device to run the model on. If None, automatically selects `'cuda'` if available,
        otherwise `'cpu'`.
    num_hidden : int, default=128
        Number of hidden channels in the graph neural network.
    gt_coarse : bool, default=False
        Whether to use ground truth coarse-level annotations (for benchmarking or supervision).
    gt_fine : bool, default=False
        Whether to use ground truth fine-level annotations.

    Attributes
    ----------
    model : SparseNet_Grand
        Underlying graph neural network model instance.
    device : str
        Device string used by PyTorch (`'cuda'` or `'cpu'`).
    edge_pruning_cutoff : float
        Final threshold used for edge pruning.
    lr, alpha, epochs, lambda1, lambda2, lambda_balance, temp, drop_rate, S : float or int
        Model hyperparameters stored after initialization.

    Notes
    -----
    - The model automatically constructs an internal `SparseNet_Grand` instance based
      on dataset feature dimensions and user-specified clustering parameters.
    - This class supports GPU acceleration via CUDA when available.
    - The naming convention follows:
        * TCN1 — fine-grained tissue community nodes
        * TCN2 — coarse-grained tissue community nodes

    Examples
    --------
    >>> model = HRCHYCytoCommunityGrand(
    ...     dataset=my_dataset,
    ...     num_tcn1=10,
    ...     num_tcn2=2,
    ...     cell_meta=meta_df,
    ...     lr=1e-3,
    ...     device='cuda'
    ... )
    >>> print(model.device)
    'cuda'
    >>> print(model.model)
    SparseNet_Grand(...)
    """
    def __init__(self,
                 dataset,
                 num_tcn1,
                 num_tcn2,
                 cell_meta,
                 lr = 1e-4,
                 alpha = 0.9,
                 num_epoch = 1500,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda_balance = 1.0,
                 edge_pruning_cutoff = None,
                 temp = 1.0,
                 s = 10,
                 drop_rate = 0.5,
                 device = None,
                 num_hidden = 128,
                 gt_coarse = False,
                 gt_fine = False,
                 ):
        
        self.dataset = dataset
        self.Num_TCN = num_tcn1
        self.Num_TCN2 = num_tcn2
        self.lr = lr
        self.alpha = alpha
        self.epochs = num_epoch
        self.drop_rate = drop_rate
        self.S = s
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_balance = lambda_balance
        self.temp = temp
        self.cell_meta = cell_meta
        self.gt_coarse = gt_coarse
        self.gt_fine = gt_fine
        if edge_pruning_cutoff is None:
            self.edge_pruning_cutoff = 1/(self.Num_TCN-1)
        else:
            self.edge_pruning_cutoff = edge_pruning_cutoff
        print(f"edge_pruning_cutoff = {self.edge_pruning_cutoff}")
        self.model = SparseNet_Grand(dataset.num_features,
                                    self.Num_TCN,
                                    self.Num_TCN2,
                                    S = self.S,
                                    drop_rate=self.drop_rate,
                                    edge_pruning_threshold = self.edge_pruning_cutoff,
                                    hidden_channels=num_hidden) #Initializing the model.
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            

    def train(self,save_dir,output = False,vis_while_training = False):
        """
        Parameters
        ----------
        output : bool
        whether output the training information to screen or not
        vis_while_training : bool
        whether visualize the hierarchical tissue structure assignment during training or not

        """
        loss_all = 0
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0
        loss_5 = 0
        loss_6 = 0
        loss_7 = 0
        data = self.dataset   # only support single graph training
        alpha_min = 1-self.alpha                # 
        update_every = 100              # update alpha for each N epoch
        decay_num = (alpha_min-self.alpha) * update_every /self.epochs
        RunFolderName = save_dir
        if not os.path.exists(RunFolderName):
            os.makedirs(RunFolderName)  #Creating the Run folder.
        filename_0 = RunFolderName + "/Epoch_TrainLoss.csv"
        headers_0 = ["Epoch", "TrainLoss", "MincutLoss1", "OrthoLoss1", "MincutLoss2", "OrthoLoss2","ConsisLoss1","ConsisLoss2",'EntropyLoss','AMI_fine','AMI_coarse']
        with open(filename_0, "w", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow(headers_0)
        for epoch in tqdm(range(self.epochs)): 
            self.model = self.model.to(self.device)
            data = data.to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            mc1, o1, mc2, o2, output_list_s1, output_list_s2 = self.model(
                data.x, 
                data.edge_index, 
                data.mask if hasattr(data, 'mask') else None,
                training = True
            )
            if (epoch+1) % update_every == 0 :
                #self.alpha *= decay_rate
                self.alpha +=decay_num
                #print(f"Epoch {epoch+1}: alpha updated to {self.alpha:.4f}")
            loss_unsup = self.alpha * (mc1 + self.lambda2 * o1) + (1 - self.alpha) * (mc2 + self.lambda2* o2)
            cl1 = consis_loss(output_list_s1, self.temp) # consistent loss of fine level TCN distribution
            cl2 = consis_loss(output_list_s2, self.temp) # consistent loss of coarse level TCN distribution
            loss_consis = self.alpha * cl1 + (1 - self.alpha) * cl2
            loss_entropy = entropy_loss_grand(output_list_s2)
            # loss = mc_loss1 + o_loss1 + mc_loss2 +o_loss2
            loss = loss_unsup + self.lambda1 * loss_consis + self.lambda_balance * loss_entropy
            
            # loss = mc_loss1 + o_loss1 + mc_loss2 +o_loss2
            loss.backward()
            loss_all += loss.item()
            loss_1 += mc1.item()
            loss_2 += o1.item()
            loss_3 += mc2.item()
            loss_4 += o2.item()
            loss_5 += cl1.item()
            loss_6 += cl2.item()
            loss_7 += loss_entropy.item()
            optimizer.step()

            if (epoch+1)%20 == 0:
                cell_meta = self.cell_meta
                _,fine_cluster_id,_,coarse_cluster_id = self.predict(save=False)
                cell_meta["fine_cluster_id"] = fine_cluster_id.flatten()
                cell_meta["coarse_cluster_id"] = coarse_cluster_id[fine_cluster_id.flatten()].flatten()
                cell_meta["fine_cluster_id"]+=1
                cell_meta["coarse_cluster_id"]+=1
                cell_meta["fine_cluster_id"] = cell_meta["fine_cluster_id"].astype('str')
                cell_meta["coarse_cluster_id"] = cell_meta["coarse_cluster_id"].astype('str')
                if self.gt_fine:
                    ami_fine = adjusted_mutual_info_score(cell_meta['fine_GT'].tolist(),cell_meta['fine_cluster_id'].tolist())
                else:
                    ami_fine = 0
                if self.gt_coarse:
                    ami_coarse = adjusted_mutual_info_score(cell_meta['coarse_GT'].tolist(),cell_meta['coarse_cluster_id'].tolist())
                else:
                    ami_coarse = 0
                with open(filename_0, "a", newline='') as f0:
                    f0_csv = csv.writer(f0)
                    f0_csv.writerow([epoch+1, loss_all/20, loss_1/20, loss_2/20, loss_3/20, loss_4/20,loss_5/20, loss_6/20,loss_7/20,ami_fine,ami_coarse])
                if output:
                    print(f"{epoch+1}\t{loss_all/20}\t{loss_1/20}\t{loss_2/20}\t{loss_3/20}\t{loss_4/20}\t{loss_5/20}\t{loss_6/20}\t{loss_7/20}\t{ami_fine}\t{ami_coarse}")
                if vis_while_training:
                    if (epoch+1)%500 == 0:
                        vis_dir = os.path.join(save_dir,'figures')
                        if not os.path.exists(vis_dir):
                            os.makedirs(vis_dir)
                        # fine level 
                        vis_scatter_label(cell_meta,"fine_cluster_id",dict_color=None,title=f'fine_cluster_id_epoch_{epoch+1}',
                                        output_dir=vis_dir,level='fine')
                        vis_scatter_label(cell_meta,"coarse_cluster_id",dict_color=None,title=f'coarse_cluster_id_epoch_{epoch+1}',
                                        output_dir=vis_dir,level = 'coarse')
                loss_all = 0
                loss_1 = 0
                loss_2 = 0
                loss_3 = 0
                loss_4 = 0
                loss_5 = 0
                loss_6 = 0
                loss_7 = 0
        return 
    
    def predict(self,save = False,save_dir = './results'):
        """
        predict the hierarchical tissue structure assignment

        Parameters
        ----------
        save : bool
        whether save the hierarchical tissue structure assignment to disk or not 
        save_dir : str (path)
        the directory to save the hierarchical tissue structure assignment

        """
        data = self.dataset
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)
            data = data.to(self.device)
            TestModelResult = self.model(
                data.x, 
                data.edge_index, 
                data.mask if hasattr(data, 'mask') else None,
                training = False
            )
            # fine-level clustering
            ClusterAssignMatrix1 = TestModelResult[4]
            ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)
            ClusterAssignMatrix1_hard = torch.argmax(ClusterAssignMatrix1,dim = 1)  #Checked, consistent with function built in "dense_mincut_pool".
            ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy()
            ClusterAssignMatrix1_hard = ClusterAssignMatrix1_hard.unsqueeze(-1).cpu().detach().numpy()
            # fine-level adj
            ClusterAdjMatrix1 = TestModelResult[5]
            ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
            # coarse-level clustering
            ClusterAssignMatrix2 = TestModelResult[6]
            ClusterAssignMatrix2 = torch.softmax(ClusterAssignMatrix2, dim=-1)  # Checked, consistent with function built in "dense_mincut_pool".
            ClusterAssignMatrix2_hard = torch.argmax(ClusterAssignMatrix2,dim = 1)
            ClusterAssignMatrix2 = ClusterAssignMatrix2.cpu().detach().numpy()
            ClusterAssignMatrix2_hard = ClusterAssignMatrix2_hard.unsqueeze(-1).cpu().detach().numpy()
            # coarse-level adj
            ClusterAdjMatrix2 = TestModelResult[7]
            ClusterAdjMatrix2 = ClusterAdjMatrix2.cpu().detach().numpy()
        if save:
            RunFolderName = save_dir
            if not os.path.exists(RunFolderName):
                os.makedirs(RunFolderName)
            #Extract the soft clustering matrix using the trained model.
            filename1 = RunFolderName + "/fine_ClusterAssignMatrix_soft.csv"
            numpy.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')
            filename1 = RunFolderName + "/fine_ClusterAssignMatrix_hard.csv"
            numpy.savetxt(filename1, ClusterAssignMatrix1_hard,fmt = "%d", delimiter=',')
            filename2 = RunFolderName + "/ClusterAdjMatrix1.csv"
            numpy.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

            filename3 = RunFolderName + "/coarse_ClusterAssignMatrix_soft.csv"
            numpy.savetxt(filename3, ClusterAssignMatrix2, delimiter=',')
            filename3 = RunFolderName + "/coarse_ClusterAssignMatrix_hard.csv"
            numpy.savetxt(filename3, ClusterAssignMatrix2_hard,fmt = "%d", delimiter=',')

            filename4 = RunFolderName + "/ClusterAdjMatrix2.csv"
            numpy.savetxt(filename4, ClusterAdjMatrix2, delimiter=',')

            GraphIdxArray = data.graph_idx.view(-1).cpu().numpy()
            filename6 = RunFolderName + "/GraphIdx.csv"
            numpy.savetxt(filename6, GraphIdxArray, delimiter=',', fmt='%i')  #save as integer.
        return ClusterAssignMatrix1,ClusterAssignMatrix1_hard,ClusterAssignMatrix2,ClusterAssignMatrix2_hard



class HRCHYCytoCommunity:
    """
    Hierarchical Community Detection Model (base model)

    This class implements a hierarchical tissue structure identification base model
    that integrates a differentiable graph pooling mechanism (MinCut-based)
    with spatial feature learning on single-cell spatial omics data.

    Parameters
    ----------
    dataset : object
        Input dataset object, which must contain at least `num_features` attribute
        and provide node features and adjacency information for graph construction.
    num_tcn1 : int
        Number of fine-grained Cellular neighborhood (TCN1 level).
    num_tcn2 : int
        Number of coarse-grained tissue compartments (TCN2 level).
    cell_meta : pandas.DataFrame or dict
        Metadata for each cell, typically including cell type, position, or annotations(optimal).
    lr : float, default=1e-4
        Learning rate for the optimizer.
    alpha : float, default=0.9
        Initial balance coefficient between fine-grained and coarse-grained objectives, not recommend to change.
    num_epoch : int, default=1500
        Number of training epochs.
    lambda2 : float, default=1.0
        Weight of orthogonality loss term. Not recommend to change.
    edge_pruning_cutoff : float or None, optional
        Threshold for edge pruning. If None, defaults to 1 / (num_tcn1 - 1).
    device : str or torch.device or None, optional
        Device to run the model on. If None, automatically selects `'cuda'` if available,
        otherwise `'cpu'`.
    num_hidden : int, default=128
        Number of hidden channels in the graph neural network.
    gt_coarse : bool, default=False
        Whether to use ground truth coarse-level annotations (for benchmarking or supervision).
    gt_fine : bool, default=False
        Whether to use ground truth fine-level annotations.

    Attributes
    ----------
    model : SparseNet
        Underlying graph neural network model instance.
    device : str
        Device string used by PyTorch (`'cuda'` or `'cpu'`).
    edge_pruning_cutoff : float
        Final threshold used for edge pruning.
    lr, alpha, epochs, lambda1: float or int
        Model hyperparameters stored after initialization.

    Notes
    -----
    - The model automatically constructs an internal `SparseNet` instance based
      on dataset feature dimensions and user-specified clustering parameters.
    - This class supports GPU acceleration via CUDA when available.
    - The naming convention follows:
        * TCN1 — fine-grained tissue community nodes
        * TCN2 — coarse-grained tissue community nodes

    Examples
    --------
    >>> model = HRCHYCytoCommunityGrand(
    ...     dataset=my_dataset,
    ...     num_tcn1=10,
    ...     num_tcn2=2,
    ...     cell_meta=meta_df,
    ...     lr=1e-3,
    ...     device='cuda'
    ... )
    >>> print(model.device)
    'cuda'
    >>> print(model.model)
    SparseNet(...)
    """
    def __init__(self,
                 dataset,
                 num_tcn1,
                 num_tcn2,
                 cell_meta,
                 lr = 1e-4,
                 alpha = 0.9,
                 num_epoch = 1500,
                 lambda1 = 1.0,
                 lambda2 = 1.0,
                 lambda_balance = 1.0,
                 edge_pruning_cutoff = 0.2,
                 device = None,
                 num_hidden = 128,
                 gt_coarse = False,
                 gt_fine = False,
                 ):
        self.dataset = dataset
        self.Num_TCN = num_tcn1
        self.Num_TCN2 = num_tcn2
        self.lr = lr
        self.alpha = alpha
        self.epochs = num_epoch
        self.device = device
        self.lambda2 = lambda2
        self.lambda_balance = lambda_balance
        self.cell_meta = cell_meta
        self.gt_coarse = gt_coarse
        self.gt_fine = gt_fine
        if edge_pruning_cutoff is None:
            self.edge_pruning_cutoff = 1/(self.Num_TCN-1)
        else:
            self.edge_pruning_cutoff = edge_pruning_cutoff
        self.model = SparseNet(dataset.num_features,
                               self.Num_TCN,
                               self.Num_TCN2,
                               edge_pruning_threshold = self.edge_pruning_cutoff,
                               hidden_channels=num_hidden) #Initializing the model.
        

    def train(self,save_dir,output = False,vis_while_training = False):
        """
        Parameters
        ----------
        output : bool
        whether output the training information to screen or not
        vis_while_training : bool
        whether visualize the hierarchical tissue structure assignment during training or not

        """
        loss_all = 0
        loss_1 = 0
        loss_2 = 0
        loss_3 = 0
        loss_4 = 0
        loss_5 = 0
        data = self.dataset   # only support single graph training
        RunFolderName = save_dir
        alpha_min = 1-self.alpha                # 目标最小值
        update_every = 100              # 每 N 个 epoch 更新一次 alpha
        decay_num = (alpha_min-self.alpha) * 100 /self.epochs
        if not os.path.exists(RunFolderName):
            os.makedirs(RunFolderName)  #Creating the Run folder.
        filename_0 = RunFolderName + "/Epoch_TrainLoss.csv"
        headers_0 = ["Epoch", "TrainLoss", "MincutLoss1", "OrthoLoss1", "MincutLoss2", "OrthoLoss2",'EntropyLoss','AMI_fine','AMI_coarse']
        with open(filename_0, "w", newline='') as f0:
            f0_csv = csv.writer(f0)
            f0_csv.writerow(headers_0)
        for epoch in tqdm(range(self.epochs)): 
            self.model = self.model.to(self.device)
            data = data.to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            optimizer.zero_grad()
            mc1, o1, mc2, o2, s1,_ , s2, _ = self.model(
                data.x, 
                data.edge_index, 
                data.mask if hasattr(data, 'mask') else None
            )
            if (epoch+1) % update_every == 0 :
                self.alpha += decay_num
                # print(f"Epoch {epoch+1}: alpha updated to {self.alpha:.4f}")
            loss_unsup = self.alpha * (mc1 + self.lambda2 * o1) + (1 - self.alpha) * (mc2 + self.lambda2* o2)
            loss_entropy = entropy_loss(s2)
            # loss = mc_loss1 + o_loss1 + mc_loss2 +o_loss2
            loss = loss_unsup #+ self.lambda_balance * loss_entropy
            loss.backward()
            loss_all += loss.item()
            loss_1 += mc1.item()
            loss_2 += o1.item()
            loss_3 += mc2.item()
            loss_4 += o2.item()
            loss_5 += loss_entropy.item()
            optimizer.step()

            if (epoch+1)%20 == 0:
                cell_meta = self.cell_meta
                _,fine_cluster_id,_,coarse_cluster_id = self.predict(save=False)
                cell_meta["fine_cluster_id"] = fine_cluster_id.flatten()
                cell_meta["coarse_cluster_id"] = coarse_cluster_id[fine_cluster_id.flatten()].flatten()
                cell_meta["fine_cluster_id"]+=1
                cell_meta["coarse_cluster_id"]+=1
                cell_meta["fine_cluster_id"] = cell_meta["fine_cluster_id"].astype('str')
                cell_meta["coarse_cluster_id"] = cell_meta["coarse_cluster_id"].astype('str')
                if self.gt_fine:
                    ami_fine = adjusted_mutual_info_score(cell_meta['fine_GT'].tolist(),cell_meta['fine_cluster_id'].tolist())
                else:
                    ami_fine = 0
                if self.gt_coarse:
                    ami_coarse = adjusted_mutual_info_score(cell_meta['coarse_GT'].tolist(),cell_meta['coarse_cluster_id'].tolist())
                else:
                    ami_coarse = 0
                with open(filename_0, "a", newline='') as f0:
                    f0_csv = csv.writer(f0)
                    f0_csv.writerow([epoch+1, loss_all/20, loss_1/20, loss_2/20, loss_3/20, loss_4/20,loss_5/20,ami_fine,ami_coarse])
                if output:
                    print(f"{epoch+1}\t{loss_all/20}\t{loss_1/20}\t{loss_2/20}\t{loss_3/20}\t{loss_4/20}\t{loss_5/20}\t{ami_fine}\t{ami_coarse}")
                if vis_while_training:
                    if (epoch+1)%100 == 0:
                        #print(s2)
                        vis_dir = os.path.join(save_dir,'figures')
                        if not os.path.exists(vis_dir):
                            os.makedirs(vis_dir)
                        # fine level 
                        vis_scatter_label(cell_meta,"fine_cluster_id",dict_color=None,title=f'fine_cluster_id_epoch_{epoch+1}',
                                        output_dir=vis_dir,level='fine')
                        vis_scatter_label(cell_meta,"coarse_cluster_id",dict_color=None,title=f'coarse_cluster_id_epoch_{epoch+1}',
                                        output_dir=vis_dir,level = 'coarse')
                loss_all = 0
                loss_1 = 0
                loss_2 = 0
                loss_3 = 0
                loss_4 = 0
                loss_5 = 0
        return 
    
    def predict(self, save = False,save_dir = './results'):
        """
        predict the hierarchical tissue structure assignment

        Parameters
        ----------
        save : bool
        whether save the hierarchical tissue structure assignment to disk or not 
        save_dir : str (path)
        the directory to save the hierarchical tissue structure assignment

        """
        data = self.dataset
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to(self.device)
            data = data.to(self.device)
            TestModelResult = self.model(
                data.x, 
                data.edge_index, 
                data.mask if hasattr(data, 'mask') else None,
            )
            # fine-level clustering
            ClusterAssignMatrix1 = TestModelResult[4]
            ClusterAssignMatrix1 = torch.softmax(ClusterAssignMatrix1, dim=-1)
            ClusterAssignMatrix1_hard = torch.argmax(ClusterAssignMatrix1,dim = 1)  #Checked, consistent with function built in "dense_mincut_pool".
            ClusterAssignMatrix1 = ClusterAssignMatrix1.cpu().detach().numpy()
            ClusterAssignMatrix1_hard = ClusterAssignMatrix1_hard.unsqueeze(-1).cpu().detach().numpy()
            # fine-level adj
            ClusterAdjMatrix1 = TestModelResult[5]
            ClusterAdjMatrix1 = ClusterAdjMatrix1.cpu().detach().numpy()
            # coarse-level clustering
            ClusterAssignMatrix2 = TestModelResult[6]
            ClusterAssignMatrix2 = torch.softmax(ClusterAssignMatrix2, dim=-1)  # Checked, consistent with function built in "dense_mincut_pool".
            ClusterAssignMatrix2_hard = torch.argmax(ClusterAssignMatrix2,dim = 1)
            ClusterAssignMatrix2 = ClusterAssignMatrix2.cpu().detach().numpy()
            ClusterAssignMatrix2_hard = ClusterAssignMatrix2_hard.unsqueeze(-1).cpu().detach().numpy()
            # coarse-level adj
            ClusterAdjMatrix2 = TestModelResult[7]
            ClusterAdjMatrix2 = ClusterAdjMatrix2.cpu().detach().numpy()
        if save:
            RunFolderName = save_dir
            if not os.path.exists(RunFolderName):
                os.makedirs(RunFolderName)
            #Extract the soft clustering matrix using the trained model.
            filename1 = RunFolderName + "/fine_ClusterAssignMatrix_soft.csv"
            numpy.savetxt(filename1, ClusterAssignMatrix1, delimiter=',')
            filename1 = RunFolderName + "/fine_ClusterAssignMatrix_hard.csv"
            numpy.savetxt(filename1, ClusterAssignMatrix1_hard,fmt = "%d", delimiter=',')
            filename2 = RunFolderName + "/ClusterAdjMatrix1.csv"
            numpy.savetxt(filename2, ClusterAdjMatrix1, delimiter=',')

            filename3 = RunFolderName + "/coarse_ClusterAssignMatrix_soft.csv"
            numpy.savetxt(filename3, ClusterAssignMatrix2, delimiter=',')
            filename3 = RunFolderName + "/coarse_ClusterAssignMatrix_hard.csv"
            numpy.savetxt(filename3, ClusterAssignMatrix2_hard,fmt = "%d", delimiter=',')

            filename4 = RunFolderName + "/ClusterAdjMatrix2.csv"
            numpy.savetxt(filename4, ClusterAdjMatrix2, delimiter=',')

            GraphIdxArray = data.graph_idx.view(-1).cpu().numpy()
            filename6 = RunFolderName + "/GraphIdx.csv"
            numpy.savetxt(filename6, GraphIdxArray, delimiter=',', fmt='%i')  #save as integer.
        return ClusterAssignMatrix1,ClusterAssignMatrix1_hard,ClusterAssignMatrix2,ClusterAssignMatrix2_hard

