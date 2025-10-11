import scanpy as sc
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import kneighbors_graph
import datetime
from typing import Optional
def formulate_HRCHYCytoCommunity_input_from_anndata(adata,sample_id,ct_col,output_dir,graph_id,coarse_gt_col = None,fine_gt_col = None):
    # extract and save coordinate 
    coords = adata.obsm['spatial']
    #coords[:,1] = -coords[:,1]
    filename = os.path.join(output_dir,sample_id+'_Coordinates.txt')
    np.savetxt(filename,coords,fmt='%f',delimiter='\t')
    # extract and save celltype label
    celltype_label = adata.obs[ct_col].values
    filename = os.path.join(output_dir,sample_id+'_CellTypeLabel.txt')
    np.savetxt(filename,celltype_label,fmt='%s',delimiter='\t')
    # extract and save Ground Truth label
    if coarse_gt_col is not None:
        gt_label = adata.obs[coarse_gt_col].values
        filename = os.path.join(output_dir,sample_id+'_coarseGT.txt')
        np.savetxt(filename,gt_label,fmt='%s',delimiter='\t')
    if fine_gt_col is not None:
        gt_label = adata.obs[fine_gt_col].values
        filename = os.path.join(output_dir,sample_id+'_fineGT.txt')
        np.savetxt(filename,gt_label,fmt='%s',delimiter='\t')
    # extract and save Node attribute
    categories = sorted(adata.obs[ct_col].cat.categories) # ensure the categories would be the same order
    node_attr = pd.get_dummies(adata.obs[ct_col]).T.reindex(index = categories).T.values
    filename = os.path.join(output_dir,sample_id+'_NodeAttr.txt')
    np.savetxt(filename,node_attr,fmt='%d',delimiter='\t')
    filename = os.path.join(output_dir,sample_id+'_NodeName.txt')
    np.savetxt(filename,np.array(categories),fmt='%s',delimiter='\t')
    # save GraphIndex
    graph_id_mat = np.array([graph_id])
    filename = os.path.join(output_dir,sample_id+'_GraphIndex.txt')
    np.savetxt(filename,graph_id_mat,fmt='%d',delimiter='\t')

def compute_knn(coords, K, sample_id, save_folder: Optional[str] = None):
    """
    construct KNN graph and save it into file
    
    参数:
    coords: (n, 2) ndarray, the coordinates of cells
    K: the number of nearest neighbors
    sample_id: sample id
    
    save_folder: the path of HRCHY-CytoCommunity input data
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f'Constructing KNN graph for {len(coords)} points...')
    A = kneighbors_graph(coords, K, mode='connectivity', include_self=False, n_jobs=-1)  # CSR

    # 对称化并去重（A = A ∪ A^T）
    A = A.maximum(A.T).tocsr()
    A.eliminate_zeros()
    A.sort_indices()

    # 取所有有向边（对称后会同时包含 i->j 和 j->i）
    src, dst = A.nonzero()
    edge_index = np.vstack((src, dst)).astype(np.int64) 
    edge_index = edge_index.T # or int32
    if save_folder is not None:
        filename = os.path.join(save_folder, f"{sample_id}_EdgeIndex.txt")
        np.savetxt(filename, edge_index, delimiter='\t', fmt='%d')
        print(f"Saved {len(edge_index)} edges to {filename}")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return edge_index

