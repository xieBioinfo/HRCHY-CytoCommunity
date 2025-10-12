import scanpy as sc
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import kneighbors_graph
import datetime
from typing import Optional
def formulate_HRCHYCytoCommunity_input_from_anndata_singlecell(adata,sample_id,ct_col,categories,
                                                               output_dir,graph_id,coarse_gt_col = None,fine_gt_col = None):
    """
    Formulate HRCHY-CytoCommunity input files from a single-cell spatial transcriptomics AnnData object.

    This function converts an AnnData object containing single-cell spatial transcriptomics data
    into a series of text files required by **HRCHY-CytoCommunity** for downstream hierarchical
    tissue structure identification.  
    The generated files include spatial coordinates, cell type labels, optional ground truth labels,
    node attribute matrices, and graph index information.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell spatial transcriptomics data object containing:
        - Spatial coordinates in ``adata.obsm['spatial']`` (array of shape (n_cells, 2))
        - Cell metadata in ``adata.obs`` (must include at least the column specified by `ct_col`)
    sample_id : str
        Unique sample identifier. Used as prefix for output file names.
    ct_col : str
        Column name in ``adata.obs`` representing cell type labels.
    categories : list of str
        Ordered list of all possible cell type categories. Used to ensure consistent ordering
        of node attribute (one-hot) matrices across samples.
    output_dir : str or Path
        Directory where all generated text files will be saved.
        If the directory does not exist, it will be created automatically.
    graph_id : int
        Graph index (integer ID) assigned to this spatial sample, typically used for multi-sample integration.
    coarse_gt_col : str, optional
        Column name in ``adata.obs`` representing **coarse-grained** ground truth labels.
        If None, no coarse ground truth file is generated.
    fine_gt_col : str, optional
        Column name in ``adata.obs`` representing **fine-grained** ground truth labels.
        If None, no fine ground truth file is generated.

    Outputs
    -------
    The following tab-separated files will be written to `output_dir`:

    - ``<sample_id>_Coordinates.txt`` — spatial coordinates (x, y)
    - ``<sample_id>_CellTypeLabel.txt`` — cell type label per cell
    - ``<sample_id>_NodeAttr.txt`` — node attribute matrix (one-hot encoding of cell types)
    - ``<sample_id>_NodeName.txt`` — names of node attribute dimensions (cell type names)
    - ``<sample_id>_GraphIndex.txt`` — integer index of the graph
    - ``<sample_id>_fineGT.txt`` — fine-level ground truth labels (optional)
    - ``<sample_id>_coarseGT.txt`` — coarse-level ground truth labels (optional)

    Notes
    -----
    - The function assumes that all `adata.obs` columns used (`ct_col`, `coarse_gt_col`, `fine_gt_col`)
      contain categorical or string labels.
    - One-hot encoding of cell types ensures consistent node attribute dimensions across multiple samples.
    - Missing values in the one-hot matrix are automatically replaced with 0.
    - All files are tab-delimited and saved in plain text for downstream compatibility.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("sample1.h5ad")
    >>> categories = ['B_cell', 'T_cell', 'Macrophage', 'Endothelial']
    >>> formulate_HRCHYCytoCommunity_input_from_anndata_singlecell(
    ...     adata=adata,
    ...     sample_id="sample1",
    ...     ct_col="cell_type",
    ...     categories=categories,
    ...     output_dir="data/HRCHY_input/",
    ...     graph_id=0,
    ...     coarse_gt_col="coarse_label",
    ...     fine_gt_col="fine_label"
    ... )
    """
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
    # categories = sorted(adata.obs[ct_col].cat.categories) # ensure the categories would be the same order
    node_attr = pd.get_dummies(adata.obs[ct_col]).T.reindex(index = categories).T.values
    node_attr = np.array(node_attr, dtype=float)  # 
    node_attr = np.nan_to_num(node_attr, nan=0.0).astype(int)
    filename = os.path.join(output_dir,sample_id+'_NodeAttr.txt')
    np.savetxt(filename,node_attr,fmt='%d',delimiter='\t')
    filename = os.path.join(output_dir,sample_id+'_NodeName.txt')
    np.savetxt(filename,np.array(categories),fmt='%s',delimiter='\t')
    # save GraphIndex
    graph_id_mat = np.array([graph_id])
    filename = os.path.join(output_dir,sample_id+'_GraphIndex.txt')
    np.savetxt(filename,graph_id_mat,fmt='%d',delimiter='\t')
    return

def formulate_HRCHYCytoCommunity_input_from_anndata_spot(adata,sample_id,output_dir,graph_id,
                                                         coarse_gt_col = None,fine_gt_col = None):
    """
    Formulate HRCHY-CytoCommunity input files from an AnnData object
    of spatial transcriptomics data with cell type deconvolution results.

    This function converts a *spot-level* spatial transcriptomics dataset into
    a set of text files that serve as standardized input for
    **HRCHY-CytoCommunity**.  
    Unlike the single-cell version, this function assumes that each spot
    contains mixed cell-type proportions (deconvolution results stored in
    ``adata.obsm['deconv_ret']``).

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics dataset.  
        Must contain:
        - ``adata.obsm['spatial']`` : array-like of shape (n_spots, 2), spatial coordinates.  
        - ``adata.obsm['deconv_ret']`` : pandas.DataFrame of shape (n_spots, n_celltypes),
          containing cell type proportions per spot.
    sample_id : str
        Unique sample identifier, used as prefix for all output files.
    output_dir : str or Path
        Directory path where the HRCHY-CytoCommunity input files will be saved.
        Created automatically if it does not exist.
    graph_id : int
        Integer identifier for the current sample (graph index).
        Used for multi-sample integration or batch processing.
    coarse_gt_col : str, optional
        Column name in ``adata.obs`` specifying **coarse-grained** ground truth labels.
        If None, the coarse ground truth file is not generated.
    fine_gt_col : str, optional
        Column name in ``adata.obs`` specifying **fine-grained** ground truth labels.
        If None, the fine ground truth file is not generated.

    Outputs
    -------
    The following tab-separated files are generated in `output_dir`:

    - ``<sample_id>_Coordinates.txt`` — spatial coordinates (x, y)
    - ``<sample_id>_CellTypeLabel.txt`` — list of cell type names (columns from deconvolution result)
    - ``<sample_id>_NodeAttr.txt`` — node attribute matrix (cell type proportions per spot)
    - ``<sample_id>_NodeName.txt`` — names of cell type attributes (same as above)
    - ``<sample_id>_GraphIndex.txt`` — integer index of this sample/graph
    - ``<sample_id>_coarseGT.txt`` — optional coarse ground truth labels
    - ``<sample_id>_fineGT.txt`` — optional fine ground truth labels

    Notes
    -----
    - The deconvolution result ``adata.obsm['deconv_ret']`` must be a DataFrame with
      cell type names as columns and spots as rows.
    - Missing values are not explicitly handled; users should ensure numeric completeness
      before calling this function.
    - The output format is consistent with the single-cell version
      (`formulate_HRCHYCytoCommunity_input_from_anndata_singlecell`), enabling joint
      downstream analysis in HRCHY-CytoCommunity.
    - All files are written in tab-delimited text format.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("Visium_BC_sample.h5ad")
    >>> formulate_HRCHYCytoCommunity_input_from_anndata_spot(
    ...     adata=adata,
    ...     sample_id="VisiumBC_P2",
    ...     output_dir="data/HRCHY_input/",
    ...     graph_id=1,
    ...     coarse_gt_col="compartment",
    ...     fine_gt_col="subregion"
    ... )
    """
    # extract and save coordinate 
    coords = adata.obsm['spatial']
    #coords[:,1] = -coords[:,1]
    filename = os.path.join(output_dir,sample_id+'_Coordinates.txt')
    np.savetxt(filename,coords,fmt='%f',delimiter='\t')
    # extract and save celltype label
    deconv_ret = adata.obsm['deconv_ret']
    celltype_label = deconv_ret.columns
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
    # categories = sorted(adata.obs[ct_col].cat.categories) # ensure the categories would be the same order
    node_attr = deconv_ret.values
    # node_attr = np.array(node_attr, dtype=float)  # 
    # node_attr = np.nan_to_num(node_attr, nan=0.0).astype(int)
    filename = os.path.join(output_dir,sample_id+'_NodeAttr.txt')
    np.savetxt(filename,node_attr,fmt='%f',delimiter='\t')
    filename = os.path.join(output_dir,sample_id+'_NodeName.txt')
    np.savetxt(filename,celltype_label,fmt='%s',delimiter='\t')
    # save GraphIndex
    graph_id_mat = np.array([graph_id])
    filename = os.path.join(output_dir,sample_id+'_GraphIndex.txt')
    np.savetxt(filename,graph_id_mat,fmt='%d',delimiter='\t')
    return

def compute_knn(coords, K, sample_id, save_folder: Optional[str] = None):
    """
    Construct a K-Nearest Neighbor (KNN) graph and optionally save it to file.

    This function builds an undirected KNN graph from spatial coordinates,
    symmetrizes the adjacency, and outputs the edge list as a NumPy array or
    tab-separated text file compatible with **HRCHY-CytoCommunity**.

    Parameters
    ----------
    coords : numpy.ndarray of shape (n_cells, 2)
        Spatial coordinates of all cells or spots, where each row represents
        a point (x, y).
    K : int
        Number of nearest neighbors to connect for each node.
    sample_id : str
        Identifier for the current sample, used as prefix for saved edge list.
    save_folder : str or Path, optional
        Directory to save the resulting edge list file.  
        If ``None``, the graph is constructed but not written to disk.

    Returns
    -------
    edge_index : numpy.ndarray of shape (n_edges, 2)
        Array of integer pairs representing undirected edges in the KNN graph.
        Each row corresponds to one edge ``[source, target]``.

    Outputs
    -------
    If ``save_folder`` is provided, the following file will be generated:

    - ``<sample_id>_EdgeIndex.txt`` — tab-separated list of undirected edges.

    Notes
    -----
    - The KNN graph is constructed using scikit-learn’s
      :func:`sklearn.neighbors.kneighbors_graph` with
      ``mode='connectivity'`` and ``include_self=False``.
    - The resulting adjacency matrix is **symmetrized** (`A = A ∪ Aᵀ`)
      to ensure undirected connectivity.
    - The sparse adjacency is converted into an explicit edge list for
      downstream graph-based modeling.
    - The graph size (number of edges) depends on both `K` and the
      local density of points.

    Examples
    --------
    >>> import numpy as np
    >>> coords = np.random.rand(100, 2) * 100  # 100 spatial points
    >>> edge_index = compute_knn(coords, K=10, sample_id="sample1",
    ...                          save_folder="data/HRCHY_input/")
    >>> edge_index.shape
    (2000, 2)
    """
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(f'Constructing KNN graph for {len(coords)} points...')
    A = kneighbors_graph(coords, K, mode='connectivity', include_self=False, n_jobs=-1)  # CSR

    # Symmetric（A = A ∪ A^T）
    A = A.maximum(A.T).tocsr()
    A.eliminate_zeros()
    A.sort_indices()

    # Take all directed edges (after symmetry, both i->j and j->i will be included)
    src, dst = A.nonzero()
    edge_index = np.vstack((src, dst)).astype(np.int64) 
    edge_index = edge_index.T # or int32
    if save_folder is not None:
        filename = os.path.join(save_folder, f"{sample_id}_EdgeIndex.txt")
        np.savetxt(filename, edge_index, delimiter='\t', fmt='%d')
        print(f"Saved {len(edge_index)} edges to {filename}")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return edge_index

