import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只为触发3D后端
import os


def load_base_data(InputFolderName,graph_index,is_single_cell = True,fine_GT = False,coarse_GT = False):
    """
    Load spatial coordinate and optional annotation data for a specific region graph.

    This utility function reads spatial coordinate files and associated metadata
    (such as cell type labels and ground-truth cluster labels) from a given input directory.
    It constructs a pandas DataFrame containing per-cell (or per-spot) spatial locations,
    and optionally includes cell-type and hierarchical ground truth annotations.

    Parameters
    ----------
    InputFolderName : str or Path
        Path to the input folder that contains all region files.
        The folder must include:
            - ``ImageNameList.txt`` — list of region names (one per line, tab-separated)
            - ``<region>_Coordinates.txt`` — x/y coordinates of each cell or spot
            - ``<region>_CellTypeLabel.txt`` (optional, required if `is_single_cell=True`)
            - ``<region>_fineGT.txt`` (optional, required if `fine_GT=True`)
            - ``<region>_coarseGT.txt`` (optional, required if `coarse_GT=True`)
    graph_index : int
        The index (0-based) of the region name to be loaded, corresponding to the
        row in ``ImageNameList.txt``.
    is_single_cell : bool, default=True
        Whether the dataset represents single-cell resolution.
        If True, the function attempts to load cell type labels from
        ``<region>_CellTypeLabel.txt`` and adds them to the output DataFrame.
    fine_GT : bool, default=False
        Whether to load fine-grained ground truth labels (from ``_fineGT.txt``).
    coarse_GT : bool, default=False
        Whether to load coarse-grained ground truth labels (from ``_coarseGT.txt``).

    Returns
    -------
    target_graph_map : pandas.DataFrame
        A DataFrame containing spatial coordinates and optional annotation columns.
        Columns include:
            - ``x_coordinate`` : float — x position of each cell/spot
            - ``y_coordinate`` : float — y position of each cell/spot
            - ``CellType`` : str — (if `is_single_cell=True`) cell-type label
            - ``fine_GT`` : int — (if `fine_GT=True`) fine-level ground truth cluster
            - ``coarse_GT`` : int — (if `coarse_GT=True`) coarse-level ground truth cluster

    Notes
    -----
    - The function assumes all input files are tab-separated text files.
    - Coordinate files and label files must have the same number of rows.
    - This function does **not** modify coordinate orientation, but a commented line
      shows how to flip the y-axis if needed for consistency with specific references.

    Examples
    --------
    >>> df = load_base_data(
    ...     InputFolderName="data/Mouse_Spleen/",
    ...     graph_index=0,
    ...     is_single_cell=True,
    ...     fine_GT=True,
    ...     coarse_GT=True
    ... )
    >>> df.head()
       x_coordinate  y_coordinate    CellType  fine_GT  coarse_GT
    0         123.4          98.7  B_cell_A1         2          1
    1         128.9         101.3  B_cell_A2         2          1
    """
    Region_filename = os.path.join(InputFolderName, "ImageNameList.txt")
    region_name_list = pd.read_csv(
            Region_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["Image"],  # set our own names for the columns
        )

    # Import target graph x/y coordinates.
    region_name = region_name_list.Image[graph_index]
    GraphCoord_filename = os.path.join(InputFolderName, region_name + "_Coordinates.txt")
    x_y_coordinates = pd.read_csv(
            GraphCoord_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["x_coordinate", "y_coordinate"],  # set our own names for the columns
        )
    target_graph_map = x_y_coordinates
    #target_graph_map["y_coordinate"] = 0 - target_graph_map["y_coordinate"]  # for consistent with original paper. Don't do this is also ok.
    if is_single_cell:
        # Import cell type label.
        CellType_filename = os.path.join(InputFolderName, region_name + "_CellTypeLabel.txt")
        cell_type_label = pd.read_csv(
                CellType_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["cell_type"],  # set our own names for the columns
            )
        # Add cell type labels to target graph x/y coordinates.
        target_graph_map["CellType"] = cell_type_label.cell_type
    if fine_GT:
        fine_GT_filename = os.path.join(InputFolderName, region_name + "_fineGT.txt")
        fine_GT_label = pd.read_csv(
                fine_GT_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["fine_GT"],  # set our own names for the columns
            )
        # Add cell type labels to target graph x/y coordinates.
        target_graph_map["fine_GT"] = fine_GT_label.fine_GT
    if coarse_GT:
        coarse_GT_filename = os.path.join(InputFolderName, region_name + "_coarseGT.txt")
        coarse_GT_label = pd.read_csv(
                coarse_GT_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["coarse_GT"],  # set our own names for the columns
            )
        # Add cell type labels to target graph x/y coordinates.
        target_graph_map["coarse_GT"] = coarse_GT_label.coarse_GT
    return target_graph_map



def load_data(InputFolderName,ret_output_dir0,GTFolderName,slice_name):
    GraphIndex_filename = os.path.join(ret_output_dir0,slice_name,"GraphIdx.csv")
    graph_index = np.loadtxt(GraphIndex_filename, dtype='int', delimiter="\t")
    # Import region name list.
    Region_filename = InputFolderName + "ImageNameList.txt"
    region_name_list = pd.read_csv(
            Region_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["Image"],  # set our own names for the columns
        )

    # Import target graph x/y coordinates.
    region_name = region_name_list.Image[graph_index]
    GraphCoord_filename = InputFolderName + region_name + "_Coordinates.txt"
    x_y_coordinates = pd.read_csv(
            GraphCoord_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["x_coordinate", "y_coordinate"],  # set our own names for the columns
        )
    target_graph_map = x_y_coordinates
    #target_graph_map["y_coordinate"] = 0 - target_graph_map["y_coordinate"]  # for consistent with original paper. Don't do this is also ok.

    # Import cell type label.
    CellType_filename = InputFolderName + region_name + "_CellTypeLabel.txt"
    cell_type_label = pd.read_csv(
            CellType_filename,
            sep="\t",  # tab-separated
            header=None,  # no heading row
            names=["cell_type"],  # set our own names for the columns
        )
    # Add cell type labels to target graph x/y coordinates.
    target_graph_map["CellType"] = cell_type_label.cell_type

    # Gold standard
    target_graph_map["GT_label"] = pd.read_csv(os.path.join(GTFolderName,slice_name+'_GT.csv'), usecols=["GT_Label"])

    #!!! Add consensus cluster labels to target graph x/y coordinates.
    target_graph_map["fine_cluster_id"] = np.loadtxt(
        os.path.join(ret_output_dir0,slice_name,"fine_ClusterAssignMatrix_hard.csv"), dtype='int', delimiter=",")
    label2 = np.loadtxt(
        os.path.join(ret_output_dir0,slice_name,"coarse_ClusterAssignMatrix_hard.csv"), dtype='int', delimiter=",")

    target_graph_map["coarse_cluster_id"] = label2[target_graph_map["fine_cluster_id"]]
    target_graph_map["fine_cluster_id"]+=1
    target_graph_map["coarse_cluster_id"]+=1
    # Converting integer list to string list for making color scheme discrete.
    target_graph_map["fine_cluster_id"] = target_graph_map["fine_cluster_id"].astype(str)
    target_graph_map["coarse_cluster_id"] = target_graph_map["coarse_cluster_id"].astype(str)
    target_graph_map["GT_label"] = target_graph_map["GT_label"].astype(str)
    return target_graph_map

def vis_scatter_label(target_graph_map:pd.DataFrame,label_name,dict_color,
                      title = None,output_dir = None,label_order = None,level = 'fine',output_screen = False):
    """
    Visualize spatial scatter plots colored by cluster or cell-type labels.

    This function generates a 2D scatter plot of spatial coordinates from
    single-cell or spot-level spatial omics data, where points are colored
    by the specified label (e.g., fine-grained cluster, coarse-grained cluster,
    or cell type). The function supports custom color palettes, legend ordering,
    and output saving.

    Parameters
    ----------
    target_graph_map : pandas.DataFrame
        DataFrame containing at least the following columns:
        - ``x_coordinate`` : float — x-axis coordinate of each cell or spot  
        - ``y_coordinate`` : float — y-axis coordinate of each cell or spot  
        - plus one or more label columns (e.g., ``fine_GT``, ``coarse_GT``, ``CellType``)
        that will be used for coloring.
    label_name : str
        Column name in `target_graph_map` to be visualized as color categories.
    dict_color : dict or list or None
        Color mapping for each label.
        If None, automatically selects a default palette:
        - ``Set2`` for coarse level
        - ``hsv`` for fine level
    title : str, optional
        Title of the figure. If None, no title is shown.
    output_dir : str or Path, optional
        Directory to save the output PNG figure.
        If None, the figure is not saved.
    label_order : list of str, optional
        Specific order of labels for the legend.
        If None, the legend follows the default seaborn order.
    level : {'fine', 'coarse'}, default='fine'
        Determines which default color palette to use when `dict_color` is None.
    output_screen : bool, default=False
        Whether to display the figure on screen (`plt.show()`).

    Returns
    -------
    None
        The function creates and optionally saves a scatter plot but does not
        return any object.

    Notes
    -----
    - The function automatically removes axis ticks and borders for clean visualization.
    - The y-axis is inverted (`ax.invert_yaxis()`) to match typical spatial transcriptomics
      coordinate orientation.
    - Legends are dynamically generated and can be reordered by providing `label_order`.
    - When both `output_dir` and `title` are given, the saved file name will be
      ``<output_dir>/<title>.png``.

    Examples
    --------
    >>> vis_scatter_label(
    ...     target_graph_map=df,
    ...     label_name='fine_GT',
    ...     dict_color=None,
    ...     title='Fine-level Clusters',
    ...     output_dir='results/figures',
    ...     level='fine',
    ...     output_screen=True
    ... )
    """
    if dict_color is None:
        if level == 'coarse':
            # dict_color = dict_color_coarse
            dict_color = sns.color_palette("Set2")
        elif level == 'fine':
            # dict_color = dict_color_fine
            dict_color = sns.color_palette("hsv")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.scatterplot(x="x_coordinate", 
                    y="y_coordinate", 
                    data=target_graph_map, 
                    hue=label_name, 
                    legend=True, 
                    # palette=dict_color,  
                    alpha = 1,
                    s=10.0,  # dot size
                    ax=ax)
    
    # remove tick and label in axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    # set title, if given
    if title is not None:
        ax.set_title(title)

    # remove the axis border
    sns.despine(left=True, bottom=True, ax=ax)
    
    # add legend
    handles, labels = ax.get_legend_handles_labels()
    
    if label_order is not None:
        # sort legend according to give order
        unique_labels = list(set(labels))
        label_to_handle = {label: handle for label, handle in zip(labels, handles)}
        
        # create legend with give order
        ordered_handles = [label_to_handle[label] for label in label_order if label in unique_labels]
        ax.legend(ordered_handles, 
                  label_order, 
                  title=label_name, 
                  bbox_to_anchor=(1.05, 0.5),
                  loc='center left',
                  markerscale=3,
                  )
    else:
        # 
        ax.legend(handles, 
                  labels, 
                  title=label_name,
                  bbox_to_anchor=(1.05, 0.5),
                  loc='center left',
                  markerscale=3,
                  )
    
    plt.tight_layout(pad=0)
    ax.invert_yaxis()
    
    # Save the figure.
    if output_screen:
        plt.show()
    if output_dir is not None:
        os.makedirs(output_dir,exist_ok=True)
        if title is None:
            TCN_fig_filename1 = os.path.join(output_dir,f"{label_name}_vis.png")
        else:
            TCN_fig_filename1 = os.path.join(output_dir,f"{title}.png")
        plt.savefig(TCN_fig_filename1,bbox_inches='tight',dpi = 300)
    plt.close()
    return 


def vis_heatmap(mat:pd.DataFrame,output = True,save_path = None):
    """
    Visualize a heatmap for a given matrix or DataFrame.

    This function creates a heatmap visualization using Seaborn,
    displaying matrix values with optional annotations and saving
    the figure to disk if a path is provided.

    Parameters
    ----------
    mat : pandas.DataFrame
        Input data matrix to visualize. Each element will be represented
        as a colored cell in the heatmap. The index and column names (if any)
        will be used as axis labels.
    output : bool, default=True
        Whether to display the generated heatmap using ``plt.show()``.
        Set to ``False`` when generating figures in non-interactive environments.
    save_path : str or Path, optional
        File path to save the generated heatmap as a PNG image.
        If ``None``, the figure will not be saved.

    Returns
    -------
    None
        The function produces a visual output (heatmap) but does not return
        any data.

    Notes
    -----
    - The heatmap uses the ``Reds`` color map by default.
    - Each cell value in the input matrix is displayed directly (`annot=True`).
    - When both ``output=False`` and ``save_path=None``, the plot will be generated
      but not displayed or saved.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame(np.random.rand(4, 4),
    ...                     index=['A', 'B', 'C', 'D'],
    ...                     columns=['X1', 'X2', 'X3', 'X4'])
    >>> vis_heatmap(data, output=True, save_path="results/heatmap_example.png")
    """
    ax = sns.heatmap(mat, cmap="Reds", annot=True)
    ax.set_title("Heatmap")
    if save_path is not None:
        plt.savefig(save_path,dpi = 300)
    if output:
        plt.show()
    return



