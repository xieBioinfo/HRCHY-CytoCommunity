import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只为触发3D后端
import os

# dict_color_fine={"1": "#5a3b1c", "2": "#939396", "3": "#2c663b", "4": "#d63189", "5": "#54a9dd",
#                    "6": "#813188", "7": "Orange"}

# dict_color_coarse = {"1": "#C1C1C1", "2": "#FF6A6A","3": "#54a9dd","4": "#5a3b1c"}

def load_base_data(InputFolderName,graph_index,is_single_cell = True,fine_GT = False,coarse_GT = False):
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
    if dict_color is None:
        if level == 'coarse':
            # dict_color = dict_color_coarse
            dict_color = sns.color_palette("Set2")
        elif level == 'fine':
            # dict_color = dict_color_fine
            dict_color = sns.color_palette("hsv")
    fig, ax = plt.subplots(figsize=(5, 4))
    # 使用 scatterplot 替代 lmplot
    sns.scatterplot(x="x_coordinate", 
                    y="y_coordinate", 
                    data=target_graph_map, 
                    hue=label_name, 
                    legend=True, 
                    # palette=dict_color,  # 如果需要取消注释
                    alpha = 1,
                    s=10.0,  # 点大小
                    ax=ax)
    
    # 移除坐标轴刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    # 设置标题（如果有）
    if title is not None:
        ax.set_title(title)
    
    # 移除坐标轴边框
    sns.despine(left=True, bottom=True, ax=ax)
    
    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    
    if label_order is not None:
        # 按指定顺序排序图例
        unique_labels = list(set(labels))
        label_to_handle = {label: handle for label, handle in zip(labels, handles)}
        
        # 创建按指定顺序排列的图例
        ordered_handles = [label_to_handle[label] for label in label_order if label in unique_labels]
        ax.legend(ordered_handles, 
                  label_order, 
                  title=label_name, 
                  bbox_to_anchor=(1.05, 0.5),
                  loc='center left',
                  markerscale=3,
                  )
    else:
        # 使用默认图例
        ax.legend(handles, 
                  labels, 
                  title=label_name,
                  bbox_to_anchor=(1.05, 0.5),
                  loc='center left',
                  markerscale=3,
                  )
    
    # 移除图形周围的空白（可选）
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
    ax = sns.heatmap(mat, cmap="Reds", annot=True)
    ax.set_title("Heatmap")
    if save_path is not None:
        plt.savefig(save_path,dpi = 300)
    if output:
        plt.show()
    return



