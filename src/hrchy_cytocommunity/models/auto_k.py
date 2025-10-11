from .HRCHYCytoCommunity import HRCHYCytoCommunity,HRCHYCytoCommunityGrand
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from .dataset import SpatialOmicsImageDataset
from ..visualization.visualization import load_base_data, vis_heatmap
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
def _dd_list():
    return defaultdict(list)

def _dd_float():
    return defaultdict(float)
class HRCHYClusterAutoK:
    """
    识别聚类数量的最佳候选值
 
    Parameters
    ----------
    
    """
    
    def __init__(self,dataset,cell_meta,coarse_range:tuple,fine_range:tuple, model_params: dict = None,max_runs=10, 
                similarity_function=None):
        # 处理n_clusters参数: 元组转范围列表
        self.model_params = model_params
        self.dataset = dataset
        self.cell_meta = cell_meta
        self.fine_k_range = list(range(*(max(2, fine_range[0]),fine_range[1]+1)))
        self.coarse_k_range = list(range(*(max(2, coarse_range[0]),coarse_range[1]+1)))
        self.max_runs = max_runs
        self.similarity_function = similarity_function or fowlkes_mallows_score
        self.coarse_labels   = defaultdict(_dd_list)   # 两层：outer -> inner(list)
        self.fine_labels     = defaultdict(_dd_list)
        self.coarse_stability= defaultdict(_dd_list)
        self.fine_stability  = defaultdict(_dd_list)
        self.stability       = defaultdict(_dd_float)  # 两层：outer -> inner(float)
    def fit(self,save_dir):
        """
        为每个聚类数量多次聚类数据并计算平均稳定性
        
        """
        """搜索最佳聚类数量"""
        # 确保保存目录存在
        if not os.path.exists(os.path.join(save_dir,'autok_output')):
            os.makedirs(os.path.join(save_dir,'autok_output'))
        #
        if os.path.exists(f"{save_dir}/autok_output/coarse_labels.pickle"):
            with open(f"{save_dir}/autok_output/coarse_labels.pickle",'rb') as f:
                self.coarse_labels = pickle.load(f)
            with open(f"{save_dir}/autok_output/fine_labels.pickle",'rb') as f:
                self.fine_labels = pickle.load(f)
        
        for coarse_k in tqdm(self.coarse_k_range,desc="粗聚类数量"):
            # 更新模型参数
            for fine_k in tqdm(self.fine_k_range, desc="精细聚类数量"):
                if fine_k < coarse_k:
                    continue
                if self.coarse_labels.__contains__(coarse_k):
                    if self.coarse_labels[coarse_k].__contains__(fine_k):
                        print(f'loading {coarse_k} and {fine_k} clustering results')
                        continue
                for run in range(self.max_runs):
                    print(f"运行轮次 {run+1}/{self.max_runs}")
                    new_fine_labels = {}
                    # 训练模型
                    if self.model_params is None:
                        model = HRCHYCytoCommunity(self.dataset)
                    else:
                        if self.model_params['mode'] == 'base':
                            model = HRCHYCytoCommunity(
                                dataset=self.dataset,
                                num_tcn1=fine_k,
                                num_tcn2=coarse_k,
                                cell_meta= self.cell_meta,
                                lr = self.model_params['lr'],
                                alpha=self.model_params['alpha'],
                                num_epoch=self.model_params['num_epoch'],
                                lambda1 = self.model_params['lambda1'],
                                lambda2 = self.model_params['lambda2'],
                                lambda_balance = self.model_params['lambda_balance'],
                                edge_pruning_cutoff = self.model_params['edge_pruning_cutoff'],
                                device = self.model_params['device'],
                                num_hidden = self.model_params['num_hidden'],
                                gt_coarse = self.model_params['gt_coarse'],
                                gt_fine = self.model_params['gt_fine'],
                            )
                        else:
                            model = HRCHYCytoCommunityGrand(
                                dataset=self.dataset,
                                num_tcn1=fine_k,
                                num_tcn2=coarse_k,
                                cell_meta= self.cell_meta,
                                lr = self.model_params['lr'],
                                alpha=self.model_params['alpha'],
                                num_epoch=self.model_params['num_epoch'],
                                lambda1 = self.model_params['lambda1'],
                                lambda2 = self.model_params['lambda2'],
                                lambda_balance = self.model_params['lambda_balance'],
                                edge_pruning_cutoff = self.model_params['edge_pruning_cutoff'],
                                device = self.model_params['device'],
                                num_hidden = self.model_params['num_hidden'],
                                temp =  self.model_params['temp'],
                                s =  self.model_params['s'],
                                drop_rate =  self.model_params['drop_rate'],
                                gt_coarse = self.model_params['gt_coarse'],
                                gt_fine = self.model_params['gt_fine'],
                            )
                    # if os.path.exists(os.path.join(save_dir, f"fine_{fine_k}_coarse_{coarse_k}_run_{run}")):
                    #     continue
                    model.train(os.path.join(save_dir, f"fine_{fine_k}_coarse_{coarse_k}_run_{run}"), 
                                output=False,vis_while_training=True)
                    model.predict(save = True,save_dir=os.path.join(save_dir, f"fine_{fine_k}_coarse_{coarse_k}_run_{run}"))
                    # 获取预测结果
                    _, fine_label, _, coarse_label = model.predict(save=False)
                    # 保存当前轮次的粗粒度结果
                    fine_label_percell = fine_label.flatten()
                    coarse_label_percell = coarse_label[fine_label_percell].flatten()
                    self.coarse_labels[coarse_k][fine_k].append(coarse_label_percell)
                    # 保存当前轮次得细粒度结果
                    self.fine_labels[coarse_k][fine_k].append(fine_label_percell)
                with open(f"{save_dir}/autok_output/coarse_labels.pickle","wb+") as f:
                    pickle.dump(self.coarse_labels,f)
                with open(f"{save_dir}/autok_output/fine_labels.pickle","wb+") as f:
                    pickle.dump(self.fine_labels,f)

        # 计算每种cluster得设置下得结果的相似性
        for coarse_k in tqdm(self.coarse_k_range,desc="粗聚类数量"):
            # 更新模型参数
            for fine_k in tqdm(self.fine_k_range, desc="精细聚类数量"):
                for run_i in range(self.max_runs-1):
                    for run_j in range(run_i+1,self.max_runs):
                        coarse_labels_i = self.coarse_labels[coarse_k][fine_k][run_i]
                        coarse_labels_j = self.coarse_labels[coarse_k][fine_k][run_j]
                        
                        if np.min(coarse_labels_i)==np.max(coarse_labels_i) or np.min(coarse_labels_j)==np.max(coarse_labels_j):
                            self.coarse_stability[coarse_k][fine_k].append(0)
                        else:
                            self.coarse_stability[coarse_k][fine_k].append(self.similarity_function(
                                coarse_labels_i, 
                                coarse_labels_j  # 使用上一轮的结果
                            ))
                        fine_labels_i = self.fine_labels[coarse_k][fine_k][run_i]
                        fine_labels_j = self.fine_labels[coarse_k][fine_k][run_j]
                        if np.min(fine_labels_i)==np.max(fine_labels_i) or np.min(fine_labels_j)==np.max(fine_labels_j):
                            self.fine_stability[coarse_k][fine_k].append(0)
                        else:
                            self.fine_stability[coarse_k][fine_k].append(self.similarity_function(
                                fine_labels_i, 
                                fine_labels_j  # 使用上一轮的结果
                            ))
                self.stability[coarse_k][fine_k] = np.mean([np.median(self.coarse_stability[coarse_k][fine_k]),
                                                            np.median(self.fine_stability[coarse_k][fine_k])])
    def save(self,save_dir):
        os.makedirs(save_dir,exist_ok=True)
        with open(f"{save_dir}/stability.pickle","wb+") as f:
            pickle.dump(self.stability,f)
        with open(f"{save_dir}/coarse_stability.pickle","wb+") as f:
            pickle.dump(self.coarse_stability,f)
        with open(f"{save_dir}/fine_stability.pickle","wb+") as f:
            pickle.dump(self.fine_stability,f)
        with open(f"{save_dir}/coarse_labels.pickle","wb+") as f:
            pickle.dump(self.coarse_labels,f)
        with open(f"{save_dir}/fine_labels.pickle","wb+") as f:
            pickle.dump(self.fine_labels,f)
        return
    
    def load(self,save_dir):
        with open(f"{save_dir}/stability.pickle",'rb') as f:
            self.stability = pickle.load(f)
        with open(f"{save_dir}/coarse_stability.pickle",'rb') as f:
            self.coarse_stability = pickle.load(f)
        with open(f"{save_dir}/fine_stability.pickle",'rb') as f:
            self.fine_stability = pickle.load(f)
        with open(f"{save_dir}/coarse_labels.pickle",'rb') as f:
            self.coarse_labels = pickle.load(f)
        with open(f"{save_dir}/fine_labels.pickle",'rb') as f:
            self.fine_labels = pickle.load(f)
        return 

    def convert_stability_2_mat(self):
        num_coarse = self.coarse_k_range[-1]-self.coarse_k_range[0]+1
        num_fine = self.fine_k_range[-1]-self.fine_k_range[0]+1
        stability_mat =  pd.DataFrame(np.zeros((num_coarse,num_fine)),index = [f'coarse_{k}' for k in self.coarse_k_range],columns=[f'fine_{k}' for k in self.fine_k_range])
        for i,coarse_k in enumerate(self.coarse_k_range):
            for j,fine_k in enumerate(self.fine_k_range):
                stability_mat.iloc[i,j] = self.stability[coarse_k][fine_k]
        return stability_mat
    
    def find_best_model(self, save_dir,num_coarse,num_fine):
        best_loss = np.inf
        for run_i in range(self.max_runs):
            input_dir = f"{save_dir}/fine_{num_fine}_coarse_{num_coarse}_run_{run_i}"
            train_loss = pd.read_csv(os.path.join(input_dir,'Epoch_TrainLoss.csv'))
            if train_loss.iloc[-1,1] < best_loss:
                best_loss = train_loss.iloc[-1,1]
                best_fine = self.fine_labels[num_coarse][num_fine][run_i]
                best_coarse = self.coarse_labels[num_coarse][num_fine][run_i]
        return best_fine,best_coarse

