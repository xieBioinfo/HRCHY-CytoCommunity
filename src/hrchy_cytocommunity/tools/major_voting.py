import numpy as np

def majority_voting(clusterings, n_classes=None):
    """
    对多次聚类结果进行多数投票整合
    
    参数:
    clusterings: 聚类结果列表，每个元素是一个聚类标签数组 (n_samples,)
    n_classes: 最终聚类数量（可选）
    
    返回:
    整合后的聚类标签 (n_samples,)
    """
    n_samples = len(clusterings[0])
    n_clusterings = len(clusterings)
    
    # 如果没有指定类别数，使用最多类别数
    if n_classes is None:
        n_classes = max(max(np.unique(c)) for c in clusterings)+1
    
    # 步骤2: 创建投票矩阵
    vote_matrix = np.zeros((n_samples, n_classes), dtype=int)
    
    for clustering in clusterings:
        for sample_idx, label in enumerate(clustering):
            if label < n_classes:  # 确保标签在范围内
                vote_matrix[sample_idx, label] += 1
    
    # 步骤3: 进行多数投票
    consensus_labels = np.argmax(vote_matrix, axis=1)
    
    # 处理平票情况
    max_votes = np.max(vote_matrix, axis=1)
    tie_indices = np.where(max_votes == 0)[0]  # 处理0投票的情况
    if len(tie_indices) > 0:
        # 对于没有投票的情况，随机分配一个类别
        consensus_labels[tie_indices] = np.random.randint(0, n_classes, size=len(tie_indices))
    
    return consensus_labels