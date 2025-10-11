# HRCHY-CytoCommunity

## Overview

<div align=center><img src="https://github.com/xieBioinfo/HRCHY-CytoCommunity/blob/main/Figures/Schematic.jpg" width="650" height="900" alt="pipline"/></div>  


Diverse cell types within a tissue assemble into multicellular structures to shape the functions of the tissue. These structural modules typically comprise specialized consist of subunits, each performing unique roles. Uncovering these hierarchical multicellular structures holds significant importance for gaining deep insights into the assembly principles from individual cells to the entire tissue. However, the methods available for identifying hierarchical tissue structures are quite limited and have several limitations as below.

(1) May not be suitable for single-cell-resolution spatial omics data with limited number of gene or protein expression features available. Instead, cell phenotypes typically serve as better initial cell features for learning tissue structures using such data type.

(2) The identified hierarchical tissue structures may not cover all cells within the dataset.

(3) There may not be a clearly nested relationship between the identified different levels of tissue structures.

(4) Cannot correctly identify tissue structures with spatial discontinuous distribution.

Building upon our recently established tissue structure identification framework [CytoCommunity](https://github.com/huBioinfo/CytoCommunity), we developed **HRCHY-CytoCommunity**, which utilized a graph neural network (GNN) model to identify hierarchical tissue structures on single-cell spatial maps. HRCHY-CytoCommunity models the identification of hierarchical tissue structures as a MinCut-based hierarchical community detection problem, offering several advantages:

(1) HRCHY-CytoCommunity identifies hierarchical tissue structures from a cellular-based perspective, making it suitable for single-cell-resolution spatial omics data, while ensuring that the hierarchical structures cover all cells within the data.

(2) By leveraging differentiable graph pooling and graph pruning, HRCHY-CytoCommunity is capable of simultaneously identifying tissue structures of various hierarchical levels at multiple resolutions and exhibiting clearly nested relationship between them.

(3) HRCHY-CytoCommunity possesses the ability to discover structures with spatial discontinuous distribution.

(4) HRCHY-CytoCommunity employs a consistency training strategy to enhance the stability of the model, while maintaining the unambiguously nested relationship between the hierarchical tissue structures.

(5) HRCHY-CytoCommunity utilizes an additional cell-type enrichment-based clustering module to generate a unified set of nested multicellular structures across all tissue samples, thereby addressing the issue of cross-sample comparative analysis.



## Getting started
Please refer to the documentation. In particular, the

- [Installation](https://hrchy-cytocommunity.readthedocs.io/en/latest/Installation.html)
- [API documentation](https://hrchy-cytocommunity.readthedocs.io/en/latest/index.html)
- [Tutorials](https://hrchy-cytocommunity.readthedocs.io/en/latest/index.html)
- [User guide](https://hrchy-cytocommunity.readthedocs.io/en/latest/user_guide/hyperparameter.html)


## Maintainers

Runzhi Xie(rzxie@stu.xidian.edu.cn)

Lin Gao (lgao@mail.xidian.edu.cn)

Yuxuan Hu (huyuxuan@xidian.edu.cn)


## Citation