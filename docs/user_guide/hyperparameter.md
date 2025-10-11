# User guide

## Hyperparameter selection

We conducted various sensitivity experiments on both simulated and real spatial transcriptomics data to evaluate the robussness of model under different hyperparameters. The detailed results and interpretations can be found in the HRCHY-CytoCommunity manuscript.

The hyperparameter in HRCHY-CytoCommunity includes:
- mode: base / full, we recommend user use full model of HRCHY-CytoCommunity,with higher consistency and accuracy.
- K: The number of K-nearest neighbor. Empirically, the higher K setting, the corresponding identified tissue structure will be smoother
- max_Run: maximum times of run to automatically identify the optimal number of hierarchical tissue structures. [Default=10]
- num_epoch: The number of training epochs. [Default=10000]
- num_tcn1: The maximum number of fine-grained tissue structures expected to identify.
- num_tcn2: The maximum number of coarse-grained tissue structures expected to identify.
- Alpha: The weight parameter used to balance fine-grained and coarse-grained loss. [Default=0.9]
- lambda1: Coefficient of consistency regularization. recommend to keep default value [Default=1]
- lambda_balance: Coefficient of cluster balance regularization. recommend to keep default value [Default=1]
- drop_rate : rate of drop node,excessively high dropout rates may degraded performance. recommend to keep default value [Default=0.5]
- num_hidden: The dimension of the embedding features. recommend to keep default value [Default=128]
- Learning_Rate: This parameter determines the step size at each iteration while moving toward a minimum of a loss function. [Default=1E-4]
- s: Increasing this value will increase the memory usage and time consumption. 2 or 5 is recommended. [Default=5]

