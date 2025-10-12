# Installation
HRCHY-CytoCommunity  is available for Python 3.10. It does yet not support Apple silicon.

## Hardware requirement 

- Memory: 2GB or more
- Storage: 4GB or more
- CUDA memory: 4GB or more (For CODEX example dataset K=50, #Cell=80000)

## Software requirement

For convenience, we suggest using a separate conda environment for running HRCHY-CytoCommunity

### Step 1. create conda environment

```bash
#create an environment called hrchy_cytocommunity_env
conda create -n hrchy_cytocommunity_env python=3.10

#activate your environment
conda activate hrchy_cytocommunity_env
```

### Step 2. install additional libraries

To use HRCHY-CytoCommunity, you need to install some external libraries. These include:

- [PyTorch](http://pytorch.org/)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/)
- scanpy
- pandas
- scipy
- statsmodels 
- scikit-learn 
- seaborn
- matplotlib
- tqdm 

#### PyTorch and PyG

We recommend to install the PyTorch libraries with GPU support. If you have
CUDA, this can be done as:

```bash
pip install torch==${TORCH}+${CUDA} --index-url https://download.pytorch.org/whl/${CUDA}
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.4.0 and CUDA 11.8, type:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
```
Or using conda to install:
```bash
conda install pytorch==2.4.0  cudatoolkit=11.8
pip install torch_geometric
```

#### Other dependencies

then we install other library for anlysis

```bash
pip install scanpy pandas scipy statsmodels scikit-learn seaborn tqdm matplotlib 
```



#### Alternative

Alternatively, we have provided a conda environment file with all required external libraries, which you can use as:

```bash
conda env create -f environment.yaml
```



## Installation via PyPi
Subsequently, install HRCHY-CytoCommunity via pip:

```bash
pip install hrchy-cytocommunity
```