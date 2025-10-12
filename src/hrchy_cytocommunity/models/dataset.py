import torch
import numpy
import pandas as pd
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
import os


## Define "Dataset" class based on ordinary Python list.
class SpatialOmicsImageDataset(InMemoryDataset):         
    """
    Spatial omics dataset loader for HRCHY-CytoCommunity.

    This class inherits from :class:`torch_geometric.data.InMemoryDataset`
    and is designed to read preprocessed spatial omics graph data files
    (coordinates, edges, node attributes, and graph indices) from a specified
    directory and construct PyTorch Geometric :class:`torch_geometric.data.Data`
    objects for downstream graph neural network training.

    Each sample (region/tissue section) corresponds to one graph, and all
    graphs are collated into a single dataset stored in
    ``processed/SpatialOmicsImageDataset.pt``.

    Parameters
    ----------
    root : str or Path
        Root directory containing the following subfolders:
        - ``raw/`` — containing raw graph files (text format).
        - ``processed/`` — where the processed dataset will be saved.
    transform : callable, optional
        Data transformation function applied before returning a graph sample.
        See :class:`torch_geometric.transforms`.
    pre_transform : callable, optional
        Data preprocessing transformation function applied before saving
        processed data.

    Attributes
    ----------
    data : torch_geometric.data.Data
        Tensor representation of the concatenated graph dataset.
    slices : dict
        Indexing dictionary used by PyTorch Geometric to retrieve individual
        graphs efficiently.
    processed_paths : list[str]
        List of output file paths (by default ``['SpatialOmicsImageDataset.pt']``).

    Methods
    -------
    raw_file_names()
        Returns the list of expected raw input files (empty list in this case).
    processed_file_names()
        Returns the list of expected processed dataset files.
    download()
        Placeholder for downloading data (not implemented).
    process()
        Constructs :class:`torch_geometric.data.Data` objects from input text
        files under ``raw_dir``. The following files are required for each
        region name listed in ``ImageNameList.txt``:
            - ``<region>_EdgeIndex.txt`` — edge list (tab-delimited, int64)
            - ``<region>_NodeAttr.txt`` — node attributes (tab-delimited, float32)
            - ``<region>_GraphIndex.txt`` — graph index (int)
        The resulting dataset is saved to ``processed/SpatialOmicsImageDataset.pt``.

    Notes
    -----
    - The input file ``ImageNameList.txt`` must be located in ``raw_dir``,
      containing one region name per line.
    - The class automatically symmetrizes edge indices when necessary and
      converts NumPy arrays to PyTorch tensors.
    - This class is intended for use with HRCHY-CytoCommunity and compatible
      with PyTorch Geometric’s standard data pipeline.

    Examples
    --------
    >>> from hrchy_cytocommunity.models.dataset import SpatialOmicsImageDataset
    >>> dataset = SpatialOmicsImageDataset(root="data/HRCHY_input/")
    >>> print(len(dataset))
    5
    >>> print(dataset[0])
    Data(x=[1024, 30], edge_index=[2, 4096], graph_idx=[1])
    """                                
    def __init__(self, root, transform=None, pre_transform=None):
        super(SpatialOmicsImageDataset, self).__init__(root, transform, pre_transform)  
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['SpatialOmicsImageDataset.pt']                                           

    def download(self):
        pass
    
    def process(self):
        ## Construct ordinary Python list to hold all input graphs.
        Region_filename = os.path.join(self.raw_dir , "ImageNameList.txt")
        region_name_list = pd.read_csv(
                Region_filename,
                sep="\t",  # tab-separated
                header=None,  # no heading row
                names=["Image"],  # set our own names for the columns
            )
        
        data_list = []
        for i in range(0, len(region_name_list)):
            region_name = region_name_list.Image[i]
            print(f"{region_name} is processing !")
            # Import network topology.
            EdgeIndex_filename = os.path.join(self.raw_dir, f"{region_name}_EdgeIndex.txt")
            edge_ndarray = numpy.loadtxt(EdgeIndex_filename, dtype = 'int64', delimiter = "\t")
            edge_index = torch.from_numpy(edge_ndarray)
            #print(edge_index.type()) #should be torch.LongTensor due to its dtype=torch.int64

            # Import node attribute.
            NodeAttr_filename = os.path.join(self.raw_dir, f"{region_name}_NodeAttr.txt")
            x_ndarray = numpy.loadtxt(NodeAttr_filename, dtype='float32', delimiter="\t")  #should be float32 not float or float64.
            x = torch.from_numpy(x_ndarray)
            #print(x.type()) #should be torch.FloatTensor not torch.DoubleTensor.

            # Import graph index.
            GraphIndex_filename = os.path.join(self.raw_dir, f"{region_name}_GraphIndex.txt")
            graph_index = numpy.loadtxt(GraphIndex_filename, dtype = 'int', delimiter="\t")
            graph_idx = torch.from_numpy(graph_index)
            
            data = Data(x=x, edge_index=edge_index.t().contiguous(), graph_idx=graph_idx)
            data_list.append(data)
        # Read data_list into huge `Data` list.
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])