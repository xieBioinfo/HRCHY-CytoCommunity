Welcome to HRCHY-CytoCommunity's documentation!
===============================================
.. HRCHY-CytoCommunity documentation master file, created by
   sphinx-quickstart on Sat Oct 11 14:26:09 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HRCHY-CytoCommunity: A computational method for identifying hierarchical tissue structures in a cell phenotype-annotated cellular spatial map.
==============================================================================================================================================




.. toctree::
   :maxdepth: 1
   :caption: Getting Started:
   
   Installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   notebooks/Tutorial_1_CODEX
   notebooks/Tutorial_2_MERFISH
   notebooks/Tutorial_3_MIBITOF

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   user_guide/hyperparameter

.. image:: ../Figures/Schematic.jpg
   :width: 1400

Overview
========

Diverse cell types within a tissue assemble into multicellular structures to shape the functions of the tissue. These structural modules typically consist of specialized subunits, each performing unique roles. Uncovering these hierarchical multicellular structures holds significant importance for gaining deep insights into the assembly principles from individual cells to the entire tissue. However, the methods available for identifying hierarchical tissue structures are quite limited and have several limitations, as listed below:

#. May not be suitable for single-cell-resolution spatial omics data with a limited number of gene or protein expression features available. Instead, cell phenotypes typically serve as better initial cell features for learning tissue structures using such data types.

#. The identified hierarchical tissue structures may not cover all cells within the dataset.

#. There may not be a clearly nested relationship between the identified different levels of tissue structures.

#. Cannot correctly identify tissue structures with spatially discontinuous distribution.

Building upon our recently established tissue structure identification framework `CytoCommunity <https://github.com/huBioinfo/CytoCommunity>`_, we developed **HRCHY-CytoCommunity**, which utilizes a graph neural network (GNN) model to identify hierarchical tissue structures on single-cell spatial maps. HRCHY-CytoCommunity models the identification of hierarchical tissue structures as a MinCut-based hierarchical community detection problem, offering several advantages:

#. HRCHY-CytoCommunity identifies hierarchical tissue structures from a cellular-based perspective, making it suitable for single-cell-resolution spatial omics data, while ensuring that the hierarchical structures cover all cells within the data.

#. By leveraging differentiable graph pooling and graph pruning, HRCHY-CytoCommunity is capable of simultaneously identifying tissue structures of various hierarchical levels at multiple resolutions and exhibiting clearly nested relationships between them.

#. HRCHY-CytoCommunity possesses the ability to discover structures with spatially discontinuous distributions.

#. HRCHY-CytoCommunity employs a consistency training strategy to enhance the stability of the model, while maintaining the unambiguously nested relationship between the hierarchical tissue structures.

#. HRCHY-CytoCommunity utilizes an additional cell-type-enrichment-based clustering module to generate a unified set of nested multicellular structures across all tissue samples, thereby addressing the issue of cross-sample comparative analysis.


Citation
========