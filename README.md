# GREY
Gradient-free optimizing agent by meta-REinforcement learning to Yield logic parameters of Boolean networks.
 
# Requirements
- tensorflow==1.15.5
- tensorflow-probability==0.8.0
- numba==0.53.1
- networkx==2.5.1
- [graph-nets](https://github.com/deepmind/graph_nets)==1.1.0
- scikit-learn==0.24.2

GREY has been tested in Linux and Python 3.6.9.

# Instructions

**Required data**

Network, genomic mutation profile and AUC(drug response value) are required to run GREY.

`example_data/` directory has example files for trametinib as follows.

`example_data/trametinib_network.csv` - Drug response core network for trametinib. (Column-source, Row-target)
`example_data/Core_profile_MM.csv` - Genomic mutation profile of nodes(genes) in the `trametinib_network.csv`. (Column-Cell line, Row-Gene, AUC)

**Usage**

`run_example.ipynb` has example codes to run GREY.