# GNNforFakeNewsDetection
GNN Methods for Fake News Detection. CPSC 483 Final Project, Fall 2024, Andrew Xu and Annli Zhu.

Download the PHEME dataset from [here](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078). The FakeNewsNet dataset is accessed through the UPFD wrapper on PyTorch Geometric.

Preprocess the PHEME dataset for the Heterogeneous Graph Transformer (HGT) using "pheme_data_parsing.py". Resulting pickles are in `pheme_graphs` directory. Visualize the data using "pheme.ipynb". Run the data through the HGT pipeline using "HGT_pipeline.ipynb" (non-functional). 

Preprocess the PHEME dataset for the Temporal Graph Attention Network (TGAT) using "process_for_TGAT.py". Resulting files should be in `processed` directory, but were too large to fit into Github. We run the pipeline from this [repository](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) using the preprocessed files (non-functional). 

We conducted preliminary baseline results on FakeNewsNet using Graph Attention Networks (GAT). The python script is listed in "upfd.py". Data of results over 60 epochs are presented under the `results` directory, which are then analyzed for the paper results.

To replicate our baseline results, run the following command `python upfd.py --dataset="politifact" --feature="bert" --model="GAT" > results/GAT_gossipcop_out1.txt`.
