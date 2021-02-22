# HMSG

This repository provides a reference implementation of the paper: 

HMSG: Heterogeneous graph neural network based on metapath subgraph learning

which is under review.


## Environment settings

- python==3.7.2
- numpy==1.19.5
- scikit-learn==0.23.2
- dgl==0.5.3
- torch==1.7.0


## Basic Usage


We provide two processed datasets:

- Amazon is used for link prediction tasks. 
- IMDB is used for node classification and node clusting tasks. 


### Run

Please run the 'multi.py' for model training and evaluation.

```
python multi.py
```
