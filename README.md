
# RS-GNN
An offical PyTorch implementation of "Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels" (WSDM 2022). [[paper]](https://arxiv.org/pdf/2201.00232.pdf)
## Overview
* `./models`: This directory contains the model of RSGNN.
* `./dataset.py`: This is the code to load datasets and perturbed adjacency matrix.
* `./data`: The pre-perturbed adjacency matrices of the datasets are stored here.
* `./scripts`: It contains the scripts to reproduce the major reuslts of our paper.
* `./generate_attack.py`: An example code of obtaining the perturbed dataset. To run this code, it is required to install [DeepRobust](https://deeprobust.readthedocs.io/en/latest/)
* `./train_RSGNN.py`: The program to train RSGNN model.

## Dataset
The original **Cora, Cora-ML, Citeseer, and Pubmed** will be automatically downloaded to `./data`. The val and test indices are the same as nettack settings. 

For the perturbed adjacency matrix, it is stored as: `./data/{label_rate}/{dataset}_{attack_method}_adj_{ptb_rate}.npz`. 

## Requirements

```
torch==1.7.1
torch-geometric==1.7.2 
```

## Experiments
To reproduce the performance in the paper, you can run the bash files in the `.\scripts`. For example, to get results on cora datasets
```
bash scripts\train_cora.sh
```


## Cite
If you find this repo to be useful, please cite our paper. Thank you.
```
@article{dai2022towards,
  title={Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels},
  author={Dai, Enyan and Jin, Wei and Liu, Hui and Wang, Suhang},
  journal={WSDM},
  year={2022}
}
```
