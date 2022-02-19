#%%
import os
import argparse
import torch
import numpy as np
import scipy.sparse as sp
from models.GCN import GCN
from models.RSGNN import RSGNN
from dataset import Dataset, get_PtbAdj

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--estimator', type=str, default='MLP',
                    choices=['MLP','GCN'])
parser.add_argument('--mlp_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="cora",
                    choices=['cora', 'cora_ml', 'citeseer','pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='random',
                    choices=['meta', 'random', 'nettack'])
parser.add_argument("--label_rate", type=float, default=0.01, 
                    help='rate of labeled data')
parser.add_argument('--ptb_rate', type=float, default=0.15, 
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=1000, 
                    help='Number of epochs to train.')

parser.add_argument('--alpha', type=float, default=0.01, 
                    help='weight of rec loss')
parser.add_argument('--sigma', type=float, default=100, 
                    help='the parameter to control the variance of sample weights in rec loss')
parser.add_argument('--beta', type=float, default=0.3, help='weight of label smoothness loss')
parser.add_argument('--threshold', type=float, default=0.8, 
                    help='threshold for adj of label smoothing')
parser.add_argument('--t_small',type=float, default=0.1,
                    help='threshold of eliminating the edges')

parser.add_argument('--inner_steps', type=int, default=2, 
                    help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, 
                    help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.001, 
                    help='lr for training adj')
parser.add_argument("--n_p", type=int, default=100, 
                    help='number of positive pairs per node')
parser.add_argument("--n_n", type=int, default=50, 
                    help='number of negitive pairs per node')

parser.add_argument("--r_type",type=str,default="flip",
                    choices=['add','remove','flip'])
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

data = Dataset(root='./data', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

if args.attack == 'no':
    perturbed_adj = adj

if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    import random
    random.seed(15)
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    attacker.attack(adj, n_perturbations, type=args.r_type)
    perturbed_adj = attacker.modified_adj
    file_path = "./data/{}/{}_{}_adj_{}.npz".format(args.label_rate,args.dataset,args.attack,args.ptb_rate)
    sp.save_npz(file_path,perturbed_adj.tocsr())

if args.attack in ['meta','nettack']:
    perturbed_adj = get_PtbAdj(root="./data/{}".format(args.label_rate),
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)

#%%
import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            self_loop=True,
            dropout=args.dropout, device=device).to(device)
rsgnn = RSGNN(model,args,device)
rsgnn.fit(features, perturbed_adj, labels, idx_train, idx_val)
print("=====Test set accuracy=======")
rsgnn.test(idx_test)
# %%
