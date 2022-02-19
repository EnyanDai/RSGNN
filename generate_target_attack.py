#%%
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Pubmed', choices=['cora', 'cora_ml', 'citeseer','Pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.15,  help='pertubation rate')
parser.add_argument("--label_rate", type=float, default=0.1, help='rate of labeled data')
args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
print('cuda: %s' % args.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(15)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='/tmp/', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_train = idx_train[:int(args.label_rate * adj.shape[0])]
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device=device)

surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

def test(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device=device)

    gcn = gcn.to(device)

    gcn.fit(features, adj, labels, idx_train, idx_val, patience=10)

    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Overall test set results:",
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def multi_test_poison():
    # test on 40 nodes on poisoining attack
    cnt = 0
    degrees = adj.sum(0).A1
    np.random.seed(42)
    idx = np.arange(0,adj.shape[0])
    np.random.shuffle(idx)
    node_list = idx[:int(args.ptb_rate*len(idx))]

    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)

    modified_adj = adj
    for target_node in tqdm(node_list):
        n_perturbations = int(degrees[target_node])
        model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
        model = model.to(device)

        model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
        modified_adj = model.modified_adj
        modified_features = model.modified_features
        acc = single_test(modified_adj, modified_features, target_node)
        if acc == 0:
            cnt += 1
    print('misclassification rate : %s' % (cnt/num))
    import os
    import scipy.sparse as sp
    path = os.path.join("./data/{}".format(args.label_rate),"nettack/")
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path,"{}.npz".format(args.dataset))
    if type(modified_adj) is torch.Tensor:
        sparse_adj = to_scipy(modified_adj)
        sp.save_npz(file_path, sparse_adj)
    else:
        sp.save_npz(file_path, modified_adj)



def single_test(adj, features, target_node, gcn=None):
    if gcn is None:
        # test on GCN (poisoning attack)
        gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)

        gcn = gcn.to(device)

        gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
        gcn.eval()
        output = gcn.predict()
    else:
        # test on GCN (evasion attack)
        output = gcn.predict(features, adj)
    probs = torch.exp(output[[target_node]])

    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    return acc_test.item()

#%%
cnt = 0
degrees = adj.sum(0).A1
np.random.seed(42)
idx = np.arange(0,adj.shape[0])
np.random.shuffle(idx)
node_list = idx[:int(args.ptb_rate*len(idx))]
# node_list=[0]

modified_adj = adj
num = len(node_list)
print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
for target_node in tqdm(node_list):
    n_perturbations = int(degrees[target_node])
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
    model = model.to(device)
    model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    # acc = single_test(modified_adj, modified_features, target_node)
    # if acc == 0:
    #     cnt += 1
print('misclassification rate : %s' % (cnt/num))

#%%
import os
import scipy.sparse as sp
path = os.path.join("./data/{}".format(args.label_rate),"nettack/")
if not os.path.exists(path):
    os.makedirs(path)
file_path = os.path.join(path,"{}.npz".format(args.dataset))
sp.save_npz(file_path, modified_adj.tocsr())