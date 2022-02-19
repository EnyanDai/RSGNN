#%%
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy,sparse_mx_to_torch_sparse_tensor
import torch_geometric.utils as utils
from models.GCN import GCN
import scipy.sparse as sp
import numpy as np
class RSGNN:


    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = model.to(device)

    def fit(self, features, adj, labels, idx_train, idx_val):
        """Train RS-GNN.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices
        """
        args = self.args
        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        labels = torch.LongTensor(np.array(labels)).to(self.device)

        self.features = features
        self.labels = labels


        self.estimator = EstimateAdj(edge_index, features, args, device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
        self.optimizer_adj = optim.Adam(self.estimator.parameters(),
                               lr=args.lr_adj, weight_decay=args.weight_decay)

        # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
            for i in range(int(args.outer_steps)):
                self.train_adj(epoch, features, edge_index, labels,
                        idx_train, idx_val)

            for i in range(int(args.inner_steps)):
                self.train_gcn(epoch, features, edge_index,
                        labels, idx_train, idx_val)

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # Testing
        print("picking the best model according to validation performance")
        self.model.load_state_dict(self.weights)

        print("=====validation set accuracy=======")
        self.test(idx_val)
        print("===================================")

    def train_gcn(self, epoch, features, edge_index, labels, idx_train, idx_val):
        args = self.args

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())
        acc_train = accuracy(output[idx_train], labels[idx_train])


        loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
        loss_label_smooth = self.label_smoothing(self.estimator.poten_edge_index,\
                                                 self.estimator.estimated_weights.detach(),\
                                                 output, idx_train, self.args.threshold)
        loss = loss_train  + self.args.beta * loss_label_smooth
        loss.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = self.estimator.estimated_weights.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())

        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()),
                      'time: {:.4f}s'.format(time.time() - t))



    def train_adj(self, epoch, features, edge_index, labels, idx_train, idx_val):
        args = self.args
        if args.debug:
            print("\n=== train_adj ===")
        t = time.time()

        self.optimizer_adj.zero_grad()

        rec_loss = self.estimator(edge_index,features)

        output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights)
        loss_gcn = F.cross_entropy(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        loss_label_smooth = self.label_smoothing(self.estimator.poten_edge_index,\
                                                 self.estimator.estimated_weights.detach(),\
                                                 output, idx_train, self.args.threshold)


        total_loss = loss_gcn + args.alpha *rec_loss


        total_loss.backward()

        self.optimizer_adj.step()


        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, self.estimator.poten_edge_index, self.estimator.estimated_weights.detach())

        loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = self.estimator.estimated_weights.detach()
            self.weights = deepcopy(self.model.state_dict())
            if args.debug:
                print('\t=== saving current graph/gcn, best_val_acc: %s' % self.best_val_acc.item())


        if args.debug:
            if epoch % 1 == 0:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_gcn: {:.4f}'.format(loss_gcn.item()),
                      'rec_loss: {:.4f}'.format(rec_loss.item()),
                      'loss_label_smooth: {:.4f}'.format(loss_label_smooth.item()),
                      'loss_total: {:.4f}'.format(total_loss.item()))
                print('Epoch: {:04d}'.format(epoch+1),
                        'acc_train: {:.4f}'.format(acc_train.item()),
                        'loss_val: {:.4f}'.format(loss_val.item()),
                        'acc_val: {:.4f}'.format(acc_val.item()),
                        'time: {:.4f}s'.format(time.time() - t))


    def test(self, idx_test):
        """Evaluate the performance of ProGNN on test set
        """
        print("\t=== testing ===")
        features = self.features
        labels = self.labels
        self.model.eval()
        estimated_weights = self.best_graph
        if self.best_graph is None:
            estimated_weights = self.estimator.estimated_weights
        output = self.model(features, self.estimator.poten_edge_index,estimated_weights)
        loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        return float(acc_test)

    
    def label_smoothing(self, edge_index, edge_weight, representations, idx_train, threshold):


        num_nodes = representations.shape[0]
        n_mask = torch.ones(num_nodes, dtype=torch.bool).to(self.device)
        n_mask[idx_train] = 0

        mask = n_mask[edge_index[0]] \
                & (edge_index[0] < edge_index[1])\
                & (edge_weight >= threshold)\
                | torch.bitwise_not(n_mask)[edge_index[1]]

        unlabeled_edge = edge_index[:,mask]
        unlabeled_weight = edge_weight[mask]

        Y = F.softmax(representations)

        loss_smooth_label = unlabeled_weight\
                            @ torch.pow(Y[unlabeled_edge[0]] - Y[unlabeled_edge[1]], 2).sum(dim=1)\
                            / num_nodes

        return loss_smooth_label
                        
#%%
class EstimateAdj(nn.Module):
    """Provide a pytorch parameter matrix for estimated
    adjacency matrix and corresponding operations.
    """

    def __init__(self, edge_index, features, args ,device='cuda'):
        super(EstimateAdj, self).__init__()

        
        if args.estimator=='MLP':
            self.estimator = nn.Sequential(nn.Linear(features.shape[1],args.mlp_hidden),
                                    nn.ReLU(),
                                    nn.Linear(args.mlp_hidden,args.mlp_hidden))
        else:
            self.estimator = GCN(features.shape[1], args.mlp_hidden, args.mlp_hidden,dropout=0.0,device=device)
        self.device = device
        self.args = args
        self.poten_edge_index = self.get_poten_edge(edge_index,features,args.n_p)
        self.features_diff = torch.cdist(features,features,2)
        self.estimated_weights = None


    def get_poten_edge(self, edge_index, features, n_p):

        if n_p == 0:
            return edge_index

        poten_edges = []
        for i in range(len(features)):
            sim = torch.div(torch.matmul(features[i],features.T), features[i].norm()*features.norm(dim=1))
            _,indices = sim.topk(n_p)
            poten_edges.append([i,i])
            indices = set(indices.cpu().numpy())
            indices.update(edge_index[1,edge_index[0]==i])
            for j in indices:
                if j > i:
                    pair = [i,j]
                    poten_edges.append(pair)
        poten_edges = torch.as_tensor(poten_edges).T
        poten_edges = utils.to_undirected(poten_edges,len(features)).to(self.device)

        return poten_edges
    

    def forward(self, edge_index, features):

        if self.args.estimator=='MLP':
            representations = self.estimator(features)
        else:
            representations = self.estimator(features,edge_index,\
                                            torch.ones([edge_index.shape[1]]).to(self.device).float())
        rec_loss = self.reconstruct_loss(edge_index, representations)

        x0 = representations[self.poten_edge_index[0]]
        x1 = representations[self.poten_edge_index[1]]
        output = torch.sum(torch.mul(x0,x1),dim=1)

        self.estimated_weights = F.relu(output)
        self.estimated_weights[self.estimated_weights < self.args.t_small] = 0.0
        

        return rec_loss
    
    def reconstruct_loss(self, edge_index, representations):
        
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index,num_nodes=num_nodes, num_neg_samples=self.args.n_n*num_nodes)
        randn = randn[:,randn[0]<randn[1]]

        edge_index = edge_index[:, edge_index[0]<edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0,neg1),dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0,pos1),dim=1)

        neg_loss = torch.exp(torch.pow(self.features_diff[randn[0],randn[1]]/self.args.sigma,2)) @ F.mse_loss(neg,torch.zeros_like(neg), reduction='none')
        pos_loss = torch.exp(-torch.pow(self.features_diff[edge_index[0],edge_index[1]]/self.args.sigma,2)) @ F.mse_loss(pos, torch.ones_like(pos), reduction='none')

        rec_loss = (pos_loss + neg_loss) \
                    * num_nodes/(randn.shape[1] + edge_index.shape[1]) 
        

        return rec_loss

# %%
