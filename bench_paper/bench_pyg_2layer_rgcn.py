from timeit import default_timer
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.utils import k_hop_subgraph

class Timer:
    def __init__(self, device):
        self.timer = default_timer
        self.device = device

    def __enter__(self):
        if str(self.device).startswith('cuda'):
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if str(self.device).startswith('cuda'):
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--fast', action='store_true', help='use FastRGCNConv')
parser.add_argument('--feat-len', type=int, default=16)
args = parser.parse_args()

# Trade memory consumption for faster computation.
if args.fast:
    RGCNConv = FastRGCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]

# BGS and AM graphs are too big to process them in a full-batch fashion.
# Since our model does only make use of a rather small receptive field, we
# filter the graph to only contain the nodes that are at most 2-hop neighbors
# away from any training/test node.
# node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
# node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
#     node_idx, 2, data.edge_index, relabel_nodes=True)

# data.num_nodes = node_idx.size(0)
# data.edge_index = edge_index
# data.edge_type = data.edge_type[edge_mask]
# data.train_idx = mapping[:data.train_idx.size(0)]
# data.test_idx = mapping[data.train_idx.size(0):]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(data.num_nodes, args.feat_len)
        self.conv1 = RGCNConv(args.feat_len, 16, dataset.num_relations,
                              num_bases=None, root_weight=False, bias=False)
        self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations,
                              num_bases=None, root_weight=False, bias=False)

    def forward(self, edge_index, edge_type):
        x = self.emb.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu') if args.dataset == 'AM' else device
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc

ts = []
for epoch in range(1, 110):
    with Timer(device) as t:
        loss = train()
    # train_acc, test_acc = test()
    # print(f'Epoch: {epoch:02d}, ', f'Loss: {loss:.4f}, ', f'Train Acc: {train_acc:.4f}, ', f'Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:02d}, ' 
          f'Time: {t.elapsed_secs * 1000:.4f}')
    if epoch >= 10:
        ts.append(t.elapsed_secs)
print(f'Average e2e 2layer trainning time: {np.mean(ts) * 1000:.4f} ms')
