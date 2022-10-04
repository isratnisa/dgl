from timeit import default_timer
import functools
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.dataloading import MultiLayerNeighborSampler, DataLoader
import torch.nn as nn
import time
import numpy as np
import argparse, time, math
from dgl.data import register_data_args

from dgl.ops import segment_mm, gather_mm

#####################################
# Timer code from benchmarks folder
#####################################
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

class RGCNHighMemConv(nn.Module):
    def __init__(self, 
                 in_feat,
                 out_feat,
                 num_rels):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(self.weight.shape[1]), 1/math.sqrt(self.weight.shape[1]))

    def forward(self,
                g, 
                feat, 
                etypes,
                norm=None):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        # norm (optional) : (|E|,)
        g.srcdata['h'] = feat
        if norm is not None:
            g.edata['norm'] = norm
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        weight = self.weight.index_select(0, etypes)
        msg = torch.bmm(edges.src['h'].unsqueeze(1), weight).squeeze(1)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'m' : msg}

class RGCNLowMemConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(self.weight.shape[1]), 1/math.sqrt(self.weight.shape[1]))

    def forward(self,
                g, 
                feat, 
                etypes,
                norm=None):
        # g : DGL block
        # feat : (|V|, D)
        # etypes : (|E|,)
        # sort etypes
        sorted_etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # Create a new etypes to be an integer list of number of edges.
        num_rels = self.weight.shape[0]
        pos = torch.searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device))
        num = torch.tensor([len(etypes)], device=g.device)
        etypes = (torch.cat([pos[1:], num]) - pos).tolist()
        # message passing
        g.srcdata['h'] = feat
        if norm is not None:
            g.edata['norm'] = norm
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        h_t = torch.split(edges.src['h'], etypes)
        msg = []
        for r in range(self.weight.shape[0]):
            msg.append(torch.matmul(h_t[r], self.weight[r]))
        msg = torch.cat(msg)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'m' : msg}

class RGCNSegmentMMConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(self.weight.shape[1]), 1/math.sqrt(self.weight.shape[1]))

    def forward(self,
                g, 
                feat, 
                etypes,
                norm=None):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        # norm (optional) : (|E|,)

        # sort etypes
        etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # message passing
        g.srcdata['h'] = feat
        if norm is not None:
            g.edata['norm'] = norm
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        # w = self.weight.view(-1, self.weight.shape[2])
        num_rels = self.weight.shape[0]
        # out = torch.zeros((h.shape[0], self.weight.shape[2]), dtype=torch.float32, device=h.device)
        pos_l = torch.searchsorted(etypes, torch.arange(num_rels, device=h.device))
        pos_r = torch.cat([pos_l[1:], torch.tensor([len(etypes)], device=h.device)])
        seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize

        msg = segment_mm(h, self.weight, seglen_a=seglen)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'m' : msg}

class RGCNGatherMMConv(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_rels, in_feat, out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(self.weight.shape[1]), 1/math.sqrt(self.weight.shape[1]))

    def forward(self, 
                g, 
                feat, 
                etypes,
                norm=None):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        # norm (optional): (|E|,) 
        g.srcdata['h'] = feat
        if norm is not None:
            g.edata['norm'] = norm
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        # w = self.weight.view(-1, self.weight.shape[2])
        msg = gather_mm(h, self.weight, idx_b=etypes)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'m' : msg}



def main_wg(args, data):
    class RGCN(nn.Module):
        def __init__(self,
                    num_nodes,
                    h_dim,
                    out_dim,
                    num_rels,
                    conv="high"):
            super().__init__()
            self.emb = nn.Embedding(num_nodes, h_dim)
            if conv == 'high':
                print(f'Using high-mem Conv')
                self.conv1 = RGCNHighMemConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNHighMemConv(h_dim, out_dim, num_rels)
            elif conv == 'low':
                print(f'Using low-mem Conv')
                self.conv1 = RGCNLowMemConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNLowMemConv(h_dim, out_dim, num_rels)
            elif conv == 'seg':
                print(f'Using segment_mm Conv')
                self.conv1 = RGCNSegmentMMConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNSegmentMMConv(h_dim, out_dim, num_rels)
            else:
                print(f'Using gather_mm Conv')
                self.conv1 = RGCNGatherMMConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNGatherMMConv(h_dim, out_dim, num_rels)

        def forward(self, g):
            x = self.emb.weight
            h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata['norm']))
            h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata['norm'])
            return h

    def evaluate(g, target_idx, labels, test_mask, model):
        test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
        model.eval()
        with torch.no_grad():
            logits = model(g)
        logits = logits[target_idx]
        return accuracy(logits[test_idx].argmax(dim=1), labels[test_idx]).item()

    def train(g, target_idx, labels, train_mask, model):
        # define train idx, loss function and optimizer
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

        model.train()
        ts = []
        for epoch in range(args.epoch):
            with Timer(g.device) as t:
                logits = model(g)
                logits = logits[target_idx]
                loss = loss_fcn(logits[train_idx], labels[train_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc = accuracy(logits[train_idx].argmax(dim=1), labels[train_idx]).item()
            print("Epoch {:05d} | Loss {:.4f} | Train Accuracy {:.4f} | Time {:.4f} ms"
                .format(epoch, loss.item(), acc, t.elapsed_secs * 1000))
            if epoch >= 3:
                ts.append(t.elapsed_secs)
        print("Average e2e 2-layers training time {:.4f} ms".format(np.mean(ts) * 1000))
        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training with whole graph RGCN module.')
    g = data[0]
    g = g.to(device)
    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    labels = g.nodes[category].data.pop('labels')
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    # calculate normalization weight for each edge, and find target category and node id
    for cetype in g.canonical_etypes:
        g.edges[cetype].data['norm'] = dgl.norm_by_dst(g, cetype).unsqueeze(1)
    category_id = g.ntypes.index(category)
    g = dgl.to_homogeneous(g, edata=['norm'])
    node_ids = torch.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    # create RGCN model    
    in_size = g.num_nodes() 
    out_size = data.num_classes
    model = RGCN(in_size, args.hdim, out_size, num_rels, args.conv).to(device)
    
    train(g, target_idx, labels, train_mask, model)
    acc = evaluate(g, target_idx, labels, test_mask, model)
    print("Test accuracy {:.4f}".format(acc))
        
def main_mb(args, data):
    class RGCN(nn.Module):
        def __init__(self,
                    num_nodes,
                    h_dim,
                    out_dim,
                    num_rels,
                    conv="high"):
            super().__init__()
            self.emb = nn.Embedding(num_nodes, h_dim)
            if conv == 'high':
                print(f'Using high-mem Conv')
                self.conv1 = RGCNHighMemConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNHighMemConv(h_dim, out_dim, num_rels)
            elif conv == 'low':
                print(f'Using low-mem Conv')
                self.conv1 = RGCNLowMemConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNLowMemConv(h_dim, out_dim, num_rels)
            elif conv == 'seg':
                print(f'Using segment_mm Conv')
                self.conv1 = RGCNSegmentMMConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNSegmentMMConv(h_dim, out_dim, num_rels)
            else:
                print(f'Using gather_mm Conv')
                self.conv1 = RGCNGatherMMConv(h_dim, h_dim, num_rels)
                self.conv2 = RGCNGatherMMConv(h_dim, out_dim, num_rels)

        def forward(self, g):
            x = self.emb(g[0].srcdata[dgl.NID])
            h = F.relu(self.conv1(g[0], x, g[0].edata[dgl.ETYPE], g[0].edata['norm']))
            h = self.conv2(g[1], h, g[1].edata[dgl.ETYPE], g[1].edata['norm'])
            return h
    
    def evaluate(model, labels, dataloader, inv_target):
        model.eval()
        eval_logits = []
        eval_seeds = []
        with torch.no_grad():
            for input_nodes, output_nodes, blocks in dataloader:
                output_nodes = inv_target[output_nodes]
                for block in blocks:
                    block.edata['norm'] = dgl.norm_by_dst(block).unsqueeze(1)
                logits = model(blocks)
                eval_logits.append(logits)
                eval_seeds.append(output_nodes)
        eval_logits = torch.cat(eval_logits)
        eval_seeds = torch.cat(eval_seeds)
        return accuracy(eval_logits.argmax(dim=1), labels[eval_seeds]).item()
    
    def train(device, g, target_idx, labels, train_mask, model, inv_target):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
        # construct sampler and dataloader
        sampler = MultiLayerNeighborSampler(args.fanout)
        train_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                                  batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                                batch_size=args.batch_size, shuffle=False)
        
        ts = []
        for epoch in range(args.epoch):
            model.train()
            total_loss = 0
            with Timer(device) as t:
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                    output_nodes = inv_target[output_nodes]
                    for block in blocks:
                        block.edata['norm'] = dgl.norm_by_dst(block).unsqueeze(1)
                    logits = model(blocks)
                    loss = loss_fcn(logits, labels[output_nodes])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # total_loss += loss.item()
            acc = evaluate(model, labels, val_loader, inv_target)
            print("Epoch {:05d} | Val. Accuracy {:.4f} | Time {:.4f} ms "
                  .format(epoch, acc, t.elapsed_secs * 1000))
            if epoch >= 3:
                ts.append(t.elapsed_secs)
        print("Average e2e 2-layers training time {:.4f} ms".format(np.mean(ts) * 1000))
    
    g = data[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    labels = g.nodes[category].data.pop('labels').to(device)
    train_mask = g.nodes[category].data.pop('train_mask').to(device)
    test_mask = g.nodes[category].data.pop('test_mask').to(device)

    # find target category and node id
    category_id = g.ntypes.index(category)
    g = dgl.to_homogeneous(g).to(device)
    node_ids = torch.arange(g.num_nodes()).to(device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE)
    g.ndata['type_id'] = g.ndata.pop(dgl.NID)

    # find the mapping (inv_target) from global nodes IDs to type-specific node IDs
    inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64).to(device)
    inv_target[target_idx] = torch.arange(0, target_idx.shape[0], dtype=inv_target.dtype).to(device)

    # create RGCN model
    in_size = g.num_nodes()
    out_size = data.num_classes
    model = RGCN(in_size, args.hdim, out_size, num_rels).to(device)
    train(device, g, target_idx, labels, train_mask, model, inv_target)

    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    test_sampler = MultiLayerNeighborSampler([-1, -1])
    test_loader = DataLoader(g, target_idx[test_idx], test_sampler, device=device,
                            batch_size=32, shuffle=False)
    acc = evaluate(model, labels, test_loader, inv_target)
    print("Test accuracy {:.4f}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification')
    parser.add_argument("--dataset", type=str, default="aifb",
                        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').")
    parser.add_argument("--conv", type=str, default="high", choices={'high', 'low', 'gather', 'seg'})
    parser.add_argument("--hdim", type=int, default=16)
    parser.add_argument("--fanout", type=int, nargs=2)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()

    # load and preprocess dataset
    if args.dataset == 'aifb':
        data = AIFBDataset()
    elif args.dataset == 'mutag':
        data = MUTAGDataset()
    elif args.dataset == 'bgs':
        data = BGSDataset()
    elif args.dataset == 'am':
        data = AMDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.fanout:
        main_mb(args, data)
    else:
        main_wg(args, data)
