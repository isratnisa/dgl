from timeit import default_timer
import functools
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from dgl.dataloading import DataLoader
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

ts = []
dev = torch.device("cuda")

class RGCNHighMem(nn.Module):
    def __init__(self, 
                 in_feat,
                 out_feat,
                 num_rels):
        super().__init__()
        self.weight = torch.randn(num_rels, in_feat, out_feat).to(dev)

    def forward(self,
                g, 
                feat, 
                etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.srcdata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        with Timer(dev) as t:
            weight = self.weight.index_select(0, etypes)
            msg = torch.bmm(edges.src['h'].unsqueeze(1), weight).squeeze(1)
        ts.append(t.elapsed_secs)
        return {'m' : msg}

class RGCNLowMem(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = torch.randn(num_rels, in_feat, out_feat).to(dev)
        self.kernel_t = 0.0

    def forward(self,
                g, 
                feat, 
                etypes):
        # g : DGL block
        # feat : (|V|, D)
        # etypes : (|E|,)
        # sort etypes
        self.kernel_t = 0.0
        with Timer(dev) as t:
            sorted_etypes, index = torch.sort(etypes)
            g = dgl.edge_subgraph(g, index, relabel_nodes=False)
            # Create a new etypes to be an integer list of number of edges.
            num_rels = self.weight.shape[0]
            pos = torch.searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device))
            num = torch.tensor([len(etypes)], device=g.device)
            etypes = (torch.cat([pos[1:], num]) - pos).tolist()
        self.kernel_t += t.elapsed_secs
        # message passing
        g.srcdata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        ts.append(self.kernel_t)
        return g.dstdata['h']

    def message(self, edges, etypes):
        with Timer(dev) as t:
            h_t = torch.split(edges.src['h'], etypes)
            msg = []
            for r in range(self.weight.shape[0]):
                msg.append(torch.matmul(h_t[r], self.weight[r]))
            msg = torch.cat(msg)
        self.kernel_t += t.elapsed_secs
        return {'m' : msg}

class RGCNSegmentMM(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = torch.randn(num_rels, in_feat, out_feat).to(dev)
        self.kernel_t = 0.

    def forward(self,
                g, 
                feat, 
                etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)

        # sort etypes
        self.kernel_t = 0.
        with Timer(dev) as t:
            etypes, index = torch.sort(etypes)
            g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        self.kernel_t += t.elapsed_secs
        # message passing
        g.srcdata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        ts.append(self.kernel_t)
        return g.dstdata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        num_rels = self.weight.shape[0]
        with Timer(dev) as t:
            pos_l = torch.searchsorted(etypes, torch.arange(num_rels, device=h.device))
            pos_r = torch.cat([pos_l[1:], torch.tensor([len(etypes)], device=h.device)])
            seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize

            msg = segment_mm(h, self.weight, seglen_a=seglen)
        self.kernel_t += t.elapsed_secs
        return {'m' : msg}

class RGCNGatherMM(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels):
        super().__init__()
        self.weight = torch.randn(num_rels, in_feat, out_feat).to(dev)

    def forward(self, 
                g, 
                feat, 
                etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.srcdata['h'] = feat
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        return g.dstdata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        with Timer(dev) as t:
            msg = gather_mm(h, self.weight, idx_b=etypes)
        ts.append(t.elapsed_secs)
        return {'m' : msg}

def main(args):
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

    iters = 100 
    in_feat = args.feat_len
    # out_feat = args.feat_len
    out_feat = 16

    # torch.cuda.set_device(dev)

    name = args.dataset
    dataset = data

    g = dgl.to_homogeneous(dataset[0]).to(dev)
    etypes = g.edata[dgl.ETYPE].to(dev)
    num_rels = len(dataset[0].etypes)

    print(f"""Dataset: {name}
    num_nodes: {g.num_nodes()}
    num_edges: {g.num_edges()}
    num_rels: {num_rels}
    in_feat: {in_feat}
    out_feat: {out_feat}
    """)

    feat = torch.randn(g.num_nodes(), in_feat).to(dev)
    weight = torch.randn(num_rels, in_feat, out_feat).to(dev)

    conv = None
    if args.conv == "high":
        conv = RGCNHighMem(in_feat, out_feat, num_rels)
    elif args.conv == "low":
        conv = RGCNLowMem(in_feat, out_feat, num_rels)
    elif args.conv == "gather":
        conv = RGCNGatherMM(in_feat, out_feat, num_rels)
    else:
        conv = RGCNSegmentMM(in_feat, out_feat, num_rels)

    conv.eval()

    # dry run
    for i in range(10):
        h = conv(g, feat, etypes)
    # test
    for i in range(iters):
        with Timer(dev) as t:
            h = conv(g, feat, etypes)
    print(len(ts))
    print(args.conv, "kernel time", "{:.4f}".format(np.mean(ts[10:]) * 1000), "ms\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hetero Kernel Benchmark')
    register_data_args(parser)
    parser.add_argument("--feat_len", type=int, default=16,
            help="feature length")
    parser.add_argument("--conv", type=str, default="high", choices=["high", "low", "gather", "seg"])
    args = parser.parse_args()
    print(args)

    main(args)