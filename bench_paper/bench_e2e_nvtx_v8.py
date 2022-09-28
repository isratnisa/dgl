from timeit import default_timer
import functools
import torch
import torch.nn as nn
import dgl
import dgl.backend as F
import dgl.function as fn
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
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
        if self.device == 'cuda:0':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if self.device == 'cuda:0':
            self.end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic


class RGCNHighMem(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.ndata['h'] = feat
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("hi-mem-updt-all")
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return g.ndata['h']

    def message(self, edges, etypes):
        torch.cuda.nvtx.range_push("idx-select")
        weight = self.weight.index_select(0, etypes)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("bmm")
        msg = {'m' : torch.bmm(edges.src['h'].unsqueeze(1), weight).squeeze(1)}
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        return msg

class RGCNLowMem(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        # sort etypes
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("th.sort")
        sorted_etypes, index = torch.sort(etypes)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("edge-subgr&misc")
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # Create a new etypes to be an integer list of number of edges.
        num_rels = self.weight.shape[0]
        pos = torch.searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device))
        num = torch.tensor([len(etypes)], device=g.device)
        etypes = (torch.cat([pos[1:], num]) - pos).tolist()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        # message passing

        g.ndata['h'] = feat
        torch.cuda.nvtx.range_push("low-mem-updt-all")
        g.update_all(functools.partial(self.message, etypes=etypes), fn.sum('m', 'h'))
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        return g.ndata['h']

    def message(self, edges, etypes):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("matmul")
        h_t = torch.split(edges.src['h'], etypes)
        msg = []
        for r in range(self.weight.shape[0]):
            msg.append(torch.matmul(h_t[r], self.weight[r]))
        torch.cuda.nvtx.range_pop()
        return {'m' : torch.cat(msg)}

class RGCNSegmentMM(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)

        # sort etypes
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("th.sort&edgesubgraph")
        etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        torch.cuda.nvtx.range_pop()
        # message passing
        g.ndata['h'] = feat
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("update_all-segmm")
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        torch.cuda.nvtx.range_pop()
        return g.ndata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        w = self.weight.view(-1, self.weight.shape[2])
        num_rels = self.weight.shape[0]
        torch.cuda.nvtx.range_push("comp-seg-len")
        out = torch.zeros((h.shape[0], self.weight.shape[2]), dtype=torch.float32, device=h.device)
        # dgl.sparse._gather_mm(h, w, out, E_per_rel, etypes, sortedE=True)
        pos_l = torch.searchsorted(etypes, torch.arange(num_rels, device=h.device))
        pos_r = torch.cat([pos_l[1:], torch.tensor([len(etypes)], device=h.device)])
        seglen = (pos_r - pos_l).cpu()  # XXX(minjie): cause device synchronize
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("segmm")
        m = segment_mm(h, self.weight, seglen_a=seglen)
        torch.cuda.nvtx.range_pop()
        return {'m' : m}

class RGCNGatherMM(nn.Module):
    def __init__(self, weight):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, g, feat, etypes):
        # g : DGLGraph
        # feat : (|V|, D)
        # etypes : (|E|,)
        g.ndata['h'] = feat
        torch.cuda.nvtx.range_push("update-all_gmm")
        g.update_all(functools.partial(self.message, etypes=etypes),
                     fn.sum('m', 'h'))
        torch.cuda.nvtx.range_pop()
        return g.ndata['h']

    def message(self, edges, etypes):
        h = edges.src['h']
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("index-selct")
        w = self.weight.view(-1, self.weight.shape[2])
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("gathermm")
        m = gather_mm(h, self.weight, idx_b=etypes)
        torch.cuda.nvtx.range_pop()
        return {'m' : m}

class RGCNHetero(nn.Module):
    def __init__(self, weight, etypes):
        # weight : (|R|, D_in, D_out)
        super().__init__()
        self.weight = nn.ParameterDict({
            etype : nn.Parameter(weight[i]) for i, etype in enumerate(etypes)})

    def forward(self, hg, feat_dict):
        # hg : DGLGraph hetero
        # feat : dict of tensors
        for ntype in hg.ntypes:
            hg.nodes[ntype].data['h'] = feat_dict[ntype]
        fns = {}

        for rel in hg.canonical_etypes:
            fns[rel] = (
                functools.partial(self.message, weight=self.weight[rel[1]]),
                fn.sum('m', 'h')
            )
        hg.multi_update_all(fns, 'sum')
        return {ntype : hg.nodes[ntype].data['h'] for ntype in hg.ntypes}

    def message(self, edges, weight):
        return {'m' : edges.src['h'] @ weight}

dev = "cuda:0"

def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    elif args.dataset == 'aifb':
        data = AIFBDataset()
    elif args.dataset == 'mutag':
        data = MUTAGDataset()
    elif args.dataset == 'bgs':
        data = BGSDataset()
    elif args.dataset == 'am':
        data = AMDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    iters = 20
    in_feat = args.feat_len
    out_feat = args.feat_len

    torch.cuda.set_device(dev)

    name = args.dataset
    dataset = data

    g = dgl.to_homogeneous(dataset[0]).to(dev)
    etypes = g.edata[dgl.ETYPE].long().to(dev)
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

    # **** low-mem ******
    conv = RGCNLowMem(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_lowmem = conv(g, feat, etypes)
    print("low-mem rgcn:", t.elapsed_secs / iters * 1000, "ms\n")

    # **** high-mem ******
    conv = RGCNHighMem(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_highmem = conv(g, feat, etypes)
    print("high-mem rgcn:", t.elapsed_secs / iters * 1000, "ms\n")


    # **** gather_mm sorted ****
    conv = RGCNSegmentMM(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_gmm_sorted = conv(g, feat, etypes)
    print("seg_mm rgcn:", t.elapsed_secs / iters * 1000, "ms\n")

    # **** gather_mm unsorted ****
    conv = RGCNGatherMM(weight).to(dev)
    # dry run
    for i in range(3):
        h = conv(g, feat, etypes)
    torch.cuda.synchronize()
    # test
    with Timer(dev) as t:
        for i in range(iters):
            h_gmm_unsorted = conv(g, feat, etypes)
    print("gather_mm_unsorted rgcn:", t.elapsed_secs / iters * 1000, "ms")

    # # **** hetero ****
    # hg = dataset[0].to(dev)
    # conv = RGCNHetero(weight, hg.etypes).to(dev)
    # feat_dict = {ntype : torch.randn(hg.num_nodes(ntype), in_feat).to(dev) for ntype in hg.ntypes}
    # # dry run
    # for i in range(3):
    #     h_dict = conv(hg, feat_dict)
    # torch.cuda.synchronize()
    # # test
    # with Timer(dev) as t:
    #     for i in range(iters):
    #         h_dict = conv(hg, feat_dict)
    # print("hetero rgcn:", t.elapsed_secs / iters * 1000, "ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--feat_len", type=int, default=16,
            help="feature length")
    args = parser.parse_args()
    print(args)

    main(args)

