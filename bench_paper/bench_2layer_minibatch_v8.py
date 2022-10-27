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
        if norm is not None:
            g.edata['norm'] = norm
        sorted_etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # Create a new etypes to be an integer list of number of edges.
        num_rels = self.weight.shape[0]
        pos = torch.searchsorted(sorted_etypes, torch.arange(num_rels, device=g.device))
        num = torch.tensor([len(etypes)], device=g.device)
        etypes = (torch.cat([pos[1:], num]) - pos).tolist()
        # message passing
        g.srcdata['h'] = feat
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
        if norm is not None:
            g.edata['norm'] = norm
        etypes, index = torch.sort(etypes)
        g = dgl.edge_subgraph(g, index, relabel_nodes=False)
        # message passing
        g.srcdata['h'] = feat
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

class RGCN(nn.Module):
    def __init__(self,
                num_nodes,
                in_dim,
                h_dim,
                out_dim,
                device,
                num_rels,
                conv="high"):
        super().__init__()
        # self.emb = nn.Embedding(num_nodes, in_dim)
        self.emb = torch.randn(num_nodes, in_dim).to(device)
        if conv == 'high':
            print(f'Using high-mem Conv')
            self.conv1 = RGCNHighMemConv(in_dim, h_dim, num_rels)
            self.conv2 = RGCNHighMemConv(h_dim, out_dim, num_rels)
        elif conv == 'low':
            print(f'Using low-mem Conv')
            self.conv1 = RGCNLowMemConv(in_dim, h_dim, num_rels)
            self.conv2 = RGCNLowMemConv(h_dim, out_dim, num_rels)
        elif conv == 'seg':
            print(f'Using segment_mm Conv')
            self.conv1 = RGCNSegmentMMConv(in_dim, h_dim, num_rels)
            self.conv2 = RGCNSegmentMMConv(h_dim, out_dim, num_rels)
        else:
            print(f'Using gather_mm Conv')
            self.conv1 = RGCNGatherMMConv(in_dim, h_dim, num_rels)
            self.conv2 = RGCNGatherMMConv(h_dim, out_dim, num_rels)

    def forward(self, g):
        x = self.emb[g[0].srcdata[dgl.NID]]
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
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(output_nodes.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)
    return accuracy(eval_logits.argmax(dim=1), labels[eval_seeds].cpu()).item()

def train(args, device, g, target_idx, labels, train_mask, model, inv_target):
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    # construct sampler and dataloader
    sampler = MultiLayerNeighborSampler(args.fanout)
    train_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                                batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(g, target_idx[train_idx], sampler, device=device,
                            batch_size=args.batch_size, shuffle=False)
    
    ts, fs, bs = [], [], [] 
    for epoch in range(args.epoch):
        model.train()
        # total_loss = 0
        epoch_t, forward_t, backward_t = 0.0, 0.0, 0.0
        with Timer(device) as t1:
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
                output_nodes = inv_target[output_nodes]
                for block in blocks:
                    block.edata['norm'] = dgl.norm_by_dst(block).unsqueeze(1)
                with Timer(device) as t2:
                    logits = model(blocks)
                    loss = loss_fcn(logits, labels[output_nodes])
                with Timer(device) as t3:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                forward_t += t2.elapsed_secs
                backward_t += t3.elapsed_secs
        epoch_t += t1.elapsed_secs
                
            # total_loss += loss.item()
        # acc = evaluate(model, labels, val_loader, inv_target)
        print("Epoch {:05d} | Time {:.4f} ms | Forward Time {:.4f} ms | Backward Time {:.4f} "
                .format(epoch, epoch_t*1000, forward_t*1000, backward_t*1000))
    #     if epoch >= 10:
    #         ts.append(t1.elapsed_secs - forward_t - backward_t)
    #         fs.append(forward_t)
    #         bs.append(backward_t)
    # print("Average e2e minibatch 2-layers training times {:.4f} {:.4f} {:.4f} ms"
    #         .format(np.mean(ts)*1000, np.mean(fs)*1000, np.mean(bs)*1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN for entity classification (minibatch)')
    parser.add_argument("--dataset", type=str, default="aifb",
                        help="Dataset name ('aifb', 'mutag', 'bgs', 'am').")
    parser.add_argument("--conv", type=str, default="high", choices={'high', 'low', 'gather', 'seg'})
    parser.add_argument("--hdim", type=int, default=16)
    parser.add_argument("--fanout", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=110)
    parser.add_argument("--sample", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--indim", type=int, default=16)
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
    
    gpu_device = torch.device('cuda')
    cpu_device = torch.device('cpu')

    g = data[0]
    num_rels = len(g.canonical_etypes)
    category = data.predict_category
    labels = g.nodes[category].data.pop('labels').to(gpu_device)
    train_mask = g.nodes[category].data.pop('train_mask').to(cpu_device if args.sample == "cpu" else gpu_device)
    test_mask = g.nodes[category].data.pop('test_mask').to(cpu_device if args.sample == "cpu" else gpu_device)

    # find target category and node id
    category_id = g.ntypes.index(category)
    g = dgl.to_homogeneous(g).to(cpu_device if args.sample == "cpu" else gpu_device)
    node_ids = torch.arange(g.num_nodes()).to(cpu_device if args.sample == "cpu" else gpu_device)
    target_idx = node_ids[g.ndata[dgl.NTYPE] == category_id]
    g.ndata['ntype'] = g.ndata.pop(dgl.NTYPE)
    g.ndata['type_id'] = g.ndata.pop(dgl.NID)

    # find the mapping (inv_target) from global nodes IDs to type-specific node IDs
    inv_target = torch.empty((g.num_nodes(),), dtype=torch.int64).to(gpu_device)
    inv_target[target_idx] = torch.arange(0, target_idx.shape[0], dtype=inv_target.dtype).to(gpu_device)

    # create RGCN model
    num_nodes = g.num_nodes()
    out_size = data.num_classes
    model = RGCN(num_nodes, args.indim, args.hdim, out_size, gpu_device, num_rels, args.conv).to(gpu_device)
    train(args, gpu_device, g, target_idx, labels, train_mask, model, inv_target)

    # test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    # test_sampler = MultiLayerNeighborSampler([-1, -1])
    # test_loader = DataLoader(g, target_idx[test_idx], test_sampler, device=gpu_device,
    #                         batch_size=32, shuffle=False)
    # acc = evaluate(model, labels, test_loader, inv_target)
    # print("Test accuracy {:.4f}".format(acc))