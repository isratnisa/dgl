"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import time
import tqdm
from timeit import default_timer

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset

from collections import defaultdict


class Timer:
    def __init__(self, device):
        self.timer = default_timer
        self.device = device

    def __enter__(self):
        if str(self.device).startswith('cuda'):
            self.start_event = th.cuda.Event(enable_timing=True)
            self.end_event = th.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = self.timer()
        return self

    def __exit__(self, type, value, traceback):
        if str(self.device).startswith('cuda'):
            self.end_event.record()
            th.cuda.synchronize()  # Wait for the events to be recorded!
            self.elapsed_secs = self.start_event.elapsed_time(
                self.end_event) / 1e3
        else:
            self.elapsed_secs = self.timer() - self.tic

class RelGraphConvLayerHeteroAPI(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=False,
        activation=None,
        self_loop=False,
        dropout=0.0
    ):
        super(RelGraphConvLayerHeteroAPI, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    th.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(th.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        inputs_src = inputs_dst = inputs

        for srctype, _, _ in g.canonical_etypes:
            g.nodes[srctype].data["h"] = inputs[srctype]

        if self.use_weight:
            g.apply_edges(fn.copy_u("h", "m"))
            m = g.edata["m"]
            for rel in g.canonical_etypes:
                _, etype, _ = rel
                g.edges[rel].data["h*w_r"] = th.matmul(
                    m[rel], wdict[etype]["weight"]
                )
        else:
            g.apply_edges(fn.copy_u("h", "h*w_r"))

        g.update_all(fn.copy_e("h*w_r", "m"), fn.sum("m", "h"))

        def _apply(ntype):
            h = g.nodes[ntype].data["h"]
            if self.self_loop:
                h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return h

        return {ntype: _apply(ntype) for ntype in g.dsttypes}


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(
        self, g, embed_size, embed_name="embed", activation=None, dropout=0.0
    ):
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            embed = nn.Parameter(
                th.Tensor(g.number_of_nodes(ntype), self.embed_size)
            )
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
            self.embeds[ntype] = embed

    def forward(self, block=None):
        """Forward computation
        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.
        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        return self.embeds

class EntityClassify_HeteroAPI(nn.Module):
    def __init__(
        self,
        g,
        in_dim,
        h_dim,
        out_dim,
        num_bases,
        num_hidden_layers=1,
        dropout=0,
        use_self_loop=False,
    ):
        super(EntityClassify_HeteroAPI, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.embed_layer = RelGraphEmbed(g, self.in_dim)
        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayerHeteroAPI(
                self.in_dim,
                self.h_dim,
                self.rel_names,
                self.num_bases,
                activation=F.relu,
                self_loop=self.use_self_loop,
                dropout=self.dropout,
            )
        )
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(
                RelGraphConvLayerHeteroAPI(
                    self.h_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    activation=F.relu,
                    self_loop=self.use_self_loop,
                    dropout=self.dropout,
                )
            )
        # h2o
        self.layers.append(
            RelGraphConvLayerHeteroAPI(
                self.h_dim,
                self.out_dim,
                self.rel_names,
                self.num_bases,
                activation=None,
                self_loop=self.use_self_loop,
            )
        )

    def forward(self, h=None, blocks=None):
        if h is None:
            # full graph training
            h = self.embed_layer()
        if blocks is None:
            # full graph training
            for layer in self.layers:
                h = layer(self.g, h)
        else:
            # minibatch training
            for layer, block in zip(self.layers, blocks):
                h = layer(block, h)
        return h


def main(args):
    # load graph data
    if args.dataset == "aifb":
        dataset = AIFBDataset()
    elif args.dataset == "mutag":
        dataset = MUTAGDataset()
    elif args.dataset == "bgs":
        dataset = BGSDataset()
    elif args.dataset == "am":
        dataset = AMDataset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop("train_mask")
    test_mask = g.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop("labels")
    category_id = g.ntypes.index(category)

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    device = th.device('cuda' if use_cuda else 'cpu')
    print(device)
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to("cuda:%d" % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = EntityClassify_HeteroAPI(
        g,
        args.in_dim,
        args.n_hidden,
        num_classes,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers - 2,
        dropout=args.dropout,
        use_self_loop=args.use_self_loop,
    )

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(
        model.parameters(), lr=1e-2, weight_decay=5e-4
    )

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        with Timer(device) as t:
            optimizer.zero_grad()
            logits = model()[category]
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

        if epoch > 10:
            dur.append(t.elapsed_secs)
        print(
            "Epoch {:05d} | Time: {:.4f} ms".format(
                epoch,
                (t.elapsed_secs) * 1000,
            )
        )
    print("Average e2e 2-layers training time {:.4f} ms".format(np.mean(dur) * 1000))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--dropout", type=float, default=0, help="dropout probability"
    )
    parser.add_argument("--in-dim", type=int, default=16)
    parser.add_argument(
        "--n-hidden", type=int, default=16, help="number of hidden units"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument(
        "--n-bases",
        type=int,
        default=-1,
        help="number of filter weight matrices, default: -1 [use all]",
    )
    parser.add_argument(
        "--n-layers", type=int, default=2, help="number of propagation rounds"
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=50,
        help="number of training epochs",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="dataset to use"
    )

    parser.add_argument(
        "--use-self-loop",
        default=False,
        action="store_true",
        help="include self feature as a special relation",
    )

    args = parser.parse_args()
    print(args)
    main(args)