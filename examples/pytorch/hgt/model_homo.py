import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout = 0.2,
                 use_norm = False):
        super(HGTLayer, self).__init__()

        self.in_dim        = in_dim
        self.out_dim       = out_dim
        self.node_dict     = node_dict
        self.edge_dict     = edge_dict
        self.num_types     = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel     = self.num_types * self.num_relations * self.num_types
        self.n_heads       = n_heads
        self.d_k           = out_dim // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.att           = None

        self.k_linears   = nn.ModuleList()
        self.q_linears   = nn.ModuleList()
        self.v_linears   = nn.ModuleList()
        self.a_linears   = nn.ModuleList()
        self.norms       = nn.ModuleList()
        self.use_norm    = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim,   out_dim))
            self.q_linears.append(nn.Linear(in_dim,   out_dim))
            self.v_linears.append(nn.Linear(in_dim,   out_dim))
            self.a_linears.append(nn.Linear(out_dim,  out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip           = nn.Parameter(torch.ones(self.num_types))
        self.drop           = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        gidx=G._graph

        srctype, dsttype = gidx.metagraph.find_edge(0)
        node_dict, edge_dict = self.node_dict, self.edge_dict

        k_linear = self.k_linears[srctype]
        v_linear = self.v_linears[srctype]
        q_linear = self.q_linears[dsttype]
        # relation_att = self.relation_att[None]
        k = k_linear(h[None]).view(-1, self.n_heads, self.d_k)
        v = v_linear(h[None]).view(-1, self.n_heads, self.d_k)
        q = q_linear(h[None]).view(-1, self.n_heads, self.d_k)

        e_id = self.edge_dict[None]

        relation_att = self.relation_att[e_id]
        relation_pri = self.relation_pri[e_id]
        relation_msg = self.relation_msg[e_id]

        k = torch.einsum("bij,ijk->bik", k, relation_att)
        v = torch.einsum("bij,ijk->bik", v, relation_msg)

        G.srcdata['k'] = k
        G.dstdata['q'] = q
        G.srcdata['v_%d' % e_id] = v

        G.apply_edges(fn.v_dot_u('q', 'k', 't'))
        attn_score = G.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
        attn_score = edge_softmax(G, attn_score, norm_by='dst')

        G.edata['t'] = attn_score.unsqueeze(-1)

        # HG.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
        #                     for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')

        G.update_all(fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't'))
        print("Add corss reducer MEAN??")
        new_h = {}

        n_id = node_dict[None]
        alpha = torch.sigmoid(self.skip[n_id])
        t = G.ndata['t'].view(-1, self.out_dim)
        trans_out = self.drop(self.a_linears[n_id](t))
        trans_out = trans_out * alpha + h[None] * (1-alpha)
        if self.use_norm:
            new_h[None] = self.norms[n_id](trans_out)
        else:
            new_h[None] = trans_out
        return new_h[None]

class HGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm = True):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws  = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp,   n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm = use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            ntype = None
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h[None] = self.gcs[i](G, h)
        return self.out(h[None])

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        # self.weight = nn.ModuleDict({
        #         name : nn.Linear(in_size, out_size) for name in etypes
        #     })
        self.weight = nn.Linear(in_size, out_size)
        # self.weight = nn.Parameter(torch.Tensor(in_size, out_size))
    
    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        gidx = G._graph
        srctype, dsttype = gidx.metagraph.find_edge(0)
        etype = None
        # for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            # Wh = self.weight[etype](feat_dict[srctype])
        Wh = self.weight(feat_dict[None])
        # Save it in graph for message passing

        G.nodes[None].data['Wh'] = Wh # G.ndata['Wh'] = Wh
        # Specify per-relation message passing functions: (message_func, reduce_func).
        # Note that the results are saved to the same destination feature 'h', which
        # hints the type wise reducer for aggregation.
        # funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        # G.multi_update_all(funcs, 'sum')
        G.update_all(fn.copy_u('Wh', 'm'), fn.mean('m', 'h'))
        print("Add corss reducer SUM??")
        h = G.ndata.pop('h')
        return h
        
        # return the updated node feature dictionary
        # ntype = None
        # return G.ndata['h']
        # return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G, out_key):
        input_dict = {None : G.nodes[ntype].data['inp'] for ntype in G.ntypes}
        h_dict = self.layer1(G, input_dict)
        h_dict = F.leaky_relu(h_dict) #{k : F.leaky_relu(h) for k, h in h_dict.items()}
        # h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict #[out_key]
