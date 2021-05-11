import time
import dgl
import torch
import numpy as np
import dgl.function as fn

from .. import utils


@utils.benchmark('time', timeout=6000)
@utils.parametrize('feat_size', [128])
@utils.parametrize('num_relations', [10]) #[5, 50, 500])
@utils.parametrize('multi_reduce_type', ["sum"])
def track_time(feat_size, num_relations, multi_reduce_type):
    device = utils.get_bench_device()
    dd = {}
    candidate_edges = [dgl.data.CoraGraphDataset(verbose=False)[0].edges(), dgl.data.PubmedGraphDataset(verbose=False)[
        0].edges(), dgl.data.CiteseerGraphDataset(verbose=False)[0].edges()]
    for i in range(num_relations):
        dd[('n1', 'e_{}'.format(i), 'n2')] = candidate_edges[i %
                                                             len(candidate_edges)]
    graph = dgl.heterograph(dd)
    print(graph)
    graph = graph.to(device)
    graph.nodes['n1'].data['h'] = torch.randn(
        (graph.num_nodes('n1'), feat_size), device=device)
    graph.nodes['n2'].data['h'] = torch.randn(
        (graph.num_nodes('n2'), feat_size), device=device)
    
    # # dry run
    # update_dict = {}
    # for i in range(num_relations):
    #     update_dict['e_{}'.format(i)] = (
    #         fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    # print(update_dict)
    # graph.multi_update_all(
    #     update_dict,
    #     multi_reduce_type)

    # timing
    iters = 1
    with utils.Timer(iters) as t:
        for i in range(iters):
            # graph.update_all_new(fn.copy_u('h', 'm'), fn.sum('m', 'y'))
            graph.update_all_new(fn.u_mul_v('h', 'h', 'm'), fn.sum('m', 'y'))
    
    return t.elapsed_secs / iters
