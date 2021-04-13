import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx
import unittest, pytest
from dgl import DGLError
import test_utils
from test_utils import parametrize_dtype, get_cases
from scipy.sparse import rand

def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph1(idtype):
    edges = []
    edges.extend([(0, 1), (1, 2)])  # follows
    edges.extend([(0, 3), (1, 3), (2, 4), (1, 4)])  # plays
    edges.extend([(0, 4), (2, 3)])  # wishes
    edges.extend([(5, 3), (6, 4)])  # develops
    edges = tuple(zip(*edges))
    ntypes = F.tensor([0, 0, 0, 1, 1, 2, 2])
    etypes = F.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    g0 = dgl.graph(edges, idtype=idtype, device=F.ctx())
    g0.ndata[dgl.NTYPE] = ntypes
    g0.edata[dgl.ETYPE] = etypes
    return dgl.to_heterogeneous(g0, ['user', 'game', 'developer'],
                                ['follows', 'plays', 'wishes', 'develops'])

def create_test_heterograph2(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
        }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph3(idtype):
    g = dgl.heterograph({
        ('user', 'plays', 'game'): (F.tensor([0, 1, 1, 2], dtype=idtype),
                                    F.tensor([0, 0, 1, 1], dtype=idtype)),
        ('developer', 'develops', 'game'): (F.tensor([0, 1], dtype=idtype),
                                            F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())

    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.nodes['developer'].data['h'] = F.copy_to(F.tensor([3, 3], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 1, 1, 1], dtype=idtype), ctx=F.ctx())
    return g

def create_test_heterograph4(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([0, 1, 1, 2, 2, 2], dtype=idtype),
                                      F.tensor([0, 0, 1, 1, 2, 2], dtype=idtype)),
        ('user', 'plays', 'game'): (F.tensor([0, 1], dtype=idtype),
                                    F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.edges['follows'].data['h'] = F.copy_to(F.tensor([1, 2, 3, 4, 5, 6], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    return g

def create_test_heterograph5(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([1, 2], dtype=idtype),
                                      F.tensor([0, 1], dtype=idtype)),
        ('user', 'plays', 'game'): (F.tensor([0, 1], dtype=idtype),
                                    F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.edges['follows'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    return g

def get_redfn(name):
    return getattr(F, name)

@parametrize_dtype
def test_level2(idtype):
    #edges = {
    #    'follows': ([0, 1], [1, 2]),
    #    'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
    #    'wishes': ([0, 2], [1, 0]),
    #    'develops': ([0, 1], [0, 1]),
    #}
    g = create_test_heterograph(idtype)
    def rfunc(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def rfunc2(nodes):
        return {'y': F.max(nodes.mailbox['m'], 1)}
    def mfunc(edges):
        return {'m': edges.src['h']}
    def afunc(nodes):
        return {'y' : nodes.data['y'] + 1}

    #############################################################
    #  update_all
    #############################################################
    g.nodes['user'].data['h'] = F.ones((3, 2))

    # g['plays'].update_all_new(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    g.update_all_new(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    
    # g.update_all(mfunc, rfunc, etype='plays')
    # y = g.nodes['game'].data['y']
    # assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))


    # for (src_type, rel_type, dst_type) in g.canonical_etypes:
    #     g.nodes[src_type].data[‘h’] = F.ones((3, 2))
    #     g.update_all(mfunc, rfunc, etype=rel_type)
    #     y = g.nodes[dst_type].data[‘y’]
    

    # only one type
    # g['plays'].update_all(mfunc, rfunc)
    # y = g.nodes['game'].data['y']
    # assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))

    # test fail case
    # fail due to multiple types

    # with pytest.raises(DGLError):
    #     g.update_all(mfunc, rfunc)

    # def multi_update_all(self, etype_dict, cross_reducer, apply_node_func=None):
    # test multi
    # g.multi_update_all(
    #     {'plays' : (mfunc, rfunc),
    #      ('user', 'wishes', 'game'): (mfunc, rfunc2)},
    #     'sum')
    # assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))
       
    # >>> g.multi_update_all(
    # ...     {'follows': (fn.copy_src('h', 'm'), fn.sum('m', 'h')),
    # ...      'attracts': (fn.copy_src('h', 'm'), fn.sum('m', 'h'))},
    # ... "sum")
    # 

    # will require modifying invoke_sddmm
    # g.update_all_new(fn.u_mul_v('h', 'h', 'm'), fn.sum('m', 'h')) 


    # g.multi_update_all(
    #     {'plays' : (mfunc, rfunc, afunc),
    #      ('user', 'wishes', 'game'): (mfunc, rfunc2)},
    #     'sum', afunc)
    # assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    # g.nodes['user'].data['h'] = F.randn((3, 2))
    # for cred in ['sum', 'max', 'min', 'mean', 'stack']:
    #     g.multi_update_all(
    #         {'plays' : (mfunc, rfunc, afunc),
    #          'wishes': (mfunc, rfunc2)},
    #         cred, afunc)
    #     y = g.nodes['game'].data['y']
    #     g['plays'].update_all(mfunc, rfunc, afunc)
    #     y1 = g.nodes['game'].data['y']
    #     g['wishes'].update_all(mfunc, rfunc2)
    #     y2 = g.nodes['game'].data['y']
    #     if cred == 'stack':
    #         # stack has an internal order by edge type id
    #         yy = F.stack([y1, y2], 1)
    #         yy = yy + 1  # final afunc
    #         assert F.array_equal(y, yy)
    #     else:
    #         yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
    #         yy = yy + 1  # final afunc
    #         assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    # with pytest.raises(DGLError):
    #     g.update_all(
    #         {'plays' : (mfunc, rfunc),
    #          'follows': (mfunc, rfunc2)},
    #         'sum')

    g.nodes['game'].data.clear()

# @parametrize_dtype
# def test_updates(idtype):
#     def msg_func(edges):
#         return {'m': edges.src['h']}
#     def reduce_func(nodes):
#         return {'y': F.sum(nodes.mailbox['m'], 1)}
#     def apply_func(nodes):
#         return {'y': nodes.data['y'] * 2}
#     g = create_test_heterograph(idtype)
#     x = F.randn((3, 5))
#     g.nodes['user'].data['h'] = x

#     for msg, red, apply in itertools.product(
#             [fn.copy_u('h', 'm'), msg_func], [fn.sum('m', 'y'), reduce_func],
#             [None, apply_func]):
#         multiplier = 1 if apply is None else 2

#         g['user', 'plays', 'game'].update_all(msg, red, apply)
#         y = g.nodes['game'].data['y']
#         assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
#         assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
#         del g.nodes['game'].data['y']

#         g['user', 'plays', 'game'].send_and_recv(([0, 1, 2], [0, 1, 1]), msg, red, apply)
#         y = g.nodes['game'].data['y']
#         assert F.array_equal(y[0], x[0] * multiplier)
#         assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
#         del g.nodes['game'].data['y']

#         # pulls from destination (game) node 0
#         g['user', 'plays', 'game'].pull(0, msg, red, apply)
#         y = g.nodes['game'].data['y']
#         assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
#         del g.nodes['game'].data['y']

#         # pushes from source (user) node 0
#         g['user', 'plays', 'game'].push(0, msg, red, apply)
#         y = g.nodes['game'].data['y']
#         assert F.array_equal(y[0], x[0] * multiplier)
#         del g.nodes['game'].data['y']


# @parametrize_dtype
# def test_backward(idtype):
#     g = create_test_heterograph(idtype)
#     x = F.randn((3, 5))
#     F.attach_grad(x)
#     g.nodes['user'].data['h'] = x
#     with F.record_grad():
#         g.multi_update_all(
#             {'plays' : (fn.copy_u('h', 'm'), fn.sum('m', 'y')),
#              'wishes': (fn.copy_u('h', 'm'), fn.sum('m', 'y'))},
#             'sum')
#         y = g.nodes['game'].data['y']
#         F.backward(y, F.ones(y.shape))
#     print(F.grad(x))
#     assert F.array_equal(F.grad(x), F.tensor([[2., 2., 2., 2., 2.],
#                                               [2., 2., 2., 2., 2.],
#                                               [2., 2., 2., 2., 2.]]))


# @parametrize_dtype
# def test_empty_heterograph(idtype):
#     def assert_empty(g):
#         assert g.number_of_nodes('user') == 0
#         assert g.number_of_edges('plays') == 0
#         assert g.number_of_nodes('game') == 0

#     # empty src-dst pair
#     assert_empty(dgl.heterograph({('user', 'plays', 'game'): ([], [])}))

#     g = dgl.heterograph({('user', 'follows', 'user'): ([], [])}, idtype=idtype, device=F.ctx())
#     assert g.idtype == idtype
#     assert g.device == F.ctx()
#     assert g.number_of_nodes('user') == 0
#     assert g.number_of_edges('follows') == 0

#     # empty relation graph with others
#     g = dgl.heterograph({('user', 'plays', 'game'): ([], []), ('developer', 'develops', 'game'):
#         ([0, 1], [0, 1])}, idtype=idtype, device=F.ctx())
#     assert g.idtype == idtype
#     assert g.device == F.ctx()
#     assert g.number_of_nodes('user') == 0
#     assert g.number_of_edges('plays') == 0
#     assert g.number_of_nodes('game') == 2
#     assert g.number_of_edges('develops') == 2
#     assert g.number_of_nodes('developer') == 2

# @parametrize_dtype
# def test_types_in_function(idtype):
#     def mfunc1(edges):
#         assert edges.canonical_etype == ('user', 'follow', 'user')
#         return {}

#     def rfunc1(nodes):
#         assert nodes.ntype == 'user'
#         return {}

#     def filter_nodes1(nodes):
#         assert nodes.ntype == 'user'
#         return F.zeros((3,))

#     def filter_edges1(edges):
#         assert edges.canonical_etype == ('user', 'follow', 'user')
#         return F.zeros((2,))

#     def mfunc2(edges):
#         assert edges.canonical_etype == ('user', 'plays', 'game')
#         return {}

#     def rfunc2(nodes):
#         assert nodes.ntype == 'game'
#         return {}

#     def filter_nodes2(nodes):
#         assert nodes.ntype == 'game'
#         return F.zeros((3,))

#     def filter_edges2(edges):
#         assert edges.canonical_etype == ('user', 'plays', 'game')
#         return F.zeros((2,))

#     g = dgl.heterograph({('user', 'follow', 'user'): ((0, 1), (1, 2))},
#                         idtype=idtype, device=F.ctx())
#     g.apply_nodes(rfunc1)
#     g.apply_edges(mfunc1)
#     g.update_all(mfunc1, rfunc1)
#     g.send_and_recv([0, 1], mfunc1, rfunc1)
#     g.push([0], mfunc1, rfunc1)
#     g.pull([1], mfunc1, rfunc1)
#     g.filter_nodes(filter_nodes1)
#     g.filter_edges(filter_edges1)

#     g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
#     g.apply_nodes(rfunc2, ntype='game')
#     g.apply_edges(mfunc2)
#     g.update_all(mfunc2, rfunc2)
#     g.send_and_recv([0, 1], mfunc2, rfunc2)
#     g.push([0], mfunc2, rfunc2)
#     g.pull([1], mfunc2, rfunc2)
#     g.filter_nodes(filter_nodes2, ntype='game')
#     g.filter_edges(filter_edges2)

if __name__ == '__main__':
    # test_create()
    # test_query()
    # test_hypersparse()
    # test_adj("int32")
    # test_inc()
    # test_view("int32")
    # test_view1("int32")
    # test_flatten(F.int32)
    # test_convert_bound()
    # test_convert()
    # test_to_device("int32")
    # test_transform("int32")
    # test_subgraph("int32")
    # test_subgraph_mask("int32")
    # test_apply()
    # test_level1()
    # test_level2()
    # test_updates()
    # test_backward()
    # test_empty_heterograph('int32')
    # test_types_in_function()
    # test_stack_reduce()
    # test_isolated_ntype()
    # test_bipartite()
    # test_dtype_cast()
    # test_reverse("int32")
    # test_format()
    #test_add_edges(F.int32)
    #test_add_nodes(F.int32)
    #test_remove_edges(F.int32)
    #test_remove_nodes(F.int32)
    #test_clone(F.int32)
    #test_frame(F.int32)
    #test_frame_device(F.int32)
    #test_empty_query(F.int32)
    #test_create_block(F.int32)
    pass
