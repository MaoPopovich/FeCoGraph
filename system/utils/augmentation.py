import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected, to_networkx
import networkx as nx
from torch_scatter import scatter
import os
import json

def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(edge_index, k=k)
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())

    return weights


def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())


def generate_views(params, data, device, drop_feature_rate, drop_edge_rate_1, drop_edge_rate_2):   
    # Topology-level augmentation
    if params['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif params['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif params['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)

    # drop edge based on drop weights
    edge_index_1 = None
    edge_index_2 = None
    if params['drop_scheme'] == 'uniform':
        # edge_index_1 = dropout_adj(data.edge_index, p=params['drop_edge_rate_1'])[0]
        # edge_index_2 = dropout_adj(data.edge_index, p=params['drop_edge_rate_2'])[0]
        edge_index_1 = dropout_adj(data.edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(data.edge_index, p=drop_edge_rate_2)[0]
    elif params['drop_scheme'] in ['degree', 'pr', 'evc']:
        # edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=params['drop_edge_rate_1'], threshold=0.7)
        # edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=params['drop_edge_rate_2'], threshold=0.7)
        edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate_1, threshold=0.7)
        edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edge_rate_2, threshold=0.7)
    
    # Attribute-level augmentation
    if params['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        # print(data.edge_index.shape)
        node_deg = degree(edge_index_[1], num_nodes=data.num_nodes)
        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        # if args.dataset == 'WikiCS':
        #     feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        # else:
        #     feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif params['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        # if args.dataset == 'WikiCS':
        #     feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        # else:
        #     feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif params['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        # if args.dataset == 'WikiCS':
        #     feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        # else:
        #     feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    
    # mask feature dimension based on feature weights
    x_1 = None
    x_2 = None
    if params['drop_scheme'] == 'uniform':
        # x_1 = drop_feature(data.x, params['drop_feature_rate_1'])
        # x_2 = drop_feature(data.x, params['drop_feature_rate_2'])
        x_1 = drop_feature(data.x, drop_feature_rate)
        x_2 = drop_feature(data.x, drop_feature_rate)
    elif params['drop_scheme'] in ['degree', 'pr', 'evc']:
        # x_1 = drop_feature_weighted(data.x, feature_weights, params['drop_feature_rate_1'])
        # x_2 = drop_feature_weighted(data.x, feature_weights, params['drop_feature_rate_2'])
        x_1 = drop_feature_weighted(data.x, feature_weights, drop_feature_rate)
        x_2 = drop_feature_weighted(data.x, feature_weights, drop_feature_rate)
    
    return x_1, x_2, edge_index_1, edge_index_2