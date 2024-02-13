# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, dim, order=2):
        super(GraphSAGEConv, self).__init__()
        self.order = order
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, graph, features):
        g, g.ndata['L0'] = graph, features
        feats = g.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats
        embeds = g.ndata[f'L{self.order}']
        return embeds


if __name__ == '__main__':
    # Build a random graph
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    print(src_nodes.shape, dst_nodes.shape)

    graph = dgl.graph((src_nodes, dst_nodes))
    dgl.add_self_loop(graph)

    # Demo test
    dim = 128
    bs = 32
    features = torch.randn(bs, num_nodes, dim)
    graph_gcn = GraphSAGEConv(dim, order=2)
    embeds = graph_gcn(graph, features)
    print(embeds.shape)
