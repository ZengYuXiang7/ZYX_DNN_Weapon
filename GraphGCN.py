# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, dim, order=2):
        super(GraphSAGEConv, self).__init__()
        self.order = order
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
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
    graph = dgl.graph((src_nodes, dst_nodes))

    # Demo test
    dim = 128
    features = torch.randn(num_nodes, dim)  # 假设有3个节点，每个节点有3维特征
    model = GraphSAGEConv(dim, order=2)
    embeds = model(graph, features)
    print(embeds.shape)
