# coding : utf-8
# Author : yuxiang Zeng
import torch
import numpy as np
import dgl
from dgl.nn.pytorch import GATConv

class GraphGATConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.1):
        super(GraphGATConv, self).__init__()
        self.num_heads = num_heads
        self.layer = GATConv(in_dim, out_dim, num_heads=num_heads, feat_drop=dropout)
        self.norm = torch.nn.LayerNorm(out_dim * num_heads)
        self.act = torch.nn.ELU()

    def forward(self, graph, features):
        g, feats = graph, features
        feats = self.layer(g, feats).view(feats.size(0), -1)
        feats = self.norm(feats)
        feats = self.act(feats)
        feats = feats.view(feats.size(0), self.num_heads, -1)
        feats = torch.mean(feats, dim = 1)
        return feats



if __name__ == '__main__':
    dim = 128
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)
    features = torch.randn(num_nodes, dim)
    graph_gat = GraphGATConv(dim, dim, 2, 0.10)
    embeds = graph_gat(graph, features)
    print(embeds.shape)
