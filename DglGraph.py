# coding : utf-8
# Author : yuxiang Zeng
import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as dglnn


# 定义各个图神经网络模型
# 最普通的GCN
class GCNModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNModel, self).__init__()
        self.conv = dglnn.GraphConv(in_feats, out_feats)

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        return h


# 图注意力
class GATModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATModel, self).__init__()
        self.conv = dglnn.GATConv(in_feats, out_feats, num_heads=1)

    def forward(self, g, inputs):
        h = self.conv(g, inputs).squeeze(1)
        return h


# 学习图对表示
class GINModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GINModel, self).__init__()
        self.conv = dglnn.GINConv(nn.Linear(in_feats, out_feats), 'max')

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        return h


class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphSAGEModel, self).__init__()
        self.conv = dglnn.SAGEConv(in_feats, out_feats, 'gcn')

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        return h


# 拓扑自适应
class TAGModel(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TAGModel, self).__init__()
        self.conv = dglnn.TAGConv(in_feats, out_feats)

    def forward(self, g, inputs):
        h = self.conv(g, inputs)
        return h


class EdgeGATModel(nn.Module):
    def __init__(self, in_feats, edge_feats, out_feats):
        super(EdgeGATModel, self).__init__()
        self.conv = dglnn.EdgeGATConv(in_feats, edge_feats, out_feats, num_heads=1)

    def forward(self, g, node_features, edge_features):
        h = self.conv(g, node_features, edge_features).squeeze(1)
        return h


if __name__ == '__main__':
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)

    in_feats, out_feats = 64, 64

    # 测试不同的图神经网络模型
    models = {
        'GCNModel': GCNModel(in_feats, out_feats),
        'GATModel': GATModel(in_feats, out_feats),
        'GINModel': GINModel(in_feats, out_feats),
        'GraphSAGEModel': GraphSAGEModel(in_feats, out_feats),
        'TAGModel': TAGModel(in_feats, out_feats),
    }

    features = torch.rand(100, in_feats)  # 假设每个节点有5个特征

    for model_name, model in models.items():
        outputs = model(graph, features)
        print(f"{model_name:15s} output shape: {outputs.shape}")

    model_name = 'EdgeGATModel'
    node_features = torch.rand(100, in_feats)
    edge_features = torch.rand(300, in_feats)
    model = EdgeGATModel(64, 64, 64)
    outputs = model(graph, node_features, edge_features)
    print(f"{model_name:15s} output shape: {outputs.shape}")
