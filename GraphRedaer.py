# coding : utf-8
# Author : yuxiang Zeng
import dgl
import torch

class GraphReader(torch.nn.Module):
    def __init__(self, input_dim, rank, order, args):
        super(GraphReader, self).__init__()
        self.args = args
        self.rank = rank
        self.order = order
        self.dnn_embedding = torch.nn.Embedding(6, rank)
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(rank, rank, aggregator_type='gcn') for i in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)

    def forward(self, graph, features):
        g, feats = graph, self.dnn_embedding(features).reshape(features.shape[0] * 9, -1)
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            feats = norm(feats)
            feats = act(feats)
            feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        first_node_features = feats[first_nodes_idx]
        return first_node_features
