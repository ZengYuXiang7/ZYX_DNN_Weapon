# coding : utf-8
# Author : yuxiang Zeng
import dgl
import torch

from utils.config import get_config


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, rank, order, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        # Graph Encoder
        if args.graph_encoder == 'gcn':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(rank, rank) for _ in range(order)])
        elif args.graph_encoder == 'sage':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(rank, rank, aggregator_type='gcn') for _ in range(order)])
        elif args.graph_encoder == 'gat':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GATConv(rank, rank, 8, 0.10) for _ in range(order)])
        else:
            raise NotImplementedError
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)

    def forward(self, graph, features):
        g = graph
        feats = features
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            feats = feats.mean(dim=1) if self.args.graph_encoder == 'gat' else feats
            feats = norm(feats)
            feats = act(feats)
            feats = self.dropout(feats) if self.args.graph_encoder != 'gat' else feats
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)
        first_node_features = feats[first_nodes_idx]
        return first_node_features


# 添加全局节点在0号位置
def add_global_nodes(graph):
    global_node_id = 0                                     # 将全局节点设置为节点0
    new_graph = dgl.add_nodes(graph, 1)               # 增加1个新节点
    all_node_ids = torch.arange(1, new_graph.num_nodes())  # 新节点后所有其他节点
    new_graph.add_edges(global_node_id, all_node_ids)      # 全局节点 -> 其他节点
    new_graph.add_edges(all_node_ids, global_node_id)      # 其他节点 -> 全局节点
    return new_graph


def graph_encoder(encoder_type, rank, order):
    args.graph_encoder = encoder_type
    u, v = torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])
    input_dim = rank
    graph = dgl.graph((u, v), num_nodes=4)
    graph = add_global_nodes(graph)
    graph = dgl.to_bidirected(graph)
    features = torch.randn(graph.num_nodes(), input_dim)

    encoder = GraphEncoder(input_dim, rank, order, args).to(args.device)

    features = features.to(args.device)
    graph_features = encoder(graph, features)

    print(f"Test for {encoder_type} encoder:")
    print("Graph features shape:", graph_features.shape)
    print("-" * 50)


if __name__ == '__main__':
    args = get_config()
    args.rank = 32
    args.order = 4

    graph_encoder(encoder_type='gcn', rank=args.rank, order=args.order)

    graph_encoder(encoder_type='sage', rank=args.rank, order=args.order)

    graph_encoder(encoder_type='gat', rank=args.rank, order=args.order)
