# coding : utf-8
# Author : yuxiang Zeng
import torch
import numpy as np

class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = torch.nn.Parameter(torch.empty(size=(in_features, out_features)))
        torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = torch.nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = torch.nn.functional.softmax(attention, dim=1)
        attention = torch.nn.functional.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return torch.nn.functional.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


def generate_random_adjacency_matrix(num_nodes, p=0.5):
    """
    生成随机的无向图邻接矩阵。
    :param num_nodes: 图中的节点数
    :param p: 任意两个节点之间存在边的概率
    :return: 随机生成的邻接矩阵
    """
    # 生成上三角矩阵
    upper_triangle = np.random.rand(num_nodes, num_nodes) < p
    # 保证对角线为0
    np.fill_diagonal(upper_triangle, 0)
    # 创建对称的邻接矩阵
    adjacency_matrix = upper_triangle + upper_triangle.T
    # 注意归一化邻接矩阵
    adjacency_matrix = adjacency_matrix.astype(int)
    return torch.as_tensor(adjacency_matrix)


if __name__ == '__main__':
    num_nodes, dim = 100, 128
    graph = generate_random_adjacency_matrix(num_nodes)
    features = torch.randn(num_nodes, dim)
    graph_gat = GraphAttentionLayer(dim, dim, 0, 0.2, True)
    embeds = graph_gat(features, graph)
    print(embeds.shape)
