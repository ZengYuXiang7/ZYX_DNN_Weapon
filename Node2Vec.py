# coding : utf-8
# Author : yuxiang Zeng
import dgl
import torch
import numpy as np
from node2vec import Node2Vec


def get_graph_embedding(graph, dim=32, length=10, walk=20, windows=3, epochs=20, bs=32, random_state=2024):
    # p=1, q=0.5, n_clusters=6。DFS深度优先搜索，挖掘同质社群
    # p=1, q=2,   n_clusters=3。BFS宽度优先搜索，挖掘节点的结构功能。

    graph = graph.to_networkx()

    node2vec = Node2Vec(
        graph,
        dimensions=dim,      # 嵌入维度
        p=1,                 # 回家参数
        q=0.5,               # 外出参数
        walk_length=length,  # 随机游走最大长度
        num_walks=walk,      # 每个节点作为起始节点生成的随机游走个数
        # workers=1,         # 并行线程数
    )

    # 训练Node2Vec
    model = node2vec.fit(
        window=windows,  # Skip-Gram窗口大小
        epochs=epochs,
        min_count=3,     # 忽略出现次数低于此阈值的节点（词）
        batch_words=bs,  # 每个线程处理的数据量
        seed=random_state
    )
    embedding = model.wv.vectors
    embedding = np.array(embedding)
    return embedding


if __name__ == '__main__':
    # Build a random graph
    num_nodes, num_edges = 10000, 2000
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    graph = dgl.graph((src_nodes, dst_nodes))
    dgl.add_self_loop(graph)

    # Execute the Node2Vec
    embedding = get_graph_embedding(graph)
    print(f'Node2Vec for graph {embedding.shape}')
