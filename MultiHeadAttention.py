# coding : utf-8
# Author : yuxiang Zeng
# Description : Pytorch Attention API
import torch

if __name__ == '__main__':

    bs = 256  # 批次大小
    n = 4     # 模态数量
    dim = 32  # 特征维度

    heads_num = 4
    multihead_att = torch.nn.MultiheadAttention(embed_dim=dim, num_heads=heads_num, dropout=0.10)

    # 模拟的多模态输入特征 [n, bs, dim]
    features = torch.randn(n, bs, dim)

    # 应用多头注意力机制，这里features同时作为query, key和value
    outputs, weights = multihead_att(features, features, features)

    # 现在包含了经过融合的多模态信息
    print('Attention:', outputs.shape, weights.shape)

    # 聚合操作
    outputs = torch.mean(outputs, dim=0)
    print('聚合操作', outputs.shape)


