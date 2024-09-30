# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的神经网络作为编码器
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# 定义BPR损失
class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()

    def forward(self, anchor, pos, neg):
        pos_score = torch.sum(anchor * pos, dim=-1)
        neg_score = torch.sum(anchor * neg, dim=-1)
        loss = self.softplus(neg_score - pos_score)
        return torch.mean(loss)


if __name__ == '__main__':

    # 参数设置
    input_dim = 10
    hidden_dim = 50
    output_dim = 10
    batch_size = 32
    learning_rate = 0.001
    epochs = 100

    # 模型、损失函数和优化器
    model = SimpleEncoder(input_dim, hidden_dim, output_dim)
    bpr_loss = BPRLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 模拟训练数据
    def generate_data(batch_size, input_dim):
        return torch.randn(batch_size, input_dim), torch.randn(batch_size, input_dim), torch.randn(batch_size, input_dim)

    # 训练循环
    for epoch in range(epochs):
        anchor, pos, neg = generate_data(batch_size, input_dim)  # 生成数据
        anchor_enc, pos_enc, neg_enc = model(anchor), model(pos), model(neg)
        loss = bpr_loss(anchor_enc, pos_enc, neg_enc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
