# coding : utf-8
# Author : yuxiang Zeng

import torch


class FactorizationMachine(torch.nn.Module):
    def __init__(self, dim, k):
        """
        dim: 特征的数量
        k: 隐向量的维度
        """
        super(FactorizationMachine, self).__init__()
        self.w0 = torch.nn.Parameter(torch.zeros(1))
        self.w = torch.nn.Parameter(torch.zeros(dim))
        self.v = torch.nn.Parameter(torch.randn(dim, k))

    def forward(self, x):
        linear_part = self.w0 + torch.matmul(x, self.w).unsqueeze(1)

        # 计算交互部分
        sum_of_vx = torch.matmul(x, self.v)
        square_of_sum_vx = torch.sum(sum_of_vx ** 2, dim=1, keepdim=True)
        sum_of_square_vx = torch.sum(torch.matmul(x ** 2, self.v ** 2), dim=1, keepdim=True)
        interaction_part = 0.5 * (square_of_sum_vx - sum_of_square_vx)

        output = linear_part + interaction_part
        return output


if __name__ == '__main__':
    bs, dim = 512, 128
    inputs = torch.rand(bs, dim)
    fm = FactorizationMachine(dim, dim)
    outputs = fm(inputs)
    print(outputs.shape)
