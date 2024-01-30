# coding : utf-8
# Author : yuxiang Zeng
import torch


class Inner_product(torch.nn.Module):
    def __init__(self):
        super(Inner_product, self).__init__()
        pass

    def forward(self, a, b):
        outputs = a * b
        outputs = outputs.sum(dim=-1)
        return outputs


class Outer_product(torch.nn.Module):
    def __init__(self):
        super(Outer_product, self).__init__()

    def forward(self, a, b):
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("外积的输入必须是二维张量")
        outputs = torch.einsum('bi,bj->bij', a, b)  # 计算外积
        return outputs


class DNN_interaction(torch.nn.Module):
    def __init__(self, dim):
        super(DNN_interaction, self).__init__()
        self.input_dim = dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, 1)  # y
        )

    def forward(self, x):
        outputs = self.NeuCF(x)
        outputs = torch.sigmoid(outputs)
        return outputs.flatten()


class Light_dnn(torch.nn.Module):
    def __init__(self, dim):
        super(Light_dnn, self).__init__()
        self.dim = dim
        self.transfer_a = torch.nn.Linear(self.dim, self.dim)
        self.transfer_b = torch.nn.Linear(self.dim, self.dim)

    def forward(self, a, b):
        a = self.transfer_a(a)
        b = self.transfer_b(b)
        outputs = a * b
        outputs = outputs.sum(dim=-1)
        return outputs


if __name__ == '__main__':
    bs, dim = 64, 128
    inputs_1 = torch.randn(bs, dim)
    inputs_2 = torch.randn(bs, dim)

    inner_product = Inner_product()
    outputs = inner_product(inputs_1, inputs_2)
    print("内积输出形状:", outputs.shape)

    outer_product = Outer_product()
    outputs = outer_product(inputs_1, inputs_2)
    print("外积输出形状:", outputs.shape)

    dnn_interaction = DNN_interaction(inputs_1.shape[1])
    outputs = dnn_interaction(inputs_1)
    print("DNN输出形状:", outputs.shape)

    light_dnn = Light_dnn(inputs_1.shape[1])
    outputs = light_dnn(inputs_1, inputs_2)
    print("light_dnn输出形状:", outputs.shape)
