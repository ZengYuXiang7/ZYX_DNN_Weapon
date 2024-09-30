# coding : utf-8
# Author : yuxiang Zeng
import torch


class InnerOroduct(torch.nn.Module):
    def __init__(self):
        super(InnerOroduct, self).__init__()
        pass

    def forward(self, a, b):
        outputs = a * b
        outputs = outputs.sum(dim=-1)
        return outputs


class OuterProduct(torch.nn.Module):
    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, a, b):
        if a.dim() != 2 or b.dim() != 2:
            raise ValueError("外积的输入必须是二维张量")
        outputs = torch.einsum('bi,bj->bij', a, b)  # 计算外积
        return outputs


class DNNInteraction(torch.nn.Module):
    def __init__(self, dim):
        super(DNNInteraction, self).__init__()
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


class LightDnn(torch.nn.Module):
    def __init__(self, dim):
        super(LightDnn, self).__init__()
        self.dim = dim
        self.transfer = torch.nn.Linear(self.dim, self.dim)

    def forward(self, a, b):
        outputs = a * b
        outputs = self.transfer(outputs)
        outputs = outputs.sum(dim=-1).sigmoid()
        return outputs


class CNNInteraction(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(CNNInteraction, self).__init__()
        # 定义卷积层
        self.layer_1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size)
        # 可以添加更多层和其他组件，如批量归一化、激活函数等

    def forward(self, x):
        # 通过卷积层
        x = self.layer_1(x)
        # 可以添加激活函数、池化层等
        return x

class ExternalAttention(torch.nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = torch.nn.Linear(d_model, S, bias=False)
        self.mv = torch.nn.Linear(S, d_model, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        queries = queries.to(torch.float32)
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        return out


class LightDnn(torch.nn.Module):
    def __init__(self, dim):
        super(LightDnn, self).__init__()
        self.dim = dim
        self.transfer = torch.nn.Linear(self.dim, 1)

    def forward(self, a, b):
        outputs = (a * b).sum(dim = -1).sigmoid()
        return outputs


class AttInteraction(torch.nn.Module):
    def __init__(self, dim):
        super(AttInteraction, self).__init__()
        self.dim = dim
        self.input_dim = dim * 2
        self.attention = ExternalAttention(self.dim, 32)
        self.transfer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, self.input_dim // 2),  # FFN
            torch.nn.LayerNorm(self.input_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.input_dim // 2, 1)  # y
        )

    def forward(self, a, b):
        # shape = [bs, n, dim] ->  [bs, n, dim] -> [bs, n * dim] -> [bs, 1]
        embeds = torch.hstack([a.unsqueeze(1), b.unsqueeze(1)])
        embeds = self.attention(embeds).reshape(len(embeds), -1)
        outputs = self.transfer(embeds).sigmoid()
        return outputs


if __name__ == '__main__':
    bs, dim = 64, 128
    inputs_1 = torch.randn(bs, dim)
    inputs_2 = torch.randn(bs, dim)

    inner_product = InnerOroduct()
    outputs = inner_product(inputs_1, inputs_2)
    print("内积输出形状:", outputs.shape)

    outer_product = OuterProduct()
    outputs = outer_product(inputs_1, inputs_2)
    print("外积输出形状:", outputs.shape)

    dnn_interaction = DNNInteraction(inputs_1.shape[1])
    outputs = dnn_interaction(inputs_1)
    print("DNN输出形状:", outputs.shape)

    light_dnn = LightDnn(inputs_1.shape[1])
    outputs = light_dnn(inputs_1, inputs_2)
    print("Light_dnn输出形状:", outputs.shape)

    att_interaction = AttInteraction(inputs_1.shape[1])
    outputs = att_interaction(inputs_1, inputs_2)
    print("AttInteraction输出形状:", outputs.shape)
