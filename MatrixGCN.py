# coding : utf-8
# Author : yuxiang Zeng

import math
import torch

from Experiment import get_args


class GraphConvolution(torch.torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_init='thomas', bias_init='thomas'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        if bias:
            self.bias = torch.torch.nn.Parameter(torch.DoubleTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.reset_parameters()

    def reset_parameters(self):
        self.init_tensor(self.weight, self.weight_init, 'act')
        self.init_tensor(self.bias, self.bias_init, 'act')

    def forward(self, adjacency, features):
        support = torch.matmul(features, self.weight)
        output = torch.bmm(adjacency, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

    @staticmethod
    def init_tensor(tensor, init_type, nonlinearity):
        if tensor is None or init_type is None:
            return
        if init_type == 'thomas':
            size = tensor.size(-1)
            stdv = 1. / math.sqrt(size)
            torch.nn.init.uniform_(tensor, -stdv, stdv)
        elif init_type == 'kaiming_normal_in':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_normal_out':
            torch.nn.init.kaiming_normal_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_in':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity=nonlinearity)
        elif init_type == 'kaiming_uniform_out':
            torch.nn.init.kaiming_uniform_(tensor, mode='fan_out', nonlinearity=nonlinearity)
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(tensor, gain=torch.nn.init.calculate_gain(nonlinearity))
        else:
            raise ValueError(f'Unknown initialization type: {init_type}')

class GCN(torch.torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, args):
        super(GCN, self).__init__()
        self.args = args
        self.nfeat = input_dim
        self.nlayer = num_layers
        self.nhid = hidden_dim
        self.dropout_ratio = dropout
        weight_init = 'thomas'
        bias_init = 'thomas'
        augments = 0
        self.binary_classifier = False

        # Layers
        self.gcn = torch.nn.ModuleList()
        self.norm = torch.nn.ModuleList()
        self.act = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        for i in range(self.nlayer):
            self.gcn.append(GraphConvolution(self.nfeat if i == 0 else self.nhid, self.nhid, bias=True, weight_init=weight_init, bias_init=bias_init))
            self.norm.append(torch.nn.LayerNorm(self.nhid).double())
            self.act.append(torch.nn.ReLU().double())
            self.dropout.append(torch.nn.Dropout(self.dropout_ratio).double())

        if not self.binary_classifier:
            self.fc = torch.nn.Linear(self.nhid + augments, 1).double()
        else:
            if self.binary_classifier == 'naive':
                self.fc = torch.nn.Linear(self.nhid + augments, 1).double()
            elif self.binary_classifier == 'oneway' or self.binary_classifier == 'oneway-hard':
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 1).double()
            else:
                self.fc = torch.nn.Linear((self.nhid + augments) * 2, 2).double()
            if self.binary_classifier != 'oneway' and self.binary_classifier != 'oneway-hard':
                self.final_act = torch.nn.LogSoftmax(dim=1)
            else:
                self.final_act = torch.nn.Sigmoid()

    def forward_single_model(self, adjacency, features):
        x = self.act[0](self.norm[0](self.gcn[0](adjacency, features)))
        x = self.dropout[0](x)
        for i in range(1, self.nlayer):
            x = self.act[i](self.norm[i](self.gcn[i](adjacency, x)))
            x = self.dropout[i](x)
        return x

    def extract_features(self, adjacency, features, augments=None):
        x = self.forward_single_model(adjacency, features)
        x = x[:, 0]  # use global node
        if augments is not None:
            x = torch.cat([x, augments], dim=1)
        return x

    def forward(self, adjacency, features):
        adjacency = adjacency.double()
        features = features.double()
        augments = None
        if not self.binary_classifier:
            x = self.forward_single_model(adjacency, features)
            x = x[:, 0]  # use global node
            if augments is not None:
                x = torch.cat([x, augments], dim=1)
            y = self.fc(x).flatten()
            return y
        else:
            x1 = self.forward_single_model(adjacency[:, 0], features[:, 0])
            x1 = x1[:, 0]
            x2 = self.forward_single_model(adjacency[:, 1], features[:, 1])
            x2 = x2[:, 0]
            if augments is not None:
                a1 = augments[:, 0]
                a2 = augments[:, 1]
                x1 = torch.cat([x1, a1], dim=1)
                x2 = torch.cat([x2, a2], dim=1)
            if self.binary_classifier == 'naive':
                x1 = self.fc(x1)
                x2 = self.fc(x2)
            x = torch.cat([x1, x2], dim=1)
            if self.binary_classifier != 'naive':
                x = self.fc(x)
            x = self.final_act(x)
            return x


if __name__ == '__main__':
    adjacency = torch.tensor([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=torch.float32).unsqueeze(0)  # 增加一个批处理维度
    features = torch.randn((1, 3, 6), dtype=torch.float32)  # 随机生成特征矩阵

    args = get_args()
    model = GCN(6, 600, 6, 0.02, args)

    output = model(adjacency, features)
    print("Output shape:", output.shape)
