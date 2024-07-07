# coding : utf-8
# Author : yuxiang Zeng

import torch


class Predictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout):
        super(Predictor, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.dimension = input_dim
        mlp_modules = []
        input_size = self.dimension
        for i in range(self.num_layers):
            mlp_modules.append(torch.nn.Dropout(p=self.dropout))
            mlp_modules.append(torch.nn.Linear(input_size, input_size // 2))
            mlp_modules.append(torch.nn.ReLU())
            input_size = input_size // 2
        self.mlp_layers = torch.nn.Sequential(*mlp_modules)
        self.predict_layer = torch.nn.Linear(input_size, output_dim)

    def forward(self, x):
        x = self.mlp_layers(x)
        x = self.predict_layer(x)
        return x


if __name__ == '__main__':
    bs, dim = 64, 128
    inputs = torch.randn(bs, dim)
    mlp = FNN(dim, 1, 2, 0.10)
    outputs = mlp(inputs)
    print(outputs.shape)

