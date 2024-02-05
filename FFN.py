# coding : utf-8
# Author : yuxiang Zeng
import torch


class FFN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return x


if __name__ == '__main__':
    bs = 64
    input_size, hidden_size, output_size = 64, 128, 64
    ffn = FFN(input_size, hidden_size, output_size)
    inputs = torch.randn(bs, input_size)
    outputs = ffn(inputs)
    print(outputs.shape)
