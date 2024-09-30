# coding : utf-8
# Author : yuxiang Zeng


import torch
import torch.nn.functional as F


class SimpleAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleAttention, self).__init__()
        self.fc = torch.nn.Linear(input_size, hidden_size)

    def forward(self, feature1, feature2):
        inputs = torch.cat((feature1, feature2), dim=1)
        attention = self.fc(inputs)
        attention_weights = F.softmax(attention(inputs), dim=1)
        fused_modality = (inputs * attention_weights).sum(dim=1)
        return fused_modality


if __name__ == '__main__':
    # 假设的文本和视频特征维度
    text_dim = 128
    video_dim = 128
    num_heads = 4

    # 模拟输入
    bs = 32  # 批大小
    seq_len = 1  # 序列长度
    text_features = torch.randn(seq_len, bs, text_dim)
    video_features = torch.randn(seq_len, bs, video_dim)

    model = SimpleAttention(text_dim * 2,  modality2.size(1), modality1.size(1))
    output = model(modality1, modality2)
    print(modality1.shape)
