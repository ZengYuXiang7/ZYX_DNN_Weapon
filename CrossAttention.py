# coding : utf-8
# Author : yuxiang Zeng
import torch


class CrossAttention(torch.nn.Module):
    def __init__(self, first_dim, second_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.first_to_second_attention = torch.nn.MultiheadAttention(embed_dim=first_dim, num_heads=num_heads, dropout=0.1)
        self.second_to_first_attention = torch.nn.MultiheadAttention(embed_dim=second_dim, num_heads=num_heads, dropout=0.1)

    def forward(self, first_features, second_features):
        first_to_second_attention_output, _ = self.first_to_second_attention(first_features, second_features, second_features)
        second_to_first_attention_output, _ = self.second_to_first_attention(second_features, first_features, first_features)
        # 聚合
        fused_output = torch.cat((first_to_second_attention_output, second_to_first_attention_output), dim=0)
        fused_output = fused_output.mean(dim=0)  # 假设我们通过取均值来简化处理
        return fused_output


if __name__ == '__main__':
    # 假设的文本和视频特征维度
    text_dim = 128
    video_dim = 128
    num_heads = 4

    # 模拟输入
    bs = 32      # 批大小
    seq_len = 1  # 序列长度
    text_features = torch.randn(seq_len, bs, text_dim)
    video_features = torch.randn(seq_len, bs, video_dim)

    cross_attention = CrossAttention(128, 128, num_heads)
    fused_output = cross_attention(text_features, video_features)
    print(fused_output.shape)



