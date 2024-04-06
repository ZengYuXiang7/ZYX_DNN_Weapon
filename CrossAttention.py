# coding : utf-8
# Author : yuxiang Zeng
import torch

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

    text_to_video_attention = torch.nn.MultiheadAttention(embed_dim=text_dim, num_heads=num_heads, dropout=0.1)
    video_to_text_attention = torch.nn.MultiheadAttention(embed_dim=video_dim, num_heads=num_heads, dropout=0.1)

    text_to_video_attention_output, text_to_video_attention_weights = text_to_video_attention(text_features, video_features, video_features)
    video_to_text_attention_output, video_to_text_attention_weights = video_to_text_attention(video_features, text_features, text_features)

    print(text_to_video_attention_output.shape, text_to_video_attention_weights.shape)
    print(video_to_text_attention_output.shape, video_to_text_attention_weights.shape)

    # 聚合
    fused_output = torch.cat((text_to_video_attention_output, video_to_text_attention_output), dim=0)
    print(fused_output.shape)
    fused_output = fused_output.mean(dim=0)  # 假设我们通过取均值来简化处理
    print(fused_output.shape)



