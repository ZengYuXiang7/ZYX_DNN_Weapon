# coding : utf-8
# Author : yuxiang Zeng
import torch
import numpy as np


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = torch.nn.Linear(d_model, h * d_k)
        self.fc_k = torch.nn.Linear(d_model, h * d_k)
        self.fc_v = torch.nn.Linear(d_model, h * d_v)
        self.fc_o = torch.nn.Linear(h * d_v, d_model)
        self.dropout = torch.nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        queries = queries.to(torch.float32)
        keys = keys.to(torch.float32)
        values = values.to(torch.float32)

        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

if __name__ == '__main__':
    from time import time

    heads = 8
    bs, n, dim = 64, 5, 128
    inputs = torch.randn(bs, n, dim)

    t1 = time()
    attention = ScaledDotProductAttention(dim, dim, dim, h = heads, dropout=0.1)
    outputs = attention(inputs, inputs, inputs)
    t2 = time()
    print(outputs.shape, t2 - t1)


    t1 = time()
    attention = torch.nn.MultiheadAttention(dim, heads, 0.1)
    outputs, weights = attention(inputs, inputs, inputs)
    t2 = time()
    print(outputs.shape, weights.shape, t2 - t1)
