# coding : utf-8
# Author : yuxiang Zeng

import torch
import torch.nn as nn


class DynamicFeatureAggregator(nn.Module):
    def __init__(self):
        super(DynamicFeatureAggregator, self).__init__()

    def forward(self, inputs, method='add'):
        """
        前向传播
        :param inputs: 输入特征的列表，每个元素的维度为 (batch_size, feature_dim)
        :param method: 聚合方法，'add'、'average' 或 'concatenate'
        :return: 聚合后的特征
        """
        if method == 'add':
            # 直接相加
            aggregated_output = torch.sum(torch.stack(inputs, dim=0), dim=0)
        elif method == 'mean':
            # 取平均
            aggregated_output = torch.mean(torch.stack(inputs, dim=0), dim=0)
        elif method == 'cat':
            # 拼接
            # 注意：拼接操作要求除了拼接维度外，其他维度的大小必须一致
            aggregated_output = torch.cat(inputs, dim=-1)
        else:
            raise ValueError("Unsupported aggregation method: {}".format(method))
        return aggregated_output


if __name__ == '__main__':
    input_features_1 = torch.rand((10, 5))  # (batch_size, feature_dim)
    input_features_2 = torch.rand((10, 5))
    input_features_3 = torch.rand((10, 5))

    # 创建模型实例
    aggregator = DynamicFeatureAggregator()

    # 使用直接相加方法
    aggregated_features_add = aggregator([input_features_1, input_features_2, input_features_3], method='add')
    print("Directly Added Shape:", aggregated_features_add.shape)

    # 使用取平均方法
    aggregated_features_avg = aggregator([input_features_1, input_features_2, input_features_3], method='mean')
    print("Averaged Shape:", aggregated_features_avg.shape)

    # 使用拼接方法
    aggregated_features_concat = aggregator([input_features_1, input_features_2, input_features_3], method='cat')
    print("Concatenated Shape:", aggregated_features_concat.shape)

