# @Time    : 2021/07/21 19:28
# @Author  : SY.M
# @FileName: feedforward.py

import torch


class PositionFeedforward(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int = 2048):
        super(PositionFeedforward, self).__init__()

        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=d_hidden)
        self.linear2 = torch.nn.Linear(in_features=d_hidden, out_features=d_model)
        self.relu = torch.nn.ReLU()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):

        residual = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.layernorm(x + residual)

        return x