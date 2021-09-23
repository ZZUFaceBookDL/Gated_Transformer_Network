# @Time    : 2021/07/21 19:29
# @Author  : SY.M
# @FileName: encoder.py

import torch
from module.for_TS.multiHeadAttention import MultiHeadAttention
from module.for_TS.feedforward import PositionFeedforward


class Encoder(torch.nn.Module):
    def __init__(self,
                 q: int,
                 v: int,
                 h: int,
                 d_model: int,
                 d_hidden: int,
                 dropout: float = 0.2):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h)
        self.feedforward = PositionFeedforward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layernorm = torch.nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                stage: str):
        residual = x
        x, heatmap_score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layernorm(x + residual)

        x = self.feedforward(x)

        return x, heatmap_score