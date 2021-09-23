# @Time    : 2021/07/21 19:36
# @Author  : SY.M
# @FileName: embedding.py

import torch
import math


class Embedding(torch.nn.Module):
    def __init__(self,
                 d_feature: int,
                 d_timestep: int,
                 d_model: int,
                 wise: str = 'timestep' or 'feature'):
        super(Embedding, self).__init__()

        assert wise == 'timestep' or wise == 'feature', 'Embedding wise error!'
        self.wise = wise

        if wise == 'timestep':
            self.embedding = torch.nn.Linear(d_feature, d_model)
        elif wise == 'feature':
            self.embedding = torch.nn.Linear(d_timestep, d_model)

    def forward(self,
                x: torch.Tensor):
        if self.wise == 'feature':
            x = self.embedding(x)
        elif self.wise == 'timestep':
            x = self.embedding(x.transpose(-1, -2))
            x = position_encode(x)

        return x, None


def position_encode(x):

    pe = torch.ones_like(x[0])
    position = torch.arange(0, x.shape[1]).unsqueeze(-1)
    temp = torch.Tensor(range(0, x.shape[-1], 2))
    temp = temp * -(math.log(10000) / x.shape[-1])
    temp = torch.exp(temp).unsqueeze(0)
    temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
    pe[:, 0::2] = torch.sin(temp)
    pe[:, 1::2] = torch.cos(temp)

    return x + pe