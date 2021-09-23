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
                 wise: str = 'timestep' or 'feature',
                 dropout: float = 0.2):
        super(Embedding, self).__init__()

        self.wise = wise
        assert wise == 'timestep' or wise == 'feature', 'Embedding wise error!'
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv1d(in_channels=256, out_channels=d_feature, kernel_size=3, padding=1)

        if wise == 'timestep':
            self.linear = torch.nn.Linear(d_feature, d_model)
        elif wise == 'feature':
            self.linear = torch.nn.Linear(d_timestep, d_model)

        self.relu = torch.nn.ReLU(dropout)

    def forward(self, x):

        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        expand_tensor = x

        if self.wise == 'timestep':
            x = self.linear(x.transpose(-1, -2))
            x = position_encode(x)
        elif self.wise == 'feature':
            x = self.linear(x)

        return self.relu(x), expand_tensor


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