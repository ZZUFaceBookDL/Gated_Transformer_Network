# @Time    : 2021/07/21 19:30
# @Author  : SY.M
# @FileName: transformer.py


import torch
from module.for_TS.embedding import Embedding
from module.for_TS.encoder import Encoder


class Transformer(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 d_feature: int,
                 d_timestep: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 class_num: int,
                 dropout: float = 0.2):
        super(Transformer, self).__init__()

        self.timestep_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, dropout=dropout,
                                            d_model=d_model, wise='timestep')
        self.feature_embedding = Embedding(d_feature=d_feature, d_timestep=d_timestep, dropout=dropout,
                                           d_model=d_model, wise='feature')

        self.timestep_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])

        self.feature_encoderlist = torch.nn.ModuleList([Encoder(
            d_model=d_model,
            d_hidden=d_hidden,
            q=q,
            v=v,
            h=h,
            dropout=dropout) for _ in range(N)])

        self.gate = torch.nn.Linear(in_features=d_timestep * d_model + d_feature * d_model, out_features=2)
        self.linear_out = torch.nn.Linear(in_features=d_timestep * d_model + d_feature * d_model,
                                          out_features=class_num)

    def forward(self,
                x: torch.Tensor,
                stage: str = 'train' or 'test'):

        x_timestep, _ = self.timestep_embedding(x)
        x_feature, _ = self.feature_embedding(x)

        for encoder in self.timestep_encoderlist:
            x_timestep, heatmap = encoder(x_timestep, stage=stage)

        for encoder in self.feature_encoderlist:
            x_feature, heatmap = encoder(x_feature, stage=stage)

        x_timestep = x_timestep.reshape(x_timestep.shape[0], -1)
        x_feature = x_feature.reshape(x_feature.shape[0], -1)

        gate = torch.nn.functional.softmax(self.gate(torch.cat([x_timestep, x_feature], dim=-1)), dim=-1)

        gate_out = torch.cat([x_timestep * gate[:, 0:1], x_feature * gate[:, 1:2]], dim=-1)

        out = self.linear_out(gate_out)

        return out
