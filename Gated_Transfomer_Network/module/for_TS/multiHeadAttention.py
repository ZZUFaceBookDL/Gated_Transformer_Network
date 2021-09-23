# @Time    : 2021/07/21 19:29
# @Author  : SY.M
# @FileName: multiHeadAttention.py


import torch


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.q = q

        self.W_Q = torch.nn.Linear(in_features=d_model, out_features=q * h)
        self.W_K = torch.nn.Linear(in_features=d_model, out_features=q * h)
        self.W_V = torch.nn.Linear(in_features=d_model, out_features=v * h)
        self.W_out = torch.nn.Linear(in_features=v * h, out_features=d_model)

        self.inf = -2**32+1
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self,
                x: torch.Tensor,
                stage: str):

        Q = torch.cat(self.W_Q(x).chunk(self.h, dim=-1), dim=0)
        K = torch.cat(self.W_K(x).chunk(self.h, dim=-1), dim=0)
        V = torch.cat(self.W_V(x).chunk(self.h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2))  # / torch.sqrt(torch.Tensor(self.q)).to(self.device)

        heatmap_score = score

        if stage == 'train':
            mask = torch.ones_like(score[0])
            mask = mask.tril(diagonal=0)
            score = torch.where(mask > 0, score, (torch.ones_like(mask) * self.inf).to(self.device))

        score = torch.nn.functional.softmax(score, dim=-1)
        weight_V = torch.cat(torch.matmul(score, V).chunk(self.h, dim=0), dim=-1)

        out = self.W_out(weight_V)

        return out, heatmap_score