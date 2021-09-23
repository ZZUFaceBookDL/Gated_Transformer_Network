# @Time    : 2021/07/15 20:33
# @Author  : SY.M
# @FileName: loss.py


import torch


class Myloss(torch.nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, pre, target):

        loss = self.loss_function(pre, target.long())

        return loss