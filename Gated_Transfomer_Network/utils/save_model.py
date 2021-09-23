# @Time    : 2021/07/17 27:15
# @Author  : SY.M
# @FileName: save_model.py

import os
import torch
from config.param_config import Config as c


def save_model(net: torch.nn.Module,
               acc: float):
    """
    保存训练好的模型
    :param net: 训练好参数的模型
    :param acc: 其准确率
    :return: None
    """
    os.makedirs(c.archive.result_root, exist_ok=True)
    path = c.archive.result_root + f'/{c.archive.file_name} {acc} .pkl'
    index = 1
    while os.path.exists(path):
        path = c.archive.result_root + f'/{c.archive.file_name} {acc} ({index}).pkl'
        index += 1
    torch.save(net, path)
