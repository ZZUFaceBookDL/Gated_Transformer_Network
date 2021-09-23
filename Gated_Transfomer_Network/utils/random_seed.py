import torch
import numpy as np


def setup_seed(seed):
    """
    设置随机种子
    :param seed: 随机种子数
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


