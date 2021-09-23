# @Time    : 2021/07/15 26:26
# @Author  : SY.M
# @FileName: update_csv.py

import os
import torch
import numpy as np
import pandas as pd

from config.param_config import Config as c


def update_validate(max_validate_acc_list: list):
    """
    更新验证集上的结果，将超参数和对应的结果保存在.csv文件中
    :param max_validate_acc_list: k折交叉验证中每一折的acc
    :return: None
    """
    os.makedirs(c.archive.result_root, exist_ok=True)

    temp = pd.DataFrame({'max_acc': [max(max_validate_acc_list)], 'aver_acc': [np.mean(max_validate_acc_list)],
                           'variance': [np.var(max_validate_acc_list)], 'EPOCH': [c.EPOCH], 'BATCH_SIZE': [c.BATCH_SIZE],
                           'LR': [c.LR], 'd_model': [c.d_model],
                           'd_feature': [None if c.archive.__name__ != 'UCR' else c.d_feature],
                           'd_hidden': [c.d_hidden], 'q': [c.q], 'v': [c.v],
                           'h': [c.h], 'N': [c.N], 'dropout': [c.dropout], 'optimizer_name': [c.optimizer_name]})

    if not os.path.exists(c.archive.result_root + '/validate_result.csv'):
        temp.to_csv(c.archive.result_root + '/validate_result.csv', encoding='utf-8')
    else:
        df = pd.read_csv(c.archive.result_root + '/validate_result.csv', index_col=0)
        df = df.append(temp, ignore_index=True)
        df = df.drop_duplicates()
        df = df.sort_values(by=['aver_acc', 'variance', 'max_acc'], ascending=[False, True, False])
        try:
            df.to_csv(c.archive.result_root + '/validate_result.csv', encoding='utf-8')
        except PermissionError:
            print(
                "\033[0;31;40m%s\033[0m" % "The file is occupied, Please close it and the input anything to continue!!!")
            input('')
            df.to_csv(c.archive.result_root + '/validate_result.csv', encoding='utf-8')


def update_test(accuracy_on_test: float,
                params: pd.DataFrame):
    """
    更新测试集上的结果，将超参数和对应的结果保存在.csv文件中
    :param accuracy_on_test: 准确率结果
    :return: None
    """
    os.makedirs(c.archive.result_root, exist_ok=True)

    optimizer_name = params.to_numpy().squeeze()[-1]

    temp = pd.DataFrame({'acc': [accuracy_on_test], 'EPOCH': [int(params["EPOCH"])], 'BATCH_SIZE': [int(params["BATCH_SIZE"])],
                         'LR': [float(params["LR"])], 'd_model': [int(params["d_model"])],
                         'd_feature': [None if c.archive.__name__ != 'UCR' else int(params["d_feature"])],
                         'd_hidden': [int(params["d_hidden"])], 'q': [int(params["q"])], 'v': [int(params["v"])],
                         'h': [int(params["h"])], 'N': [int(params["N"])], 'dropout': [float(params["dropout"])], 'optimizer_name': [optimizer_name]})
    if not os.path.exists(c.archive.result_root + '/test_result.csv'):
        temp.to_csv(c.archive.result_root + '/test_result.csv', encoding='utf-8')
    else:
        df = pd.read_csv(c.archive.result_root + '/test_result.csv', index_col=0)
        df = df.append(temp, ignore_index=True)
        df = df.drop_duplicates()
        df = df.sort_values(by=['acc'], ascending=[False])
        try:
            df.to_csv(c.archive.result_root + '/test_result.csv', encoding='utf-8')
        except PermissionError:
            print("\033[0;31;40m%s\033[0m" % "The file is occupied, Please close it and the input anything to continue!!!")
            input('')
            df.to_csv(c.archive.result_root + '/test_result.csv', encoding='utf-8')