# @Time    : 2021/07/15 20:20
# @Author  : SY.M
# @FileName: train_and_validate.py


import torch
from tqdm import tqdm
from time import time
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy

from config.param_config import Config as c
from utils.random_seed import setup_seed
from module.loss import Myloss
from data_process.create_dataset import UCR_Dataset, MTS_Dataset, UEA_Dataset, Global_Dataset
from utils.update_csv import update_validate

# from utils.save_model import save_model


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Use device {DEVICE}')
# 设置随机种子
setup_seed(30)


def train_and_validate(net_class: torch.nn.Module,
                       all_data: UCR_Dataset or MTS_Dataset or UEA_Dataset):
    """
    使用所有默认训练集数据构建k折交叉验证，使用划分训练集进行训练，使用划分验证集进行验证，以进行超参调优
    :param net_class: 使用的模型的类的名称
    :param all_data: 包含所有所需数据的对象
    :return: None
    """
    # 创建损失函数
    loss_function = Myloss()
    # k折交叉验证下标集
    sfk = all_data.sfk_indexset
    pbar = tqdm(total=c.EPOCH * c.k_fold)
    max_validate_acc_list = []  # record max accuracy on validate data set of each fold

    begin_time = time()
    # k折交叉验证循环
    for n_fold, (train_index, val_index) in enumerate(sfk):

        # 创建训练集、验证集
        train_X, val_X = all_data.train_X[train_index], all_data.train_X[val_index]
        train_Y, val_Y = all_data.train_Y[train_index], all_data.train_Y[val_index]
        train_dataset = Global_Dataset(X=train_X, Y=train_Y)
        val_dataset = Global_Dataset(X=val_X, Y=val_Y)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=c.BATCH_SIZE, shuffle=False)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=c.BATCH_SIZE, shuffle=False)

        # 创建网络
        net = net_class(q=c.q, v=c.v, h=c.h, N=c.N, d_model=c.d_model, d_hidden=c.d_hidden,
                        d_feature=c.d_feature if c.archive.__name__ == 'UCR' else all_data.feature_dim,
                        d_timestep=all_data.time_step_len, class_num=all_data.class_num, dropout=c.dropout).to(DEVICE)

        # 创建优化器
        optimizer = None
        if c.optimizer_name == 'Adagrad':
            optimizer = torch.optim.Adagrad(net.parameters(), lr=c.LR)
        elif c.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=c.LR)
        assert optimizer is not None, 'optimizer is None!'

        loss_list = []
        accuracy_on_train = []
        accuracy_on_validate = []

        net.train()
        loss_sum_min = 99999
        best_net = None
        for index in range(c.EPOCH):
            loss_sum = 0.0
            for x, y in train_dataloader:
                optimizer.zero_grad()
                y_pre = net(x.to(DEVICE), 'train')
                loss = loss_function(y_pre, y.to(DEVICE))
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            if loss_sum < loss_sum_min:
                loss_sum_min = loss_sum
                best_net = copy.deepcopy(net)

            print(f'第{1+n_fold}折 EPOCH:{index + 1}\t\tLoss:{round(loss_sum, 5)}')
            loss_list.append(loss_sum)
            pbar.update()

            '''
            if (index + 1) % c.test_interval == 0:
                trian_acc, val_acc = validate(net=net, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
                accuracy_on_train.append(trian_acc)
                accuracy_on_validate.append(val_acc)
                print(f'第{1+n_fold}折 当前最大准确率 验证集:{max(accuracy_on_validate)}\t训练集:{max(accuracy_on_train)}')
            '''

        trian_acc, val_acc = validate(net=best_net, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
        print(f'第{1 + n_fold}折 准确率 验证集:{trian_acc}\t训练集:{val_acc}')
        # max_validate_acc_list.append(max(accuracy_on_validate))
        max_validate_acc_list.append(val_acc)

    end_time = time()
    time_cost = round((end_time - begin_time) / 60, 2)

    print(f'\033[0;34m acc in each fold:{max_validate_acc_list}\033[0m')
    print(f'\033[0;34m max_acc:{max(max_validate_acc_list)}    mean_acc:{np.mean(max_validate_acc_list)}    variance:{np.var(max_validate_acc_list)}\033[0m')
    print(f'\033[0;34m total time cost:{time_cost} min\033[0m')

    update_validate(max_validate_acc_list=max_validate_acc_list)


def validate(net,
             train_dataloader: torch.utils.data.DataLoader,
             val_dataloader: torch.utils.data.DataLoader):
    """
    计算acc
    :param net: 经过训练的模型
    :param train_dataloader: train_dataloader
    :param val_dataloader: validate_dataloader
    :return: 训练集acc，验证集acc
    """
    with torch.no_grad():
        net.eval()

        # 测试验证集
        correct = 0
        total = 0
        for x_val, y_val in val_dataloader:
            pre = net(x_val.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_val.long().to(DEVICE))).item()
        accuracy_on_validate = round(correct / total, 4) * 100
        print(f'accuracy on validate:{accuracy_on_validate}%')

        # 测试训练集
        correct = 0
        total = 0
        for x_train, y_train in train_dataloader:
            pre = net(x_train.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_train.long().to(DEVICE))).item()
        accuracy_on_train = round(correct / total, 4) * 100
        print(f'accuracy on train:{accuracy_on_train}%')

        return accuracy_on_train, accuracy_on_validate
