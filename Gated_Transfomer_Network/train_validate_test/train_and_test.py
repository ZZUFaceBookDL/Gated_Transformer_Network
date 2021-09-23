# @Time    : 2021/07/15 20:20
# @Author  : SY.M
# @FileName: train_and_test.py


import torch
from tqdm import tqdm
from time import time
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import copy
import os

# from utils.save_model import save_model
from utils.random_seed import setup_seed
from config.param_config import Config as c
from module.loss import Myloss
from data_process.create_dataset import UCR_Dataset, MTS_Dataset, UEA_Dataset, Global_Dataset
from utils.update_csv import update_test
from utils.save_model import save_model
from utils.visualization import result_visualization
from utils.check_csv import check_df

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# 设置随机种子
setup_seed(30)


def train_and_test(net_class: torch.nn.Module,
                   all_data: UCR_Dataset or MTS_Dataset or UEA_Dataset):
    """
    使用所有的默认训练集当作训练集，所有的默认测试集当作测试集，使用验证集实验结果中的最优超参进行模型初始化，并进行acc评估
    :param net_class: 使用的模型的类的名称
    :param all_data: 包含所有所需数据的对象
    :return: None
    """
    # create loss function
    loss_function = Myloss()

    # get hyper-parameters directly from param_config.py
    # params = pd.DataFrame({'q': [c.q], 'v': [c.v], 'h': [c.h], 'N': [c.N], 'd_model': [c.d_model], 'd_hidden': [c.d_hidden],
    #                        'd_feature': [c.d_feature if c.archive.__name__ == 'UCR' else all_data.feature_dim],
    #                        'dropout': [c.dropout], 'BATCH_SIZE': [c.BATCH_SIZE], 'EPOCH': [c.EPOCH], 'LR': [c.LR],
    #                        'optimizer_name': [c.optimizer_name]})

    # get best hyper-parameters from validate result
    path = c.archive.result_root + '/' + 'validate_result.csv'
    try:
        params = pd.read_csv(path, index_col=0).head(1)
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
        print("\033[0;31m%s\033[0m" % 'validate csv is empty or file cannot found, Please test after validate!\r\n')
        return

    optimizer_name = params.to_numpy().squeeze()[-1]
    check_df(params)

    # create dataset and dataloader
    train_dataset = Global_Dataset(X=all_data.train_X, Y=all_data.train_Y)
    test_dataset = Global_Dataset(X=all_data.test_X, Y=all_data.test_Y)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(params['BATCH_SIZE']), shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=int(params['BATCH_SIZE']), shuffle=False)

    # create network
    net = net_class(q=int(params['q']), v=int(params['v']), h=int(params['h']), N=int(params['N']),
                    d_model=int(params['d_model']), d_hidden=int(params['d_hidden']),
                    d_feature=int(params['d_feature']) if c.archive.__name__ == 'UCR' else all_data.feature_dim,
                    d_timestep=all_data.time_step_len, class_num=all_data.class_num, dropout=float(params['dropout'])).to(DEVICE)

    # create optimizer
    optimizer = None
    if optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=float(params['LR']))
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=float(params['LR']))
    assert optimizer is not None, 'optimizer is None!'

    loss_list = []
    accuracy_on_train = []
    accuracy_on_test = []
    # max_acc_now = 0.0  # record the best acc on current circulation
    max_acc_all = pd.read_csv(c.archive.result_root + '/test_result.csv').acc.max() \
        if os.path.exists(c.archive.result_root + '/test_result.csv') else 0.0  # record the best acc on all experiment
    save_flag = False
    loss_sum_min = 99999
    best_net = None
    pbar = tqdm(total=int(params['EPOCH']))
    begin_time = time()
    net.train()
    for index in range(int(params['EPOCH'])):
        loss_sum = 0.0
        for x, y in train_dataloader:
            optimizer.zero_grad()
            y_pre = net(x.to(DEVICE), 'train')
            loss = loss_function(y_pre, y.to(DEVICE))
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        print(f'EPOCH:{index + 1}\t\tLoss:{round(loss_sum, 5)}')
        loss_list.append(loss_sum)
        pbar.update()

        '''
        # test and save model
        if (index + 1) % c.test_interval == 0:
            trian_acc, test_acc = test(net=net, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
            accuracy_on_train.append(trian_acc)
            accuracy_on_test.append(test_acc)
            if test_acc > max_acc_now:
                max_acc_now = test_acc
                if test_acc > max_acc_all:
                    best_net = copy.deepcopy(net)
                    save_flag = True
            print(f'Current max acc\ttest:{max(accuracy_on_test)}\ttrain:{max(accuracy_on_train)}')
        '''

        if loss_sum < loss_sum_min:
            loss_sum_min = loss_sum
            best_net = copy.deepcopy(net)

    trian_acc, test_acc = test(net=best_net, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    save_flag = (test_acc > max_acc_all)

    end_time = time()
    time_cost = round((end_time - begin_time) / 60, 2)
    # show result digits on terminal
    print(f'\033[0;34macc_on_test:{test_acc}%\tacc_on_train:{trian_acc}%\033[0m')
    print(f'\033[0;34mEpoch index:{loss_list.index(min(loss_list)) + 1}\033[0m')
    print(f"\033[0;34mtotal time cost:{time_cost} min\033[0m")

    # update csv result
    update_test(accuracy_on_test=test_acc, params=params)
    # save_model
    if save_flag:
        save_model(net=best_net, acc=test_acc)
    # show or save result plot
    result_visualization(loss_list=loss_list, correct_on_test=test_acc, correct_on_train=trian_acc,
                         all_data=all_data, time_cost=time_cost, params=params)


def test(net,
         train_dataloader: torch.utils.data.DataLoader,
         test_dataloader: torch.utils.data.DataLoader):
    """
    计算acc
    :param net: 经过训练的模型
    :param train_dataloader: train_dataloader
    :param test_dataloader: test_dataloader
    :return: 训练集acc，测试集acc
    """
    with torch.no_grad():
        net.eval()

        # test on test dataset
        correct = 0
        total = 0
        for x_val, y_val in test_dataloader:
            pre = net(x_val.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_val.long().to(DEVICE))).item()
        accuracy_on_test = round(correct / total, 4) * 100
        print(f'accuracy on test:{accuracy_on_test}%')

        # test on train dataset
        correct = 0
        total = 0
        for x_train, y_train in train_dataloader:
            pre = net(x_train.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y_train.long().to(DEVICE))).item()
        accuracy_on_train = round(correct / total, 4) * 100
        print(f'accuracy on train:{accuracy_on_train}%')

        return accuracy_on_train, accuracy_on_test
