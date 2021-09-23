# @Time    : 2021/07/19 27:29
# @Author  : SY.M
# @FileName: visualization.py

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as fp
import math
from data_process.create_dataset import UCR_Dataset, MTS_Dataset, UEA_Dataset
from config.param_config import Config as c
import pandas as pd


def result_visualization(loss_list: list,
                         correct_on_test: float,
                         correct_on_train: float,
                         all_data: UCR_Dataset or MTS_Dataset or UEA_Dataset,
                         time_cost: float,
                         params: pd.DataFrame):
    """
    实验结果可视化
    :param loss_list: 损失列表
    :param correct_on_test: 测试集acc
    :param correct_on_train: 训练集acc
    :param all_data: 包含所有数据的对象
    :param time_cost: 花费时间
    :param params: 超参字典
    :return:
    """
    my_font = fp(fname=r"font/simsun.ttc")  # load font setting file

    # 设置风格
    plt.style.use('seaborn')

    fig = plt.figure()  # create basic plot
    ax1 = fig.add_subplot(311)  # create subplot
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)  # add plot
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')

    # 设置坐标轴标签 和 图的标题
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{c.test_interval}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')

    # set text
    fig.text(x=0.13, y=0.4, s=f'最小loss：{min(loss_list)}' '    '
                              f'最小loss对应的epoch数:{loss_list.index(min(loss_list)) + 1}' '    '
                              f'最后一轮loss:{loss_list[-1]}' '\n'
                              f'correct：测试集:{correct_on_test}% 训练集:{correct_on_train}%' '    '
                              f'最小loss对应的已训练epoch数:{loss_list.index(min(loss_list)) + 1}' '    '
                              f'最后一轮correct：{correct_on_test[-1]}%' '\n'
                              f'd_model={int(params["d_model"])}   q={int(params["q"])}   v={int(params["v"])}   h={int(params["h"])}   '
                              f'N={int(params["N"])}  drop_out={float(params["dropout"])}'  '\n'
                              f'共耗时{round(time_cost, 2)}分钟', FontProperties=my_font)

    optimizer_name = params.to_numpy().squeeze()[-1]
    # save plot
    if int(params["EPOCH"]) >= c.draw_key:
        plt.savefig(
            f'{c.archive.result_root}/{c.archive.file_name} {correct_on_test}% {optimizer_name} epoch={int(params["EPOCH"])} '
            f'batch={int(params["BATCH_SIZE"])} d_model={int(params["d_model"])} d_feature={None if c.archive.__name__!="UCR" else int(params["d_feature"])}'
            f'd_hidden={int(params["d_hidden"])} qvhN=[{int(params["q"])},{int(params["v"])},{int(params["h"])},{int(params["N"])}] '
            f'dropout={float(params["dropout"])} lr={float(params["LR"])}.png')

    # show plot
    plt.show()

    print('正确率', correct_on_test)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{loss_list.index(min(loss_list)) + 1}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'correct：测试集:{correct_on_test}\t 训练集:{correct_on_train}\r\n'
          f'correct对应的已训练epoch数:{loss_list.index(min(loss_list)) + 1}\r\n')

    print(f'共耗时{round(time_cost, 2)}分钟')