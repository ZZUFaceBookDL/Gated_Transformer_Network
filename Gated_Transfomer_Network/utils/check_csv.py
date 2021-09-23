# @Time    : 2021/07/19 28:39
# @Author  : SY.M
# @FileName: check_csv.py

import numpy as np
import pandas as pd


def check_csv():
    """
    展示 .csv 文件中的内容
    :return: None
    """
    pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
    pd.set_option('display.max_rows', 20)  # b就是你要设置显示的最大的行数参数
    pd.set_option('display.width', 200)  # x就是你要设置的显示的宽度，防止轻易换行

    print("\033[0;34m%s\033[0m" % 'Input the path of .csv file:')
    path = input('')

    df = pd.read_csv(path, index_col=0)
    print("=="*62)
    print(df)
    print("=="*62)


def check_df(df: pd.DataFrame):
    """
    展示DataFrame对象的内容
    :param df: pd.DataFrame对象
    :return: None
    """
    pd.set_option('display.max_columns', 20)  # a就是你要设置显示的最大列数参数
    pd.set_option('display.max_rows', 20)  # b就是你要设置显示的最大的行数参数
    pd.set_option('display.width', 200)  # x就是你要设置的显示的宽度，防止轻易换行
    print("=="*62 + "\r\nParameters show:")
    print(df)
    print("=="*62)

