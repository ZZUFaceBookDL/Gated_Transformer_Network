# @Time    : 2021/07/16 19:50
# @Author  : SY.M
# @FileName: create_dataset.py

import  torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from scipy.io import loadmat
from config.param_config import Config as c


def get_MyData(root, file_name, archive_name):
    """
    get object that obtain all data of one dataset
    :param root:  数据集根路径
    :param file_name:  数据集名称
    :param archive_name:  数据库名称
    :return:  包含所有数据的对象
    """
    all_Data = None

    if archive_name == 'MTS':
        all_Data = MTS_Dataset(root=root, file_name=file_name)
    elif archive_name == 'UCR':
        all_Data = UCR_Dataset(root=root, file_name=file_name)
    elif archive_name == 'UEA':
        all_Data = UEA_Dataset(root=root, file_name=file_name)

    assert all_Data is not None, 'archive_name value error！'

    return all_Data


class Global_Dataset(Dataset):
    def __init__(self, X, Y):
        """
        根据数据和标签构建训练集、验证集或者测试集
        :param X: 数据
        :param Y: 标签
        """
        super(Global_Dataset, self).__init__()
        self.X = X
        self.Y = Y
        self.Y[self.Y == -1] = 0
        self.min_label = min(self.Y)
        self.Y = self.Y - self.min_label
        self.data_len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.data_len


class UCR_Dataset:
    def __init__(self,
                 root: str,
                 file_name: str):
        """
        为UCR数据集构建包含全部所需数据的对象
        :param root: 数据集根路径
        :param file_name: 数据集名称
        """
        train_path = root + '/' + file_name + '/' + file_name + '_TRAIN.tsv'
        test_path = root + '/' + file_name + '/' + file_name + '_TEST.tsv'

        train_xy = np.loadtxt(train_path, delimiter='\t', dtype=np.float32)
        test_xy = np.loadtxt(test_path, delimiter='\t', dtype=np.float32)

        self.train_X = train_xy[:, 1:]
        self.train_Y = train_xy[:, 0]
        self.train_Y[self.train_Y == -1] = 0
        self.train_len = self.train_X.shape[0]

        self.test_X = test_xy[:, 1:]
        self.test_Y = test_xy[:, 0]
        self.test_Y[self.test_Y == -1] = 0
        self.test_len = self.test_X.shape[0]

        self.time_step_len = self.train_X.shape[-1]
        self.class_num = len(set(self.train_Y.tolist()))

        self.min_label = min(self.train_Y)
        self.train_Y = self.train_Y - self.min_label
        self.test_Y = self.test_Y - self.min_label
        # self.feature_dim = None

        min_count = int(min(torch.LongTensor(self.train_Y).long().bincount().data))
        self.sfk_indexset = StratifiedKFold(n_splits=c.k_fold if min_count >= c.k_fold else min_count, shuffle=False) \
            .split(self.train_X, self.train_Y)


class UEA_Dataset:
    def __init__(self,
                 root,
                 file_name):
        """
        为UEA数据集构建包含全部所需数据的对象
        :param root: 数据集根路径
        :param file_name: 数据集名称
        """
        train_X_path = root + '/' + file_name + '/' + 'X_train.npy'
        train_Y_path = root + '/' + file_name + '/' + 'y_train.npy'
        test_X_path = root + '/' + file_name + '/' + 'X_test.npy'
        test_Y_path = root + '/' + file_name + '/' + 'y_test.npy'

        self.train_X = torch.FloatTensor(np.load(train_X_path)).transpose(-1, -2)
        self.train_Y = torch.FloatTensor(np.load(train_Y_path).squeeze())
        self.train_len = self.train_X.shape[0]

        self.test_X = torch.FloatTensor(np.load(test_X_path).squeeze()).transpose(-1, -2)
        self.test_Y = torch.FloatTensor(np.load(test_Y_path).squeeze())
        self.test_len = self.test_X.shape[0]

        self.time_step_len = self.train_X.shape[-1]
        self.feature_dim = self.train_X.shape[1]
        self.class_num = len(set(self.train_Y.tolist()))

        self.min_label = min(self.train_Y)
        self.train_Y = self.train_Y - self.min_label
        self.test_Y = self.test_Y - self.min_label

        min_count = int(min(torch.LongTensor(self.train_Y.long()).bincount().data))
        self.sfk_indexset = StratifiedKFold(n_splits=c.k_fold if min_count >= c.k_fold else min_count, shuffle=False) \
            .split(self.train_X, self.train_Y)


class MTS_Dataset:
    def __init__(self,
                 root: str,
                 file_name: str):
        """
        为MTS数据集构建包含全部所需数据的对象
        :param root: 数据集根路径
        :param file_name: 数据集名称
        """
        path = root + '/' + file_name + '/' + file_name + '.mat'

        self.train_len, \
        self.test_len, \
        self.time_step_len, \
        self.feature_dim, \
        self.class_num, \
        self.train_X, \
        self.train_Y, \
        self.test_X, \
        self.test_Y = self.pre_option(path)

        self.min_label = min(self.train_Y)
        self.train_Y = self.train_Y - self.min_label
        self.test_Y = self.test_Y - self.min_label

        min_count = int(min(self.train_Y.long().bincount().data))
        self.sfk_indexset = StratifiedKFold(n_splits=c.k_fold if min_count>=c.k_fold else min_count, shuffle=False)\
            .split(self.train_X, self.train_Y)

    # 数据预处理
    def pre_option(self, path: str):
        """
        数据预处理  由于每个样本的时间步维度不同，在此使用最长的时间步作为时间步的维度，使用0进行填充
        :param path: 数据集路径
        :return: 训练集样本数量，测试集样本数量，时间步维度，通道数，分类数，训练集数据，训练集标签，测试集数据，测试集标签，测试集中时间步最长的样本列表，没有padding的训练集数据
        """
        m = loadmat(path)

        # m中是一个字典 有4个key 其中最后一个键值对存储的是数据
        x1, x2, x3, x4 = m
        data = m[x4]

        data00 = data[0][0]
        # print('data00.shape', data00.shape)  # ()  data00才到达数据的维度

        index_train = str(data.dtype).find('train\'')
        index_trainlabels = str(data.dtype).find('trainlabels')
        index_test = str(data.dtype).find('test\'')
        index_testlabels = str(data.dtype).find('testlabels')
        list = [index_test, index_train, index_testlabels, index_trainlabels]
        list = sorted(list)
        index_train = list.index(index_train)
        index_trainlabels = list.index(index_trainlabels)
        index_test = list.index(index_test)
        index_testlabels = list.index(index_testlabels)

        # [('trainlabels', 'O'), ('train', 'O'), ('testlabels', 'O'), ('test', 'O')]  O 表示数据类型为 numpy.object
        train_label = data00[index_trainlabels]
        train_data = data00[index_train]
        test_label = data00[index_testlabels]
        test_data = data00[index_test]

        train_label = train_label.squeeze()
        train_data = train_data.squeeze()
        test_label = test_label.squeeze()
        test_data = test_data.squeeze()

        train_len = train_data.shape[0]
        test_len = test_data.shape[0]
        output_len = len(tuple(set(train_label)))

        # 时间步最大值
        max_lenth = 0
        for item in train_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]
                # max_length_index = train_data.tolist().index(item.tolist())

        for item in test_data:
            item = torch.as_tensor(item).float()
            if item.shape[1] > max_lenth:
                max_lenth = item.shape[1]

        # 填充Padding  使用0进行填充
        # train_data, test_data为numpy.object 类型，不能直接对里面的numpy.ndarray进行处理
        train_dataset_with_no_paddding = []
        test_dataset_with_no_paddding = []
        train_dataset = []
        test_dataset = []
        max_length_sample_inTest = []
        for x1 in train_data:
            train_dataset_with_no_paddding.append(x1.transpose(-1, -2).tolist())
            x1 = torch.as_tensor(x1).float()
            if x1.shape[1] != max_lenth:
                padding = torch.zeros(x1.shape[0], max_lenth - x1.shape[1])
                x1 = torch.cat((x1, padding), dim=1)
            train_dataset.append(x1)

        for index, x2 in enumerate(test_data):
            test_dataset_with_no_paddding.append(x2.transpose(-1, -2).tolist())
            x2 = torch.as_tensor(x2).float()
            if x2.shape[1] != max_lenth:
                padding = torch.zeros(x2.shape[0], max_lenth - x2.shape[1])
                x2 = torch.cat((x2, padding), dim=1)
            else:
                max_length_sample_inTest.append(x2.transpose(-1, -2))
            test_dataset.append(x2)

        # 最后维度 [数据条数,时间步数最大值,时间序列维度]
        # train_dataset_with_no_paddding = torch.stack(train_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        # test_dataset_with_no_paddding = torch.stack(test_dataset_with_no_paddding, dim=0).permute(0, 2, 1)
        train_dataset = torch.stack(train_dataset, dim=0)
        test_dataset = torch.stack(test_dataset, dim=0)
        train_label = torch.Tensor(train_label)
        test_label = torch.Tensor(test_label)
        channel = test_dataset.shape[1]
        input = test_dataset.shape[-1]

        return train_len, test_len, input, channel, output_len, train_dataset, train_label, test_dataset, test_label



