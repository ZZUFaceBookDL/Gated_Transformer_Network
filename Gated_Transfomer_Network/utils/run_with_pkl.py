# @Time    : 2021/07/16 28:11
# @Author  : SY.M
# @FileName: run_with_pkl.py

from torch.utils.data import DataLoader
import torch

from utils.random_seed import setup_seed
from config.path_config import MTS, UEA, UCR
from data_process.create_dataset import get_MyData, Global_Dataset

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup_seed(30)


def run_with_pkl():
    """
    使用训练好的且存储为.pkl格式的模型在默认测试集上进行测试
    :return: None
    """
    print("\033[0;34m%s\033[0m" % 'Input the path of .pkl file:')
    path = input('')
    net = torch.load(path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    file_name = path.split('\\')[-1]
    archive_name = path.split('\\')[-3].split('_')[0]
    dataset_name = file_name.split(' ')[0]

    root = None
    if archive_name == 'MTS':
        root = MTS.root
    elif archive_name == 'UEA':
        root = UEA.root
    elif archive_name == 'UCR':
        root = UCR.root
    assert root is not None, 'Path error!'

    all_data = get_MyData(root=root, file_name=dataset_name, archive_name=archive_name)
    test_dataset = Global_Dataset(X=all_data.test_X, Y=all_data.test_Y)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        net.eval()

        # test on test dataset
        correct = 0
        total = 0
        for x, y in test_dataloader:
            pre = net(x.to(DEVICE), 'test')
            _, pre_index = torch.max(pre.data, dim=-1)
            total += pre.shape[0]
            correct += torch.sum(torch.eq(pre_index, y.long().to(DEVICE))).item()
        accuracy_on_test = round(correct / total, 4) * 100
        print(f'accuracy on test:{accuracy_on_test}%')
