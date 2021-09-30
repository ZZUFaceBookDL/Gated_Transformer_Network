# Gated Transformer Networks for Multivariate Time Serise Classification
---
***GTN: An improved deep learning network based on Transformer for multivariate time series classification tasks.Use Gating mechanism to extract features of both channel-wise and timestep-wise***

[Gated Transformer Networks for Multivariate Time Series Classification](https://arxiv.org/abs/2103.14438) / [arXiv:2103.14438](https://arxiv.org/abs/2103.14438)

>Deep learning model (primarily convolutional networks and LSTM) for time series classification has been studied broadly by the community with the wide applications in different domains like healthcare, finance, industrial engineering and IoT. Meanwhile, Transformer Networks recently achieved frontier performance on various natural language processing and computer vision tasks. In this work, we explored a simple extension of the current Transformer Networks with gating, named Gated Transformer Networks (GTN) for the multivariate time series classification problem. With the gating that merges two towers of Transformer which model the channel-wise and step-wise correlations respectively, we show how GTN is naturally and effectively suitable for the multivariate time series classification task. We conduct comprehensive experiments on thirteen dataset with full ablation study. Our results show that GTN is able to achieve competing results with current state-of-the-art deep learning models. We also explored the attention map for the natural interpretability of GTN on time series modeling. Our preliminary results provide a strong baseline for the Transformer Networks on multivariate time series classification task and grounds the foundation for future research.

Please cite as:
```bibtex
@article{liu2021gated,
  title={Gated Transformer Networks for Multivariate Time Series Classification},
  author={Liu, Minghao and Ren, Shengqi and Ma, Siyuan and Jiao, Jiahui and Chen, Yizhou and Wang, Zhiguang and Song, Wei},
  journal={arXiv preprint arXiv:2103.14438},
  year={2021}
}
```

## Network Strcuture
![GTN Network Structure](https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/image/GTN.png)

## Result
- [MTS Archive](https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/result/MTS.xlsx)
- [UEA Archive](https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/result/UEA.xlsx)

## Code
### [`transformer.py`](https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/module/for_MTS/transformer.py)

## Requirements
- Python
- Numpy
- torch
- sklearn
- scipy

## Demo
### [`demo.py`](https://github.com/ZZUFaceBookDL/Gated_Transformer_Network/blob/master/Gated_Transfomer_Network/demo.py)
```
import torch
from torch.utils.data.dataloader import DataLoader
from data_process.create_dataset import Global_Dataset
from module.transformer import Transformer
from torch.optim import Adam

# 0. load data
[...]

# 1. create dataset and dataloader
train_dataset = Global_Dataset(X=train_data, Y=train_label)
test_dataset = Global_Dataset(X=test_data, Y=test_label)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. build classifier
net = Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=16, d_timestep=128, q=q, v=v, h=h, N=N, class_num=5)

# 3. build loss function and optimizer
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. train and test
[...]
```

## Contact
Welcome to communicate, criticize and correct mistakes.
Please emailing to `masiyuan007@qq.com`
