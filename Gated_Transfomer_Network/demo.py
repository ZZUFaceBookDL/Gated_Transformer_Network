# @Time    : 2021/09/17 14:44
# @Author  : SY.M
# @FileName: demo.py


import torch
from torch.utils.data.dataloader import DataLoader
from data_process.create_dataset import Global_Dataset
from module.transformer import Transformer
from torch.optim import Adam

# hyper-parameters setting
EPOCH = 10
BATCH_SIZE = 20
LR = 1e-4
d_model = 512
d_hidden = 2048
q = 8
v = 8
h = 4
N = 4

# 0. Initialize train&test data parameters
train_data = torch.rand(100, 16, 128)
train_label = torch.randint(low=0, high=5, size=(100,))
test_data = torch.rand(100, 16, 128)
test_label = torch.randint(low=0, high=5, size=(100,))

# 1. Create train&test dataset
train_dataset = Global_Dataset(X=train_data, Y=train_label)
test_dataset = Global_Dataset(X=test_data, Y=test_label)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Initialize network
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'use device:{DEVICE}')
net = Transformer(d_model=d_model, d_hidden=d_hidden, d_feature=16, d_timestep=128, q=q, v=v, h=h, N=N, class_num=5)

# 3. Create opitmizer and loss_function
optimizer = Adam(net.parameters(), lr=LR)
loss_function = torch.nn.CrossEntropyLoss()

# 4. training
for epoch_index in range(EPOCH):
    loss_sum = 0.0
    for x, y in train_dataloader:
        optimizer.zero_grad()
        pre = net(x, 'train')
        loss = loss_function(pre, y)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
    print(f'EPOCH:{epoch_index + 1}\tloss:{loss_sum}')

# 5. testing
correct = 0
total = 0
with torch.no_grad():
    net.eval()
    for x, y in test_dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pre = net(x, 'test')
        _, label_index = torch.max(y_pre.data, dim=-1)
        total += label_index.shape[0]
        correct += (label_index == y.long()).sum().item()
    print(f'Accuracy: %.2f %%' % (100 * correct / total))

