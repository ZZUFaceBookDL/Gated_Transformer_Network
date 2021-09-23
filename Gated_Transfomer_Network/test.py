import numpy as np

from module.for_MTS.transformer import Transformer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import torch
import seaborn as sns
import random
from tqdm import tqdm

dataset = []
labels = []
for i in range(50):
    start = random.randint(10, 21)
    swing = random.random() * 0.3 + 0.3
    val1 = [torch.cos(t) + 1 for t in torch.linspace(start=0, end=math.pi * 2, steps=50)]
    val2 = [torch.sin(t) + 1 for t in torch.linspace(start=0, end=math.pi * 2, steps=50)]
    val3 = [torch.exp(t) - 0.5 for t in torch.linspace(start=0, end=1, steps=50)]

    noise = [0 for _ in range(start)] + [swing for _ in range(5)] + [0 for _ in range(45 - start)]
    val1_noise = np.add(val1, noise)

    sample1 = torch.Tensor([val1, val2, val3])
    sample2 = torch.Tensor([val1_noise, val2, val3])
    dataset.extend([sample1, sample2])
    labels.extend(torch.LongTensor([[0], [1]]))

# print(dataset)

dataset = torch.stack(dataset)
labels = torch.stack(labels)
print(dataset.shape)

val1 = [torch.cos(t) + 1 for t in torch.linspace(start=0, end=math.pi * 2, steps=50)]
val2 = [torch.sin(t) + 1 for t in torch.linspace(start=0, end=math.pi * 2, steps=50)]
val3 = [torch.exp(t) - 0.5 for t in torch.linspace(start=0, end=1, steps=50)]
noise = [0 for _ in range(15)] + [0.3 for _ in range(5)] + [0 for _ in range(30)]
val1_noise = np.add(val1, noise)

fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.plot(val1, color='red', label='val1')
ax1.plot(val2, color='green', label='val2')
ax1.plot(val3, color='blue', label='val3')
ax2.plot(val1_noise, color='red', label='val1')
ax2.plot(val2, color='green', label='val2')
ax2.plot(val3, color='blue', label='val3')
plt.legend(loc='best')
plt.show()
plt.close()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = Transformer(q=8, v=8, h=4, N=4, d_model=512, d_hidden=2048,
                  d_feature=3, d_timestep=50, class_num=2, dropout=0).to(DEVICE)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_function = torch.nn.CrossEntropyLoss()

net.train()
score_input = None
score_channel = None
EPOCH = 50
sign = tqdm(EPOCH)
for index in range(EPOCH):
    for sample, label in zip(dataset, labels):
        sample = sample.unsqueeze(0)
        optimizer.zero_grad()
        output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate = net(sample.to(DEVICE),
                                                                                                     'train')
        loss = loss_function(output, label.to(DEVICE))
        loss.backward()
        optimizer.step()
        sign.update()

score_input_ave = None
score_channel_ave = None
score_input = torch.mean(score_input, dim=0, keepdim=True).squeeze()
score_channel = torch.mean(score_channel, dim=0, keepdim=True).squeeze()
score_input = torch.mean(score_input, dim=0, keepdim=True)
score_channel = torch.mean(score_channel, dim=0, keepdim=True)
score_input = torch.softmax(score_input, dim=-1)
score_channel = torch.softmax(score_channel, dim=-1)

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # 设置figure_size尺寸
plt.subplot(121)
sns.heatmap(score_input.detach().numpy(), cmap="YlGnBu", vmin=0)

plt.subplot(122)
sns.heatmap(score_channel.detach().numpy(), cmap="YlGnBu", vmin=0)

plt.show()
plt.close()
