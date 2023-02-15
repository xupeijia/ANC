import d2l.torch as d2l
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

import common
import pandas as pd

# 加载数据
# df = pd.read_csv("x_and_y_linear1.csv")
df = pd.read_csv("x_and_y_total5.csv")
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(x.reshape(-1, 1))
x = np.squeeze(x, 1)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
T = len(x)

# 设置超参数
tau = 1000
epochs = 600
batch_size = 50
input_size = 1000
hidden_size = 10
output_size = 1
n_train = int(T * 0.7)
lr = 0.001

features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = y[tau:].reshape((-1, 1))

# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(input_size, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, output_size))
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, epochs, lr)

# 保存网络
PATH = './bp_net.path'
torch.save(net.state_dict(), PATH)

# from torchsummary import summary
# summary(net, (16, 32, 1))

# params = list(net.parameters())
# np.set_printoptions(suppress=True)
# print(params)

for name, parameters in net.state_dict().items():
    if "weight" in name:
        print(name, ':', parameters.detach().numpy())
    if "bias" in name:
        print(name, ':', parameters.detach().numpy())
