import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 超参数
tau = 64
batch_size = 64
n_train = 6000
num_epochs = 5

# 加载数据
# df = pd.read_csv("x_and_y_linear1.csv")
df = pd.read_csv("x_and_y_total1.csv")
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(x.reshape(-1, 1))
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
T = len(x)

# 按照tua设置延迟
features = torch.zeros((T - tau, tau, 1))  # （batch, seq, feature）
for i in range(tau):
    features[:, i] = x[i: T - tau + i]

labels = y[tau:]

# 划分数据集：只有前n_train个样本用于训练
data_arrays = (features[:n_train], labels[:n_train])
dataset = data.TensorDataset(*data_arrays)
train_iter = data.DataLoader(dataset, batch_size, shuffle=True)  # 打乱顺序

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 是否采用GPU


# 定义模型
class GRU_REG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq):
        super(GRU_REG, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # rnn

        for name, param in self.gru.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

        self.reg = nn.Linear(hidden_size * seq, output_size)  # 回归

    def forward(self, x):
        x, _ = self.gru(x)  # (seq, batch, hidden)
        x = torch.transpose(x, 1, 0)  # 调整为 (batch, seq, hidden)
        b, s, h = x.shape
        x = x.reshape(b, s * h)  # 转换成线性层的输入格式
        x = self.reg(x)
        return x


# 模型的初始化
model = GRU_REG(input_size=1, hidden_size=64, output_size=1, num_layers=2, seq=tau).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模型的训练
hist = np.zeros(num_epochs)  # 用来记录每一个epoch的误差

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for num, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        X = torch.transpose(X, 1, 0)  # 将batch和seq的维度置换一下
        y = y.reshape(-1, 1)
        optimizer.zero_grad()

        # forward
        output = model(X)

        # backward
        loss = criterion(output, y)
        loss.backward()
        # optimize
        optimizer.step()

    with torch.no_grad():
        epoch_loss += loss.detach().cpu().item()  # 由于我是在GPU进行训练的，因此这里将计算出的损失脱离GPU
    print(f'epoch {epoch + 1}, loss {epoch_loss:f}')

model.eval()
y_pred = []
for i in range(len(features) // batch_size):
    x_test = features[i * batch_size:(i + 1) * batch_size, ::]
    x_test = torch.transpose(x_test, 1, 0)
    x_test = x_test.to(device)
    y_pred += (model(x_test).detach().cpu())

y_pred = [i.item() for i in y_pred]

plt.plot(labels, label='true')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
