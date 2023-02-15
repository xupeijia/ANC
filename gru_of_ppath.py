import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import common

# 设置超参数
tau = 16
input_size = 1
hidden_size = 64
output_size = 16
learning_rate = 0.001
num_epochs = 5
batch_size = 4

# 加载数据
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')
y = common.get_real_y_2(x)
x = np.array(x)

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

y = y[tau:]

# 数据集划分
split_index = int(features.shape[0] * 0.7)
x_train, x_val_test = features[:split_index], features[split_index:]
y_train, y_val_test = y[:split_index], y[split_index:]
val_test_index = int(x_val_test.shape[0] * 0.5)
x_val, x_test = x_val_test[:val_test_index], x_val_test[val_test_index:]
y_val, y_test = y_val_test[:val_test_index], y_val_test[val_test_index:]

# 定义数据集
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)
test_dataset = TensorDataset(x_test, y_test)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义模型
class GRU(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, num_layers, seq):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_num, hidden_num, num_layers)
        self.fc = nn.Linear(hidden_num * seq, output_num)

    def forward(self, x):
        x, _ = self.gru(x)  # (seq, batch, hidden)
        x = torch.transpose(x, 1, 0)  # 调整为 (batch, seq, hidden)
        b, s, h = x.shape
        x = x.reshape(b, s * h)  # (batch, seq * hidden)
        x = self.fc(x)
        return x


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 初始化模型
model = GRU(input_size, hidden_size, output_size, num_layers=2, seq=tau).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# temp_set = np.zeros(batch_size, 1)
# 开始训练
prev_val_loss = float('inf')
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs(batch, seq, input_nums)->GRU输入需要的(seq, batch, input_nums)
        inputs = torch.transpose(inputs, 1, 0)
        # labels(seq)->GRU输出需要的(seq, 1)
        labels = labels.reshape(-1, 1)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # outputs = outputs.numpy()

        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
    # 每个epoch结束后使用验证集检查模型性能
    val_loss = 0
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.transpose(inputs, 1, 0)
        labels = labels.reshape(-1, 1)
        outputs = model(inputs)
        val_loss += criterion(outputs, labels)

    val_loss = val_loss / len(val_loader)
    print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, num_epochs, loss.item(), val_loss.item()))

    # 保存最优模型
    if val_loss < prev_val_loss:
        prev_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")

# 测试
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = torch.transpose(inputs, 1, 0)
    labels = labels.reshape(-1, 1)
    with torch.no_grad():
        outputs = model(inputs)
        test_loss += criterion(outputs, labels)
        test_loss /= len(test_loader)
        print("test loss: ", test_loss)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    inputs = torch.transpose(inputs, 1, 0)
    with torch.no_grad():
        prediction = model(inputs)
        predictions = predictions + prediction.tolist()

# 绘制预测结果
plt.plot(y_test, label='Original data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
