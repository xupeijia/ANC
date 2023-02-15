import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import common

# 设置超参数
tau = 64
input_size = 1
hidden_size = 64
output_size = 16
learning_rate = 0.001
num_epochs = 2
batch_size = 20

# 加载数据
# df = pd.read_csv("x_and_y_linear1.csv")

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')
y1 = common.get_real_pynl_1(x)
y = -1 * y1
Sw = torch.tensor([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0])
Sw = torch.unsqueeze(Sw, 1)

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
y = torch.unsqueeze(y, 1)

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
        x = x @ Sw.to(device)
        return x


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 初始化模型
model = GRU(input_size, hidden_size, output_size, num_layers=1, seq=tau).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 开始训练
prev_val_loss = float('inf')
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = torch.transpose(inputs, 1, 0)
        labels = labels.reshape(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss = loss.item()
        with torch.no_grad():
            loss.backward(retain_graph=True)
        optimizer.step()
        del loss, outputs, inputs, labels
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = torch.transpose(inputs, 1, 0)
            labels = labels.reshape(-1, 1)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels)
    val_loss = val_loss / len(val_loader)
    print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, num_epochs, train_loss, val_loss.item()))
    if val_loss >= prev_val_loss:
        break
    prev_val_loss = val_loss
    torch.save(model.state_dict(), "best_model.pth")


# 测试
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
test_loss = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = torch.transpose(inputs, 1, 0)
    with torch.no_grad():
        outputs = model(inputs)
        test_loss += criterion(outputs, labels)
        test_loss /= len(test_loader)
        # print("test loss: ", test_loss)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    inputs = torch.transpose(inputs, 1, 0)
    with torch.no_grad():
        prediction = model(inputs)
        predictions = predictions + prediction.tolist()

# 绘制预测结果
plt.subplot(2, 1, 1)
plt.plot(y_test, label='Original data')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
