import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import common

# 设置超参数
tau = 100
input_size = 100
hidden_size = 1000
output_size = 16
learning_rate = 0.001
num_epochs = 5
batch_size = 50

# 加载数据
# df = pd.read_csv("x_and_y_linear1.csv")
# df = pd.read_csv("x_and_y_total5.csv")
# x = df.iloc[:, 0].values
# y = df.iloc[:, 1].values

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
x = torch.squeeze(x)
y = torch.tensor(y, dtype=torch.float32)

# 按照tua设置延迟
features, y = common.create_data(x, y, tau, 2, False)

# 数据集划分
train_loader, val_loader, test_loader, y_test = common.split_data(features, y, 0.7, batch_size, True)


# 定义网络
class BpNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(BpNet, self).__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(input_num, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, output_num),
        )

    def forward(self, x):
        x = self.net1(x)
        x = x @ Sw.to(device)
        return x


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 初始化模型
model = BpNet(input_size, hidden_size, output_size, ).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# 开始训练
prev_val_loss = float('inf')
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
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
        outputs = model(inputs)
        val_loss += criterion(outputs, labels)

    val_loss = val_loss / len(val_loader)
    print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, num_epochs, loss.item(), val_loss.item()))

    # 保存最优模型
    if val_loss < prev_val_loss:
        prev_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_sp.pth")

# 测试
model.load_state_dict(torch.load("best_model_sp.pth"))
model.eval()
test_loss = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        test_loss += criterion(outputs, labels)
        test_loss /= len(test_loader)
        # print("test loss: ", test_loss)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
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
