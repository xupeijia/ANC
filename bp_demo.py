import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

# filename = 'Siemens-MAGNETOM-HEAD.txt'
from ancDemo import common

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-HEAD.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-Waist-1.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-Waist-2.txt'


# 设置超参数
tau = 1000
LAG = 64
epochs = 5
batch_size = 1
input_size = 1000
hidden_size = 100
output_size = 1
lr = 0.001

Sw = np.array([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0])

# 加载数据
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')
y = common.get_real_pynl_1(x)

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
x = scaler.fit_transform(x.reshape(-1, 1))
x = torch.tensor(x, dtype=torch.float32)
x = torch.squeeze(x)
y = torch.tensor(y, dtype=torch.float32)

# 按照tua设置延迟
features, y = common.create_data(x, y, tau, 2, False)

# 数据集划分
train_loader, val_loader, test_loader, y_test = common.split_data(features, y, 0.7, batch_size, False)


# 定义网络
class BpNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(BpNet, self).__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(input_num, hidden_num),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_num, output_num),
        )

    def forward(self, x):
        x = self.net1(x)
        return x


# 定义优化器和损失函数
model = BpNet(input_size, hidden_size, output_size)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr)

# 定义训练方法
# prev_val_loss = float('inf')
# for epoch in range(epochs):
#     # 训练模型
#     # bp_net.train()
#     running_loss = 0.0
#     sz_in: [float] = [0] * 16
#     for batch_index, data in enumerate(train_loader, 0):
#         # 获取输入
#         inputs, labels = data
#
#         # 初始化梯度
#         optimizer.zero_grad()
#
#         # 通过网络获得输出
#         outputs = model(inputs)
#         outputs_item = outputs.item()
#         del sz_in[-1]
#         sz_in = [outputs_item] + sz_in
#         sz_out = sum(np.multiply(np.array(sz_in), Sw))
#         outputs = torch.tensor(sz_out, dtype=torch.float32).reshape(1)
#         outputs = Variable(outputs, requires_grad=True)
#         # 计算损失
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#     sz_in_2: [float] = [0] * 16
#     # 每个epoch结束后使用验证集检查模型性能
#     val_loss = 0
#     for inputs, labels in val_loader:
#         outputs = model(inputs)
#         outputs_item = outputs.item()
#         del sz_in_2[-1]
#         sz_in_2 = [outputs_item] + sz_in_2
#         sz_out = sum(np.multiply(np.array(sz_in_2), Sw))
#         outputs = torch.tensor(sz_out, dtype=torch.float32).reshape(1)
#         outputs = Variable(outputs, requires_grad=True)
#         val_loss += loss_function(outputs, labels)
#
#     val_loss = val_loss / len(val_loader)
#     print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, epochs, loss.item(), val_loss.item()))
#
#     # 保存最优模型
#     if val_loss < prev_val_loss:
#         prev_val_loss = val_loss
#         torch.save(model.state_dict(), "best_bp_model.pth")

# 测试
model.load_state_dict(torch.load("best_bp_model.pth"))
model.eval()
# test_loss = 0
# sz_in_test: [float] = [0] * 16
# for inputs, labels in test_loader:
#     with torch.no_grad():
#         outputs = model(inputs)
#         outputs_item = outputs.item()
#         del sz_in_test[-1]
#         sz_in_test = [outputs_item] + sz_in_test
#         sz_out = sum(np.multiply(np.array(sz_in_test), Sw))
#         outputs = torch.tensor(sz_out, dtype=torch.float32).reshape(1)
#         outputs = Variable(outputs, requires_grad=True)
#         test_loss += loss_function(outputs, labels)
#         test_loss /= len(test_loader)
#         print("test loss: ", test_loss)

# 预测
predictions = []
sz_in_pred: [float] = [0] * 16
for inputs, _ in test_loader:
    with torch.no_grad():
        prediction = model(inputs)
        prediction_item = prediction.item()
        del sz_in_pred[-1]
        sz_in_pred = [prediction_item] + sz_in_pred
        sz_out = sum(np.multiply(np.array(sz_in_pred), Sw))
        outputs = torch.tensor(sz_out, dtype=torch.float32).reshape(1)
        outputs = Variable(outputs, requires_grad=True)
        predictions = predictions + prediction.tolist()

# 绘制预测结果
plt.plot(y_test, label='Original data')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()