import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import utils
from network import BpNet
import path_function

# 设置超参数
tau = 160
input_size = 160
hidden_size = 320
output_size = 3
learning_rate = 0.001
num_epochs = 5
batch_size = 32

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 次级路径
sw = torch.tensor([1, 0.2, 0.05], dtype=torch.float32, device=device)
# sw = torch.tensor([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0], dtype=torch.float32,
#                   device=device)
is_divided = True
is_for_crn = False
is_for_bp = True

# 加载数据
# df = pd.read_csv("x_and_y_linear1.csv")
# df = pd.read_csv("x_and_y_total5.csv")
# x = df.iloc[:, 0].values
# y = df.iloc[:, 1].values

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = utils.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')

# 降采样数据到16kHz
sig = utils.resample_and_filter(x, 50000, 16000, 8000)
y = -1 * path_function.get_real_pynl_2(sig)

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
sig = scaler.fit_transform(sig.reshape(-1, 1))
sig = torch.tensor(sig, dtype=torch.float32)
sig = torch.squeeze(sig)
y = torch.tensor(y, dtype=torch.float32)

# 按照tua设置延迟
features, y = utils.create_data(sig, y, tau, 2, False)

# 数据集划分
train_loader, val_loader, test_loader, y_test = utils.split_data(features, y, 0.7, batch_size, True)

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
        # 传入次级通道
        outputs = path_function.sp_fun(outputs, sw, is_divided, is_for_crn, is_for_bp, device)
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
        outputs = path_function.sp_fun(outputs, sw, is_divided, is_for_crn, is_for_bp, device)
        val_loss += criterion(outputs, labels)

    val_loss = val_loss / len(val_loader)
    print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, num_epochs, loss.item(), val_loss.item()))

    # 保存最优模型
    if val_loss < prev_val_loss:
        prev_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_bp.pth")

# 测试
model.load_state_dict(torch.load("best_model_bp.pth"))
model.eval()
test_loss = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        outputs = path_function.sp_fun(outputs, sw, is_divided, is_for_crn, is_for_bp, device)
        test_loss += criterion(outputs, labels)
        test_loss /= len(test_loader)
        # print("test loss: ", test_loss)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    with torch.no_grad():
        prediction = model(inputs)
        prediction = path_function.sp_fun(prediction, sw, is_divided, is_for_crn, is_for_bp, device)
        predictions = predictions + prediction.tolist()

# 绘制预测结果
plt.subplot(2, 1, 1)
plt.plot(y_test, label='Original data')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
