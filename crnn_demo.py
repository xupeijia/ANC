import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import common
import create_data
import crnn_lstm

# 设置超参数
num_epochs = 30

# 导入原始数据
filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')

# 降采样数据到16kHz
sig = create_data.resample_and_filter(x, 50000, 16000, 8000)
y = -1 * common.get_real_pynl_2(sig)

# 分割数据
sig_segment = create_data.create_segment(sig, 10, 3, 1, 16000)
y_segment = create_data.create_segment(y, 10, 3, 1, 16000)

# 把训练数据进行stft
sig_tensor = create_data.stft(sig_segment, 320, 160)

# dataloader
train_loader, val_loader, test_loader, y_test = common.split_data(sig_tensor, y_segment, 0.75, 1, True)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 次级路径传递函数
# Sw = torch.tensor([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0], device=device)
Sw = np.array([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0])

# 初始化模型
model = crnn_lstm.Net().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
# criterion = LossFunction()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

# 开始训练
prev_val_loss = float('inf')
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 清空梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 进行i_stft
        outputs = create_data.i_stft(outputs, 320, 160)
        # 传入次级路径
        outputs = 3.3 * torch.tanh_(0.3 * outputs)
        outputs = create_data.sp_fun(outputs)
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
        # 进行i_stft
        outputs = create_data.i_stft(outputs, 320, 160)
        # 传入次级路径
        outputs = 3.3 * torch.tanh_(0.3 * outputs)
        outputs = create_data.sp_fun(outputs)
        loss = criterion(outputs, labels)
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
    outputs = model(inputs)
    # 进行i_stft
    outputs = create_data.i_stft(outputs, 320, 160)
    # 传入次级路径
    outputs = 3.3 * torch.tanh_(0.3 * outputs)
    outputs = create_data.sp_fun(outputs)
    test_loss += criterion(outputs, labels)
    test_loss /= len(test_loader)
    # print("test loss: ", test_loss)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    with torch.no_grad():
        prediction = model(inputs)
        # 输出进行istft
        prediction = create_data.i_stft(prediction, 320, 160)
        # 经过次级路径
        prediction = 3.3 * torch.tanh_(0.3 * prediction)
        prediction = create_data.sp_fun(prediction)
        predictions = predictions + prediction.tolist()

# 绘制预测结果
y_test = np.array(y_test.cpu()).squeeze(0)
predictions = np.array(predictions).squeeze(0)
plt.subplot(2, 1, 1)
plt.plot(y_test, label='Original data')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
