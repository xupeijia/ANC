import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import utils
import path_function
from network import CrnNet

# 设置超参数
num_epochs = 10

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 次级路径传递函数
sw = torch.tensor([0.05, 0.2, 1], dtype=torch.float32, device=device)
# sw = torch.tensor([0.001, 0.01, 0.1, 0.2, -0.1, -0.4, -0.1, 0.6, 0.9, 0, 0], dtype=torch.float32, device=device)
is_divided = True
is_for_crn = True
is_for_bp = False

# 导入原始数据
filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = utils.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')

# 降采样数据到16kHz
sig = utils.resample_and_filter(x, 50000, 16000, 8000)
y = -1 * path_function.get_real_pynl_2(sig)

# 分割数据
sig_segment = utils.create_segment(sig, 10, 3, 1, 16000, device)
y_segment = utils.create_segment(y, 10, 3, 1, 16000, device)

# 把训练数据进行stft
sig_tensor = utils.stft(sig_segment, 320, 160, device)

# dataloader
train_loader, val_loader, test_loader, y_test = utils.split_data(sig_tensor, y_segment, 0.75, 1, True)

# 初始化模型
model = CrnNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
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
        outputs = utils.i_stft(outputs, 320, 160, device)
        # 传入次级路径
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
        # 进行i_stft
        outputs = utils.i_stft(outputs, 320, 160, device)
        # 传入次级路径
        outputs = path_function.sp_fun(outputs, sw, is_divided, is_for_crn, is_for_bp, device)
        loss = criterion(outputs, labels)
        val_loss += criterion(outputs, labels)

    val_loss = val_loss / len(val_loader)
    print("Epoch: {}/{}, Loss: {}, Val Loss: {}".format(epoch + 1, num_epochs, loss.item(), val_loss.item()))

    # 保存最优模型
    if val_loss < prev_val_loss:
        prev_val_loss = val_loss
        torch.save(model.state_dict(), "best_model_crn.pth")

# 测试
model.load_state_dict(torch.load("best_model_crn.pth"))
model.eval()
test_loss = 0
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        # 进行i_stft
        outputs = utils.i_stft(outputs, 320, 160, device)
        # 传入次级路径
        outputs = path_function.sp_fun(outputs, sw, is_divided, is_for_crn, is_for_bp, device)
        test_loss += criterion(outputs, labels)
        test_loss /= len(test_loader)

# 预测
predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)
    with torch.no_grad():
        prediction = model(inputs)
        # 输出进行i_stft
        prediction = utils.i_stft(prediction, 320, 160, device)
        # 经过次级路径
        prediction = path_function.sp_fun(prediction, sw, is_divided, is_for_crn, is_for_bp, device)
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
