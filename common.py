import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import scipy.signal as signal


def gru_lms(x, y, lens, taps, learn_rate):
    cx = [0] * taps
    cw = [0] * taps
    cy = [0] * taps
    sx = [0] * taps
    xhx = [0] * taps
    loss = [0] * lens
    for item in range(len(x)):
        del cx[-1]
        cx = [x[item]] + cx
        cx_out = sum(i * j for i, j in zip(cx, cw))  # lms控制器的输出
        del cy[0]
        cy.append(cx_out)
        loss[item] = y[item] - load_gru(cy)
        del sx[0]
        sx.append(x[item])
        shx = load_gru(sx)
        del xhx[-1]
        xhx = [shx] + xhx
        cw_old = [learn_rate * loss[item] * i for i in xhx]
        cw = [i + j for i, j in zip(cw_old, cw)]
        # print(item)

    return cw, loss


def fx_lms(x, y, sw_est, lens, taps, learn_rate):
    cx = [0] * taps
    cw = [0] * taps
    sx = [0] * taps
    shx = [0] * taps
    xhx = [0] * taps
    loss = [0] * lens
    for item in range(len(x)):
        del cx[-1]
        cx = [x[item]] + cx
        cy = sum(i * j for i, j in zip(cx, cw))  # lms控制器的输出
        del sx[-1]  # lms控制器的输出作为次级通道的输入
        sx = [cy] + sx
        loss[item] = y[item] - sum(i * j for i, j in zip(sx, sw_est))
        del shx[-1]  # fx部分
        shx = [x[item]] + shx
        del xhx[-1]
        xhx = [sum(i * j for i, j in zip(shx, sw_est))] + xhx
        cw_old = [learn_rate * loss[item] * i for i in xhx]
        cw = [i + j for i, j in zip(cw_old, cw)]

    return cw, loss


# 初始化模型
def load_gru(x):
    # 数据预处理
    x = torch.tensor(x, dtype=torch.float32)
    x = torch.reshape(x, (1, -1, 1))
    x_dataset = TensorDataset(x, x)
    x_loader = DataLoader(x_dataset, batch_size=1, shuffle=False)

    # 定义模型
    class GRU(nn.Module):
        def __init__(self, input_num, hidden_num, output_num, num_layers, seq):
            super(GRU, self).__init__()
            self.gru = nn.GRU(input_num, hidden_num, num_layers)
            self.fc = nn.Linear(hidden_num * seq, output_num)

        def forward(self, x):  # x(batch, seq, feature/input_num)->(seq, batch, feature/input_num)
            x, _ = self.gru(x)  # (seq, batch, 64)
            x = torch.transpose(x, 1, 0)  # 调整为 (batch, seq, hidden)
            b, s, h = x.shape  # (batch, seq*hidden)
            x = x.reshape(b, s * h)  # (batch, seq * hidden)
            x = self.fc(x)
            return x

    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = GRU(1, 64, 1, num_layers=2, seq=64).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    for inputs, _ in x_loader:
        inputs = inputs.to(device)
        inputs = torch.transpose(inputs, 1, 0)
        with torch.no_grad():
            output = model(inputs).item()

    return output


# 读取初始数据
def read_vibration(file_name):
    df = pd.read_table(file_name, skiprows=5)
    time = df.iloc[:, 0].values
    vibration = df.iloc[:, 1].values
    return time, vibration


# 获取主路径、次路径的线性和非线性真实输出
def get_real_pyl_1(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(12), inputs))
    for i in range(12, len(x)):
        y[i - 12] = 0.8 * x[i - 6] + 0.6 * x[i - 7] - 0.2 * x[i - 8] - 0.5 * x[i - 9] - 0.1 * x[i - 10] + 0.4 * x[
            i - 11] - 0.05 * x[i - 12]
    return y


def get_real_pynl_1(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(12), inputs))
    for i in range(12, len(x)):
        y[i - 12] = 0.8 * x[i - 6] + 0.6 * x[i - 7] - 0.2 * x[i - 8] - 0.5 * x[i - 9] - 0.1 * x[i - 10] + 0.4 * x[
            i - 11] - 0.05 * x[i - 12] + 2.5 * x[i - 6] ** 3
    return y


def get_real_syl_1(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(10), inputs))
    for i in range(10, len(x)):
        y[i - 5] = 0.9 * x[i - 2] + 0.6 * x[i - 3] - 0.1 * x[i - 4] - 0.4 * x[i - 5] - 0.1 * x[i - 6] + 0.2 * x[
            i - 7] + 0.1 * x[i - 8] + 0.01 * x[i - 9] + 0.001 * x[i - 10]
    return y


def get_real_synl_1(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(10), inputs))
    for i in range(10, len(x)):
        y[i - 5] = 0.9 * x[i - 2] + 0.6 * x[i - 3] - 0.1 * x[i - 4] - 0.4 * x[i - 5] - 0.1 * x[i - 6] + 0.2 * x[
            i - 7] + 0.1 * x[i - 8] + 0.01 * x[i - 9] + 0.001 * x[i - 10] + 2.5 * x[i - 6] ** 3
    return y


def get_real_pynl_2(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(3), inputs))
    for i in range(3, len(x)):
        y[i - 3] = x[i] + 0.8 * x[i - 1] + 0.3 * x[i - 2] + 0.4 * x[i - 3] - 0.8 * x[i] * x[i - 1] + 0.9 * x[i] * x[
            i - 2] + 0.7 * x[i] * x[i - 3]
    return y


def get_real_synl_2(inputs):
    y = np.zeros(len(inputs))
    x = np.hstack((np.zeros(2), inputs))
    for i in range(2, len(x)):
        y[i - 2] = x[i] + 0.2 * x[i - 1] + 0.05 * x[i - 2]
    return y


# 根据tau生成数据features和y
def create_data(x, y, tau, data_dim, reverse):
    lens = len(x)
    if data_dim == 2:
        features = torch.zeros((lens - tau, tau))  # (batch, feature/seq)
    else:
        features = torch.zeros((lens - tau, tau, 1))  # (batch, seq, feature)

    if reverse:
        for i in range(tau):
            features[:, i] = x[i: lens - tau + i]
    else:
        for i in range(tau):
            features[:, i] = x[tau - i: lens - i]
    y = y[tau:]
    y = torch.unsqueeze(y, 1)  # (batch, feature)

    return features, y


# 按照比例划分数据集
def split_data(x, y, rate, batch_size, train_data_shuffle):
    split_index = int(x.shape[0] * rate)
    x_train, x_val_test = x[:split_index], x[split_index:]
    y_train, y_val_test = y[:split_index], y[split_index:]
    val_test_index = int(x_val_test.shape[0] * 0.5)
    x_val, x_test = x_val_test[:val_test_index], x_val_test[val_test_index:]
    y_val, y_test = y_val_test[:val_test_index], y_val_test[val_test_index:]

    # 定义数据集
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    # 定义数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_data_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_test
