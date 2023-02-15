import torch
import torch.nn as nn
import numpy as np


def sp_fun(inputs, sw, is_divided, is_for_crn, is_for_bp, device):
    # 判断次级通道是否被拆成了线性+非线性的方式
    if is_divided:
        # 如果拆分则先通过非线性部分传函
        # tanh()和tanh_()不一样，tanh_()是原地更改输入数据，而前者是直接返回个新的tensor
        inputs = 3.3 * torch.tanh(0.3 * inputs)

    # 判断传函是否用于crn网络模型
    if is_for_crn:
        # inputs = tensor(1,48000)
        inputs = torch.unsqueeze(inputs, 0)
        sw = sw.unsqueeze(0).unsqueeze(0)
        conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding='same', bias=False, device=device)
        conv.weight.data = sw
        outputs = conv(inputs)
        # outputs = tensor(1,48000)
        outputs = torch.squeeze(outputs, 0)
        return outputs
    # 判断传函是否用于bp网络模型
    elif is_for_bp:
        # sw = torch.flip(sw, dims=[0])
        sw = sw.unsqueeze(1)
        outputs = inputs @ sw
        return outputs
    # 属于gru网络模型
    else:
        sw = sw.unsqueeze(1)
        outputs = inputs @ sw
        return outputs


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
        y[i - 5] = 2.5 * x[i - 6] ** 3 + 0.9 * x[i - 2] + 0.6 * x[i - 3] - 0.1 * x[i - 4] - 0.4 * x[i - 5] - 0.1 * x[
            i - 6] + 0.2 * x[i - 7] + 0.1 * x[i - 8] + 0.01 * x[i - 9] + 0.001 * x[i - 10]
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
