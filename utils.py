import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import scipy.signal as signal


# 读取初始数据
def read_vibration(file_name, start_line, sep):
    df = pd.read_table(file_name, skiprows=start_line, sep=sep)
    time = df.iloc[:, 0].values
    vibration = df.iloc[:, 1].values
    return time, vibration


# 根据tau生成数据features和y
def create_data(x, y, tau, data_dim, reverse):
    lens = len(x)
    if data_dim == 2:
        features = torch.zeros((lens - tau, tau))  # (batch, feature/seq)
    else:
        features = torch.zeros((lens - tau, tau, 1))  # (batch, seq, feature)

    if reverse:
        for i in range(tau):
            features[:, i] = x[tau - i: lens - i]
    else:
        for i in range(tau):
            features[:, i] = x[i: lens - tau + i]
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


# 对数据进行降采样到16k
def resample_and_filter(sig, sample_rate_old, sample_rate_new, cutoff_freq):
    # 滤波
    wn = cutoff_freq / (sample_rate_old / 2)
    b, a = signal.butter(5, wn, 'lowpass')
    filtered_input = signal.lfilter(b, a, sig)
    # 降采样
    resampled_input = signal.resample_poly(filtered_input, sample_rate_new, sample_rate_old)
    return resampled_input


def create_segment(sig, time, seg_len, shift_len, sample_rate, device):
    # 切分信号为(batch_size, segment)
    signals = np.array([sig[i: i + seg_len * sample_rate] for i in
                        range(0, time * sample_rate - seg_len * sample_rate + 1, shift_len * sample_rate)])
    signals = torch.as_tensor(signals, dtype=torch.float32, device=device)
    return signals


def create_segment2(sig, seg_len, shift_len, sample_rate, device):
    return 0


def stft(sig, window_size, shift_size, device):
    # 把序列(batch_size, segment)转成网络输入的tensor(batch_size, feature_maps, time_steps, frequency)
    window = torch.hann_window(window_size, device=device)
    stft_outputs = torch.stft(sig, n_fft=window_size, hop_length=shift_size, win_length=window_size, window=window,
                              return_complex=True)
    stft_outputs = torch.transpose(stft_outputs, 1, 2)
    real_part, imag_part = torch.real(stft_outputs), torch.imag(stft_outputs)
    return torch.stack([real_part, imag_part], dim=1)


def i_stft(out_tensor, window_size, shift_size, device):
    # 把网络输出的tensor(batch_size, feature_maps, time_steps, frequency)转成序列(batch_size, segment)
    complex_stft_outputs = out_tensor[:, 0, :, :] + 1j * out_tensor[:, 1, :, :]  # 8*299*161
    complex_stft_outputs = torch.transpose(complex_stft_outputs, 1, 2)
    window = torch.hann_window(window_size, device=device)
    inverse_stft_outputs = torch.istft(complex_stft_outputs, n_fft=window_size, hop_length=shift_size,
                                       win_length=window_size,
                                       window=window)
    return inverse_stft_outputs
