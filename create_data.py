import scipy.signal as signal
import torch
import common
import numpy as np
import torch.nn as nn

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 对数据进行降采样到16k
def resample_and_filter(sig, sample_rate_old, sample_rate_new, cutoff_freq):
    # 滤波
    wn = cutoff_freq / (sample_rate_old / 2)
    b, a = signal.butter(5, wn, 'lowpass')
    filtered_input = signal.lfilter(b, a, sig)
    # 降采样
    resampled_input = signal.resample_poly(filtered_input, sample_rate_new, sample_rate_old)
    return resampled_input


def create_segment(sig, time, seg_len, shift_len, sample_rate):
    # 切分信号为(batch_size, segment)
    signals = np.array([sig[i: i + seg_len * sample_rate] for i in
                        range(0, time * sample_rate - seg_len * sample_rate + 1, shift_len * sample_rate)])
    signals = torch.as_tensor(signals, dtype=torch.float32, device=device)
    return signals


def stft(sig, window_size, shift_size):
    # 把序列(batch_size, segment)转成网络输入的tensor(batch_size, feature_maps, time_steps, frequency)
    window = torch.hann_window(window_size, device=device)
    stft_outputs = torch.stft(sig, n_fft=window_size, hop_length=shift_size, win_length=window_size, window=window,
                              return_complex=True)
    stft_outputs = torch.transpose(stft_outputs, 1, 2)
    real_part, imag_part = torch.real(stft_outputs), torch.imag(stft_outputs)
    return torch.stack([real_part, imag_part], dim=1)


def i_stft(out_tensor, window_size, shift_size):
    # 把网络输出的tensor(batch_size, feature_maps, time_steps, frequency)转成序列(batch_size, segment)
    complex_stft_outputs = out_tensor[:, 0, :, :] + 1j * out_tensor[:, 1, :, :]  # 8*299*161
    complex_stft_outputs = torch.transpose(complex_stft_outputs, 1, 2)
    window = torch.hann_window(window_size, device=device)
    inverse_stft_outputs = torch.istft(complex_stft_outputs, n_fft=window_size, hop_length=shift_size,
                                       win_length=window_size,
                                       window=window)
    return inverse_stft_outputs


# new_x = resample_and_filter(x, 50000, 16000, 8000)
# sig = create_segment(new_x, 10, 3, 1, 16000)
# x = stft(sig, 10, 3, 1, 16000, 320, 160)
# inverse_stft_sig = i_stft(x, 320, 160)
# print(1)


def sp_fun(x):
    # x = tensor(1,48000)
    x = torch.unsqueeze(x, 0)
    w = torch.tensor([[[0.05, 0.2, 1]]], dtype=torch.float32)
    conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding='same', bias=False)
    conv.weight.data = w
    output = conv(x)
    output = torch.squeeze(output, 0)
    return output
