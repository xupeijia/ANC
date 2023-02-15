import scipy.signal as signal
import torch
import common
import numpy as np
import librosa

from ancDemo.crnn_lstm import Net

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
t, x = common.read_vibration(file_name='GE-SZNGA Premier-3D-AXT1-MPR.txt')


# y1 = common.get_real_pynl_1(x)
# y = -1 * y1


# 对数据进行降采样到16k
def resample_and_filter(sig, sample_rate_old, sample_rate_new, cutoff_freq):
    # 滤波
    wn = cutoff_freq / (sample_rate_old / 2)
    result = signal.butter(5, wn, 'lowpass')
    b, a = result
    filtered_input = signal.lfilter(b, a, sig)

    # 降采样
    resampled_input = signal.resample_poly(filtered_input, sample_rate_new, sample_rate_old)

    return resampled_input


def calc_seg_num(time, seg_len, shift_len):
    return (time - seg_len) // shift_len + 1


def calc_time_steps(seg_len, sample_rate, window_size, shift_size):
    return (seg_len * sample_rate - window_size) // shift_size + 1


def stft(sig, window_size, shift_size):
    window = np.hanning(window_size)
    spec = np.array([np.fft.rfft(window * sig[i: i + window_size])
                     for i in range(0, len(sig) - window_size + 1, shift_size)])
    return spec


def process_signal(sig, time, seg_len, shift_len, sample_rate, window_size, shift_size):
    seg_num = calc_seg_num(time, seg_len, shift_len)  # 8
    time_steps = calc_time_steps(seg_len, sample_rate, window_size, shift_size)  # 299
    signals = np.array([sig[i: i + seg_len * sample_rate] for i in
                        range(0, time * sample_rate - seg_len * sample_rate + 1, shift_len * sample_rate)])
    stft_outputs = np.array([stft(s, window_size, shift_size) for s in signals])
    real_part, imag_part = np.real(stft_outputs), np.imag(stft_outputs)
    return seg_num, 2, time_steps, 161, np.stack((real_part, imag_part), axis=1)


new_x = resample_and_filter(x, 50000, 16000, 8000)
# out = torch.tensor(new_x, dtype=torch.float32)
_, _, _, _, x = process_signal(new_x, 10, 3, 1, 16000, 320, 160)
print(1)

# net = Net()
# # x = torch.randn((4, 2, 300, 161), dtype=torch.float32)
# x = torch.tensor(out, dtype=torch.float32)


# y = net(x)
# print('{} -> {}'.format(x.shape, y.shape))
#
# print(x.shape)
# print(new_x.shape)
# print(out.shape)

