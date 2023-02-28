import os

import matplotlib.pyplot as plt
import pandas as pd
import utils
import numpy as np
import scipy.signal as signal

# 将所有序列进行带通滤波处理到20-20000Hz
# path_from = 'D:/Study/Projects/MRI/Data/Sequences/sample50k/'
path_from = 'D:/Study/Projects/MRI/Data/Body/GE/'
path_to = 'D:/Study/Projects/MRI/Data/Body/GE/sample16k/'
# path_to = 'D:/Study/Projects/MRI/Data/Body/sample800/'
filenames = [filename for filename in os.listdir(path_from) if '.txt' in filename]
for item in filenames:
    pre_filename = item.split('.')[0]
    # item为文件名，还需要拼接绝对路径才能被读取
    t, x = utils.read_vibration(file_name=path_from + item, start_line=5, sep='\t')
    # nyq = 0.5 * 50000
    # f1 = 20 / nyq
    # f2 = 10000 / nyq
    # b, a = signal.butter(5, [f1, f2], btype='bandpass')
    # filtered_x = signal.filtfilt(b, a, x)
    # 降采样之后要缩放幅值，振动信号乘以1*a，声音信号乘以2.812*a
    # x_rms = np.sqrt(np.mean(np.square(x)))
    # a = x_rms * np.sqrt(800 / 50000)
    sig = utils.resample_and_filter(x, 50000, 16000, 8000)
    # sig = a * sig
    df = pd.DataFrame({'sig': sig})
    df.to_csv(path_to + pre_filename + '.csv', index=False, header=False)
    print(pre_filename + '.csv Finished!')

print('All Finished!')

