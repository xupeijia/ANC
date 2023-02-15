import time
import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import common

filenames = ['GE-SZNGA Premier-3D-AXT1-MPR - 副本.txt', 'GE-SZNGA Premier-3D-AXT1-MPR.txt', 'GE-Verio-SZGNA-PIONEER-HEAD.txt',
             'GE-Verio-SZGNA-PIONEER-HEAD.txt', 'GE-Verio-SZGNA-PIONEER-Waist-1.txt',
             'GE-Verio-SZGNA-PIONEER-Waist-2.txt']

# filename = 'Siemens-MAGNETOM-HEAD.txt'
# filename = 'GE-SZNGA Premier-3D-AXT1-MPR - 副本.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-HEAD.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-HEAD.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-Waist-1.txt'
# filename = 'GE-Verio-SZGNA-PIONEER-Waist-2.txt'
for filename in filenames:
    if filename != 'GE-SZNGA Premier-3D-AXT1-MPR.txt':
        continue
    t, x = common.read_vibration(filename)
    t = t[:10000]
    x = x[:10000]
    y = common.get_real_y_1(x)
    x = np.array(x, dtype=np.float32)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x.reshape(-1, 1))
    x = np.squeeze(x, 1)
    x = list(x)

    LENS = len(x)
    TAPS = 64
    LEARN_RATE = 0.001

    # 从lms_demo.py引入次级通道的估计权重sw_est
    df = pd.read_csv("lms_w.csv")
    # SW_EST = list(df.iloc[:, 0].values)
    SW_EST = list(df.iloc[:, 1].values)
    # 从gru_demo.py引入次级通道模型s_est_model

    # 分别计算fx_lms和gru_lms
    start_time = time.time()
    # run your code here
    CW1, LOSS1 = common.gru_lms(x, y, LENS, TAPS, LEARN_RATE)
    CW2, LOSS2 = common.fx_lms(x, y, SW_EST, LENS, TAPS, LEARN_RATE)
    # df1 = pd.DataFrame({'LOSS1': LOSS1})
    # df1.to_csv('gru_out_' + filename + '.csv', index=False)
    # df2 = pd.DataFrame({'LOSS2': LOSS2})
    # df2.to_csv('fx_out_' + filename + '.csv', index=False)
    end_time = time.time()
    print('Time taken:', end_time - start_time)

    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(LOSS1, label='LOSS1')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(LOSS2, label='LOSS2')
    plt.legend()
    plt.show()
