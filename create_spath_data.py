import scipy.signal as signal
import numpy as np
import pandas as pd
import utils

# 产生线性部分系数向量
from matplotlib import pyplot as plt

sw = [0, 0.3, 0.4, 0.1, -0.2, -0.1, 0.1, 0.1, 0.01, 0.001]
a = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 产生高斯白噪声
size = 50000
x = list(np.random.normal(0, 1, size))

# 获取线性部分输出
y_linear = signal.lfilter(sw, a, x)

# 获取非线性部分输出
y_nl1 = list(np.tanh(x))

y_nl2 = [0] * len(x)
for i in range(len(x)):
    y_nl2[i] = x[i] - 0.1 * x[i] ** 2 + 0.2 * x[i] ** 3

# 线性输出+非线性输出
y_total1 = [i + j for i, j in zip(y_linear, y_nl1)]
y_total2 = [i + j for i, j in zip(y_linear, y_nl2)]

y_total3 = signal.lfilter([1, 0.2, 0.05], [1, 0, 0], 3.3 * np.tanh(0.3 * np.array(x)))

y_nl3: [float] = [0.] * len(x)
x = [0, 0] + x
for i in range(len(x)):
    y_nl3[i - 2] = -0.00005676 * (0.35 * x[i]) ** 2 - 0.0000219165 * (0.35 * x[i - 1]) ** 2 + 0.000073412 * (
            0.35 * x[i - 2]) ** 2

y_total4 = [i + j for i, j in zip(y_linear, y_nl3)]

x = x[2:]

y_total5 = common.get_real_y_3(x)
# 保存输入和输出
df1 = pd.DataFrame({'x': x, 'y_linear': y_linear})
df1.to_csv('x_and_y_linear1.csv', index=False)
df2 = pd.DataFrame({'x': x, 'y_total1': y_total1})
df2.to_csv('x_and_y_total1.csv', index=False)
df3 = pd.DataFrame({'x': x, 'y_total2': y_total2})
df3.to_csv('x_and_y_total2.csv', index=False)
df4 = pd.DataFrame({'x': x, 'y_total3': y_total3})
df4.to_csv('x_and_y_total3.csv', index=False)
df5 = pd.DataFrame({'x': x, 'y_total4': y_total4})
df5.to_csv('x_and_y_total4.csv', index=False)
df6 = pd.DataFrame({'x': x, 'y_total5': y_total5})
df6.to_csv('x_and_y_total5.csv', index=False)
