import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal

df1 = pd.read_csv("x_and_y_linear1.csv")
x1 = list(df1.iloc[:, 0].values)
y1 = list(df1.iloc[:, 1].values)
df2 = pd.read_csv("x_and_y_total1.csv")
x2 = list(df2.iloc[:, 0].values)
y2 = list(df2.iloc[:, 1].values)

LENS = len(x1)
TAPS = 16
LEARN_RATE = 0.1
PW = [0.01, 0.25, 0.5, 1, 0.5, 0.25, 0.01]
SW = [i * 0.25 for i in PW]


# 定义LMS算法拟合次级通道传函
def lms(x, y, lens, taps, learn_rate):
    shx = [0] * taps
    shw = [0] * taps
    loss = [0] * lens
    for item in range(lens):
        del shx[-1]
        shx = [x[item]] + shx
        shy = sum(i * j for i, j in zip(shx, shw))
        loss[item] = y[item] - shy
        shw_old = [learn_rate * loss[item] * i for i in shx]
        shw = [i + j for i, j in zip(shw_old, shw)]
    return shx, shw, loss


# 定义绘图函数
def draw(picture_num, lens, loss, sw, shw):
    plt.figure(picture_num)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, lens + 1), loss)
    plt.ylabel("A")
    plt.xlabel("T")
    plt.legend("loss")
    plt.subplot(2, 1, 2)
    plt.stem(sw, label='SW')
    plt.stem(shw, linefmt='r*', label='SHW')
    plt.ylabel("A")
    plt.xlabel("W")
    plt.legend()
    plt.show()


shx1, shw1, loss1 = lms(x1, y1, LENS, TAPS, LEARN_RATE)
shx2, shw2, loss2 = lms(x2, y2, LENS, TAPS, LEARN_RATE)
# draw(1, LENS, loss1, SW, shw1)
# draw(2, LENS, loss2, SW, shw2)

y2_identity = signal.lfilter(shw2, 1, x2)
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(y2, label='Original data')
plt.subplot(2, 1, 2)
plt.plot(y2_identity, label='Predictions')
plt.legend()
plt.show()

print(shw1)
print(shw2)
# 保存shw1和shw2
df = pd.DataFrame({'linear_w': shw1, 'nonlinear_w': shw2})
df.to_csv('lms_w.csv', index=False)
