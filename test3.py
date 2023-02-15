import common
import scipy.signal as signal

filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'
# t, x = common.read_vibration(filename)
x = range(1,20)

Pw = [0, 0, 0, 0, 0, 0, 0.8, 0.6, -0.2, -0.5, -0.1, 0.4, -0.05]


y1 = signal.lfilter(Pw, 1, x)
# y2 = common.get_real_y(x)

print(y1[:20])
# print(y2[:20])
