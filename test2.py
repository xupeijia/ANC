import torch.nn as nn
import torch

# 测试利用Conv1d模拟一维数字滤波器建立次级路径
x = torch.tensor([[[1, 2, 3, 4, 5]]], dtype=torch.float32)
w = torch.tensor([[[1, 2, 1]]], dtype=torch.float32)
# b = torch.tensor([0], dtype=torch.float32)
conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 3), padding='same', bias=False)
conv.weight.data = w
# 等价于‘bias=False’
# conv.bias.data = b
output = conv(x)
print(output)
