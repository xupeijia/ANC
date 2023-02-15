import numpy as np
import torch
import torch.nn as nn
import create_data
import scipy.signal as signal

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1 = nn.Conv1d(1, 1, (1, 16))

# 次级路径传递函数
Sw = np.array([0, 0, 0.9, 0.6, 0.1, -0.4, -0.1, 0.2, 0.1, 0.01, 0.001, 0, 0, 0, 0, 0])


class LossFunction(object):
    def __call__(self, out_tensor, label_tensor):
        outputs = create_data.i_stft(out_tensor, 320, 160)
        outputs = torch.squeeze(outputs, 0)
        length = outputs.numel()
        outputs = outputs.detach().cpu().numpy()
        outputs = signal.lfilter(Sw, 1, outputs)
        outputs = torch.tensor(outputs, dtype=torch.float32, device=device, requires_grad=True)
        loss = torch.sum((outputs - label_tensor)**2) / float(length)
        return loss
