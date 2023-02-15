import torch.nn as nn
import torch
import pandas as pd

# 测试查看和保存网络参数
batch_size = 50
input_size = 1000
hidden_size = 10
output_size = 1


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(input_size, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, output_size))
    net.apply(init_weights)
    return net


bp_net1 = get_net()
bp_net1.load_state_dict(torch.load("bp_net.path"))
bp_net1.eval()

for name, parameters in bp_net1.state_dict().items():
    if "weight" in name:
        pd.DataFrame(parameters.detach().numpy()).to_csv(name + '.csv', index=False)
        # print(name, ':', parameters.detach().numpy())
    if "bias" in name:
        pd.DataFrame(parameters.detach().numpy()).to_csv(name + '.csv', index=False)
        # print(name, ':', parameters.detach().numpy())
