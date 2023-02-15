import torch
import preprocess as prep
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

bp_net_state_dict = torch.load('bp_net.path')
new_bp = torch.nn.Sequential(
    torch.nn.Linear(64, 6),
    torch.nn.Sigmoid(),
    torch.nn.Linear(6, 1),
)

new_bp.load_state_dict(bp_net_state_dict)

# filename = 'GE-Verio-SZGNA-PIONEER-HEAD.txt'
filename = 'GE-SZNGA Premier-3D-AXT1-MPR.txt'


def final_test(file_name, lag):
    y2, v2_train_tensor = prep.main_func(file_name, lag)
    output = new_bp(v2_train_tensor)
    # y2_heat = output.tolist()
    y2_tensor = torch.tensor(y2).float()
    plt.figure(1)
    plt.plot(y2)
    plt.figure(2)
    plt.plot(y2_tensor)
    plt.show()


final_test(filename, 64)
