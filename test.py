
import torch
import numpy
def dot(xxx):


def bp_net(input_xxx:list):

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = torch.load('E:\\xxx.pth', map=)

para = net.parameters()

para_string = []
for name, parameter in para:
    print()

    para_string.append(parameter.detach().cpu().numpy())

pause_string = para_string[0]
save_str = 'extern layer1  = [ \n'
for i in range(len(pause_string)):
    for j in range(len(pause_string[i])):
        if j < len(pause_string[i]) - 1:
            save_str = save_str + str(pause_string[i][j]) + ','
        else:
            save_str = save_str + str(pause_string[i][j]) + '] \n'

with open('xxx.h','w') as note:
    note = save_str

    # GRU LSTM,