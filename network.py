import torch
import torch.nn as nn


# 带有组策略的GRU网络，作为CRN网络的Encoder和Decoder的中间部分
class GGru(nn.Module):
    def __init__(self, in_features=None, out_features=None, mid_features=None, hidden_size=1024, groups=2):
        super(GGru, self).__init__()

        hidden_size_t = hidden_size // groups

        self.lstm_list1 = nn.ModuleList(
            [nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])
        self.lstm_list2 = nn.ModuleList(
            [nn.GRU(hidden_size_t, hidden_size_t, 1, batch_first=True) for i in range(groups)])

        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.groups = groups
        self.mid_features = mid_features

    def forward(self, x):
        out = x
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.size(0), out.size(1), -1).contiguous()

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.stack([self.lstm_list1[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = torch.flatten(out, start_dim=-2, end_dim=-1)
        out = self.ln1(out)

        out = torch.chunk(out, self.groups, dim=-1)
        out = torch.cat([self.lstm_list2[i](out[i])[0] for i in range(self.groups)], dim=-1)
        out = self.ln2(out)

        out = out.view(out.size(0), out.size(1), x.size(1), -1).contiguous()
        out = out.transpose(1, 2).contiguous()

        return out


class CrnNet(nn.Module):
    def __init__(self):
        super(CrnNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2))

        self.g_gur = GGru(groups=2)

        self.conv5_t_1 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv4_t_1 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv3_t_1 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv2_t_1 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.conv1_t_1 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.conv5_t_2 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(1, 3), stride=(1, 2))
        self.conv4_t_2 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(1, 3), stride=(1, 2))
        self.conv3_t_2 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(1, 3), stride=(1, 2))
        self.conv2_t_2 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(1, 3), stride=(1, 2),
                                            output_padding=(0, 1))
        self.conv1_t_2 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=(1, 3), stride=(1, 2))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.bn5_t_1 = nn.BatchNorm2d(128)
        self.bn4_t_1 = nn.BatchNorm2d(64)
        self.bn3_t_1 = nn.BatchNorm2d(32)
        self.bn2_t_1 = nn.BatchNorm2d(16)
        self.bn1_t_1 = nn.BatchNorm2d(1)

        self.bn5_t_2 = nn.BatchNorm2d(128)
        self.bn4_t_2 = nn.BatchNorm2d(64)
        self.bn3_t_2 = nn.BatchNorm2d(32)
        self.bn2_t_2 = nn.BatchNorm2d(16)
        self.bn1_t_2 = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)

        self.fc1 = nn.Linear(in_features=161, out_features=161)
        self.fc2 = nn.Linear(in_features=161, out_features=161)

    def forward(self, x):
        out = x
        e1 = self.elu(self.bn1(self.conv1(out)))
        e2 = self.elu(self.bn2(self.conv2(e1)))
        e3 = self.elu(self.bn3(self.conv3(e2)))
        e4 = self.elu(self.bn4(self.conv4(e3)))
        e5 = self.elu(self.bn5(self.conv5(e4)))

        out = e5

        out = self.g_gur(out)

        out = torch.cat((out, e5), dim=1)

        d5_1 = self.elu(torch.cat((self.bn5_t_1(self.conv5_t_1(out)), e4), dim=1))
        d4_1 = self.elu(torch.cat((self.bn4_t_1(self.conv4_t_1(d5_1)), e3), dim=1))
        d3_1 = self.elu(torch.cat((self.bn3_t_1(self.conv3_t_1(d4_1)), e2), dim=1))
        d2_1 = self.elu(torch.cat((self.bn2_t_1(self.conv2_t_1(d3_1)), e1), dim=1))
        d1_1 = self.elu(self.bn1_t_1(self.conv1_t_1(d2_1)))

        d5_2 = self.elu(torch.cat((self.bn5_t_2(self.conv5_t_2(out)), e4), dim=1))
        d4_2 = self.elu(torch.cat((self.bn4_t_2(self.conv4_t_2(d5_2)), e3), dim=1))
        d3_2 = self.elu(torch.cat((self.bn3_t_2(self.conv3_t_2(d4_2)), e2), dim=1))
        d2_2 = self.elu(torch.cat((self.bn2_t_2(self.conv2_t_2(d3_2)), e1), dim=1))
        d1_2 = self.elu(self.bn1_t_2(self.conv1_t_2(d2_2)))

        out1 = self.fc1(d1_1)
        out2 = self.fc2(d1_2)
        out = torch.cat([out1, out2], dim=1)

        return out


# net = Net()
# x = torch.randn((4, 2, 300, 161), dtype=torch.float32)
# y = net(x)
# print('{} -> {}'.format(x.shape, y.shape))


class BpNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(BpNet, self).__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(input_num, hidden_num),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_num, output_num),
        )

    def forward(self, x):
        x = self.net1(x)
        return x


class GruNet(nn.Module):
    def __init__(self, input_num, hidden_num, output_num, num_layers, seq):
        super(GruNet, self).__init__()
        self.gru = nn.GRU(input_num, hidden_num, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_num * seq, output_num)

    def forward(self, x):
        x, _ = self.gru(x)  # (seq, batch, hidden)
        # x = torch.transpose(x, 1, 0)  # 调整为 (batch, seq, hidden)
        b, s, h = x.shape
        x = x.reshape(b, s * h)  # (batch, seq * hidden)
        x = self.fc(x)
        return x
