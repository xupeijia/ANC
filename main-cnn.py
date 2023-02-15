import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# import tensorwatch as tw
# from torchsummary import summary
# from sklearn.model_selection import train_test_split

NUM_EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001

# diction = {'B007': 0, 'B014': 1, 'B021': 2, 'OR007': 3, 'OR014': 4, 'OR021': 5, 'IR007': 6, 'IR014': 7, 'IR021': 8}
diction = {'B007': 0, 'B014': 1, 'B021': 2, 'O007': 3, 'O014': 4, 'O021': 5, 'I007': 6, 'I014': 7, 'I021': 8,
           'normal': 9}
# diction = {'Ball': 0, 'IR': 1, 'Normal': 2, 'OR': 3}
encoder = torch.eye(10)


class Net(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(Net, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),  # 16, 26 ,26
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 32, 12,12     (24-2) /2 +1

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))  # 128, 4,4

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.dropout(x)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


# 训练集
class Mytraindataset(Dataset):
    def __init__(self):
        self.root_dir = 'F://grade_four//bian//dataset4//env_3//'
        # self.root_dir = 'F://grade_four//bian//dataset3//baoluotestcut//2HP//'
        # self.root_dir = 'F://grade_four//bian//dataset2//1hp2//'
        self.signals = os.listdir(self.root_dir)  # 获取文件夹列表

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal_index = self.signals[index]
        signal_path = os.path.join(self.root_dir, signal_index)  # 把目录和文件名合成一个路径
        signal = np.load(signal_path)
        # signal = signal.T
        signal = torch.from_numpy(signal)
        signal = signal.type(torch.FloatTensor)
        label = signal_path.split('//')[-1].split('_')[0]
        label = diction[label]

        return signal, label


# 测试集
class Mytestdataset(Dataset):
    def __init__(self):
        self.root_dir = 'F://grade_four//bian//dataset4//env_2//'
        # self.root_dir = 'F://grade_four//bian//dataset3//baoluotestcut//3HP//'
        # self.root_dir = 'F://grade_four//bian//dataset2//2hp2//'
        self.signals = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal_index = self.signals[index]
        signal_path = os.path.join(self.root_dir, signal_index)
        signal = np.load(signal_path)
        # signal = signal.T
        signal = torch.from_numpy(signal)
        signal = signal.type(torch.FloatTensor)
        label = signal_path.split('//')[-1].split('_')[0]
        label = diction[label]

        return signal, label


device = torch.device("cuda")

# 导入训练集、测试集
train_loader = DataLoader(Mytraindataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(Mytestdataset(), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = Net().to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def nth_derivative(f, wrt, n):
    for i in range(n):
        if not f.requires_grad:
            return torch.zeros_like(wrt)
        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


def train(model, device, train_loader, optimizer, epoch):
    correct = 0
    running_loss = 0
    for index, (signal, label) in enumerate(train_loader):
        signal, label = signal.to(device), label.to(device)

        signal.requires_grad = True

        pred = model(signal)
        #         print('pred:\t', pred)
        #         print('label:\t', label)
        loss = F.nll_loss(pred, label)
        #         print("pred:", pred, "label:", label)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        signal_grad = signal.grad.data

        if index % 100 == 0:
            print("Train Epoch: {}, iteration: {}, Loss: {}".format(epoch, index, loss.item()))
            # print("Train Epoch: {}, Loss: {},acc:{}".format(epoch, runing_loss / 100, train_acc))
            # running_loss = 0


#         print(signal_grad.sign())


# In[13]:


def test(model, device, test_loader):
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for index, (signal, label) in enumerate(test_loader):
            signal, label = signal.to(device), label.to(device)
            output = model(signal)
            total_loss += F.nll_loss(output, label, reduction="sum").item()
            pred = output.argmax(dim=1)  # batch_size * 1
            correct += pred.eq(label.view_as(pred)).sum().item()

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100.
    print("Test loss: {}, Accuracy: {}".format(total_loss, acc))
    return acc


acc = []
e = []

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch)
    #     test(model, device, test_loader)
    acc.append(test(model, device, test_loader))
    e.append(epoch)

# In[ ]:

print(Net)
plt.figure(figsize=(5, 5))
plt.plot(e, acc, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, 0.04, step=0.005))
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# In[ ]:


# torch.save(model.state_dict(), 'originWDCNN.pth')


# In[ ]:
