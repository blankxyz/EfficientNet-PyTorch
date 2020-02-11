from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1个图像输入通道， 6个输出通道， 3*3 方形卷积
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train():
    net = Net()
    print(net)
    print(len(list(net.parameters())))
    for x in list(net.parameters()):
        print(x.size())

    input = torch.randn(1, 1, 32, 32)
    # out = net(input)
    # print(out)

    output = net(input)
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    loss = criterion(output, target)

    print(loss)
    f = loss.grad_fn
    while f.next_functions:
        print(f.next_functions[0][0])
        f = f.next_functions[0][0]

    print('backprop')

    print('')

    net.zero_grad()
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()
    print()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)
    # print(loss.grad_fn.net_functions[0][0])


class Net_3channel(nn.Module):

    def __init__(self):
        super(Net_3channel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1_1 = nn.Conv2d(6, 6, 1)
        self.conv1_2 = nn.Conv2d(6, 6, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = F.relu(self.conv1_1(x))
        # x = F.relu(self.conv1_2(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# train()
