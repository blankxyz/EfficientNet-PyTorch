from __future__ import absolute_import, division, print_function, unicode_literals

"""
Spatial transformer network(STN)
空间变换网络（简称STN）允许神经网络学习如何对输入图像进行空间变换，
以增强模型的几何不变性。
例如，它可以裁剪感兴趣的区域，缩放并更正图像的方向。
这可能是一种有用的机制，因为CNNs对旋转和缩放以及更一般的仿射变换不具有不变性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

data_root = './data'

# Load data
# 设备指定
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# =============

# 1. 准备数据

# =============

# 数据读取器
def get_dataloader():
    """
    获取数据的dataloader对象
    :return: (data_loader, test_loader)
        前一个为训练集，后一个为测试集
    """
    data_loader = torch.utils.data.DataLoader(
        # 第一个参数是一个dataset
        datasets.MNIST(root=data_root, train=True, download=True,
                       # 指定dataset需要做的变换
                       transform=transforms.Compose([
                           transforms.ToTensor,
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        shuffle=True, batch_size=64, num_workers=4
    )

    # Test dataset(测试集的定义，与训练集一样）
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=data_root, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=64, shuffle=True, num_workers=4)

    return data_loader, test_loader


# =======

# Spatial Transformer Network

# =======
class STN_Net(nn.Module):
    """
    STN网络结构（由3部分组成）
    1. localization network 标准CNN
    2. grid generator 生成输入图像与输出的一对一表格
    3. Sampler 使用transformer的参数并将其作用在输入图像上
    :return:
    """

    def __init__(self):
        super(STN_Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    # return None


def train(model, epoch, optim, dataloder):
    # 设置模型为训练模式
    model.train()
    for x in range(epoch):
        for batch_idx, (data, target) in enumerate(dataloder):
            data, target = data.to(device), target.to(device)

            optim.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optim.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloder.dataset),
                       100. * batch_idx / len(dataloder), loss.item()))


def test(model, dataloader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(dataloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(dataloader.dataset),
                      100. * correct / len(dataloader.dataset)))


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_stn(model, dataloader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(dataloader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')


def main():
    train_loader, val_loader = get_dataloader()
    model = STN_Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train(model, 20, optimizer, train_loader)
    test(model, val_loader)

    # Visualize the STN transformation on some input batch
    visualize_stn(model, val_loader)


if __name__ == '__main__':
    main()
