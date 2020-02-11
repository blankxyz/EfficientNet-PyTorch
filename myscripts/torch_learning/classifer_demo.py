from __future__ import absolute_import, division, print_function, unicode_literals

import time

import torch
import torchvision
import torchvision.transforms as transforms

# 读取正则化后的cifar10数据
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, .5))
    ]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np


# 显示图片
def imshow(img):
    img = img / 2 + .5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

print(' '.join('%5s ' % classes[labels[j]] for j in range(4)))

# 保存训练结果
PATH = './cifar_net.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from myscripts.torch_learning.networks import Net_3channel

net = Net_3channel()


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     net = nn.DataParallel(net)
def train(enable_gpu):
    # from myscripts.torch_learning.networks import Net_3channel
    #
    # net = Net_3channel()

    # 定义损失函数和优化器
    import torch.optim as optim

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=.001, momentum=.9)

    for epoch in range(2):

        running_loss = .0
        for i, data in enumerate(trainloader, 0):
            if enable_gpu:

                net.to(device)
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            # inputs, labels = data

            optimizer.zero_grad()

            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # 打印状态
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 保存训练结果
    # PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


start_t = time.time()
train(True)
end_t = time.time()
print(f'训练时间：{end_t-start_t}')

testdata = iter(testloader)
images, labels = testdata.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net.load_state_dict(torch.load(PATH))


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net.to(device)

# output = net(images)

# _, predicted = torch.max(output, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))

def check_all_data(enable_gpu=True):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            if enable_gpu:
                net.to(device)
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


# import time

start_t = time.time()
# check_all_data(False)
end_t = time.time()
print(f'使用gpu耗时：{end_t-start_t}')


def check_classes_acc(enable_gpu):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            if enable_gpu:
                net.to(device)
                images, labels = data[0].to(device), data[1].to(device)
            else:
                # net.to(cpu)
                images, labels = data
            # images, labels = data
            # images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


start_t = time.time()
# check_all_data()
# check_classes_acc(True)
end_t = time.time()
print(f'未使用gpu耗时：{end_t-start_t}')

# check_classes_acc(False)
