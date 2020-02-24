from __future__ import absolute_import, division, print_function, unicode_literals

import time

import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms

writer = SummaryWriter('./runs/transfer_demo1')

data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'val':
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
}

data_dir = 'myscripts/torch_learning/data/hymenoptera_data'

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
    for x in ['train', 'val']
}

data_loaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16, shuffle=True,
                                   num_workers=1)
    for x in ['train', 'val']
}

datasets_size = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

devices = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def imgshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(data_loaders['train']))

# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imgshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # 每个全集包含train和val两个集合
        for phase in ['train', 'val']:
            if phase == 'train':
                model.to(devices)
                model.train()
            else:
                model.eval()

            running_loss = .0
            running_correct = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(devices)
                labels = labels.to(devices)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_correct.double() / datasets_size[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * datasets_size[phase])
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Visualizing the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loaders['val']):
            inputs = inputs.to(devices)
            labels = labels.to(devices)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imgshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


# Finetuning the convnet

model_ft = models.resnet18(pretrained=True)
# 获取输出特征的大小
num_ftrs = model_ft.fc.in_features
# 追加一个全连接层，并定义输出为两个分类
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(devices)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optim_ft = optim.SGD(model_ft.parameters(), lr=.0001, momentum=.9)
# 学习速率更新规则
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optim_ft, step_size=7, gamma=0.1)

# model_ft.to(devices)
model_ft = train_model(model_ft, criterion, optim_ft, exp_lr_scheduler,
                       num_epochs=25)
