from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Underlying Principle （基本原则）
# 原理很简单：我们定义了两个距离，一个用于内容（DC），一个用于样式（DS）。
# DC测量两个图像之间的内容有多不同，而DS测量两个图像之间的样式有多不同。
# 然后，我们获取第三个图像，即输入，并对其进行变换，
# 以最小化其与内容图像的内容距离和其与样式图像的样式距离。
# 现在我们可以导入必要的包并开始神经传递。


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image
# 现在我们将导入样式和内容图像。
# 原始的PIL图像的值在0到255之间，但是当转换为torch张量时，它们的值将转换为0到1之间。
# 图像也需要调整大小以具有相同的尺寸。
# 需要注意的一个重要细节是，torch库中的神经网络使用从0到1的张量值进行训练。
# 如果您尝试为网络提供0到255张张量图像，则激活的功能图将无法感知预期的内容和样式。
# 然而，来自Caffe库的预训练网络使用0到255张张量图像进行训练。

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

transform = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = transform(image).unsqueeze(0)
    # unsqueeze增加维度
    return image.to(device, torch.float)


style_img = image_loader("./data/picasso.jpg")
content_img = image_loader("./data/dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()


plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)
