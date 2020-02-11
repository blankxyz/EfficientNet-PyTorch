from __future__ import absolute_import, division, print_function, unicode_literals

import torch


# Tensor

def demo1():
    # 构建一个5*3的矩阵，不初始化
    x = torch.empty(5, 3)
    print(x)

    # 构建一个随机初始化矩阵
    x = torch.rand(5, 3)
    print(x)

    # 构建一个矩阵填充为0，并且类型是long
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    y = torch.rand(5, 3)

    print('加法试验：')
    print(x + y)
    print(torch.add(x, y))

    print('对位加法')
    print(f'y的值是：{y}')
    z = torch.rand(5, 3)
    print(f'z的值是{z}')

    # torch.add(x, y, out=result)
    #
    # print(result)

    print(y.add_(z))
    print(y.t_())
    # print(y.copy_(x))
    print(y[:, 1])

    print('变tensor的形状和大小')
    x = torch.randn(4, 4)
    print(x)
    y = x.view(16)
    z = x.view(-1, 8)
    print(x, y, z)
    print(x.size(), y.size(), z.size())

    print('获取tensor中的值')
    x = torch.randn(5)
    print(x)
    try:
        print(x.item())
    except Exception as e:
        print(e)
        x = torch.randn(1)
        print(x.item())


def demo2():
    print('================================')
    print('====== Numpy Bridge ============')
    print('')

    print('转换一个 torch tensor为 numpy 数组')
    a = torch.ones(5)
    b = a.numpy()
    print(a)
    print(b)

    a.add_(1)
    print(a)
    print(b)


def demo3_cuda_tensor():
    x = torch.randn(5)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        y = torch.ones_like(x, device=device)

        x = x.to(device)
        z = x + y
        print(z)
        print(z.to('cpu', torch.double))


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3_cuda_tensor()
