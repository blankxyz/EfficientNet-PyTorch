from __future__ import absolute_import, division, print_function, unicode_literals

import torch


def demo_tensor():
    # 创建一个tensor并设置求梯度为True
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    # 定义一个关于x的操作
    y = x + 2
    print(y)

    print(y.grad_fn)

    # 再定义一个对y的操作
    z = y * y * 3
    out = z.mean()
    print(z, out)

    # def gradients_demo():
    print('')
    print('gradients demo')
    print('')

    out.backward()
    print(x.grad)

    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    y.backward(x)
    print(x.grad)


if __name__ == '__main__':
    demo_tensor()
