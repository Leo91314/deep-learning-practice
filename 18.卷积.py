import torch
from torch import nn
from d2l import torch as d2l

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def corr2d(X, K):
    """
    实现二维互相关运算。
    二维互相关运算是图像处理和深度学习中卷积操作的基础。它通过将核矩阵K在输入矩阵X上滑动，
    并在每个位置上计算核矩阵K与输入矩阵X对应部分的元素-wise乘积之和，生成一个新的矩阵Y。

    参数:
    X (torch.Tensor): 输入矩阵，通常是一个二维的图像张量。
    K (torch.Tensor): 核矩阵，通常是一个较小的二维矩阵，用于检测输入矩阵中的特定特征。

    返回:
    Y (torch.Tensor): 输出矩阵，是输入矩阵X和核矩阵K进行互相关运算的结果。
    """
    # 获取核矩阵K的高度和宽度
    h, w = K.shape
    # 初始化输出矩阵Y，其高度和宽度是输入矩阵X的高度和宽度减去核矩阵K的高度和宽度，并加1
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=device)
    # 在输入矩阵X上滑动核矩阵K，计算每个位置的互相关值
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 计算当前位置的互相关值，即核矩阵K与输入矩阵X对应部分的元素-wise乘积之和
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    # 返回计算得到的输出矩阵Y
    return Y


# 将输入张量和卷积核移动到GPU
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], device=device)
K = torch.tensor([[0.0, 1.0, 2.0]], device=device)

ams = corr2d(X, K)
print(ams)

def corr2d(X, K):
    """
    实现二维互相关运算。

    二维互相关运算是图像处理和深度学习中卷积操作的基础。它通过将核矩阵K在输入矩阵X上滑动，
    并在每个位置上计算核矩阵K与输入矩阵X对应部分的元素-wise乘积之和，生成一个新的矩阵Y。

    参数:
    X (torch.Tensor): 输入矩阵，通常是一个二维的图像张量。
    K (torch.Tensor): 核矩阵，通常是一个较小的二维矩阵，用于检测输入矩阵中的特定特征。

    返回:
    Y (torch.Tensor): 输出矩阵，是输入矩阵X和核矩阵K进行互相关运算的结果。
    """
    # 获取核矩阵K的高度和宽度
    h, w = K.shape
    # 初始化输出矩阵Y，其高度和宽度是输入矩阵X的高度和宽度减去核矩阵K的高度和宽度，并加1
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1), device=device)
    # 在输入矩阵X上滑动核矩阵K，计算每个位置的互相关值
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 计算当前位置的互相关值，即核矩阵K与输入矩阵X对应部分的元素-wise乘积之和
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    # 返回计算得到的输出矩阵Y
    return Y

X = torch.ones((6, 8), device=device)
X[:, 2:6] = 0
Y = corr2d(X, torch.tensor([[1.0, -1.0]], device=device))
print(Y)

# 初始化一个2D卷积层，输入通道数为1，输出通道数为1，卷积核大小为1x2，不使用偏置项，并将卷积层移动到指定设备上
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False).to(device)

# 将输入张量X重塑为1x1x6x8的形状，以适配卷积层输入要求
X = X.reshape(1, 1, 6, 8)
# 将目标张量Y重塑为1x1x6x7的形状，以适配卷积层输出要求
Y = Y.reshape(1, 1, 6, 7)

# 迭代训练卷积层参数10次
for i in range(10):
    # 使用卷积层对输入X进行前向传播，得到预测输出Y_hat
    Y_hat = conv2d(X)
    # 计算损失函数，即预测输出Y_hat与目标输出Y之间的平方差
    l = (Y_hat - Y) ** 2

    # 清零卷积层的梯度信息，以准备计算当前迭代的梯度
    conv2d.zero_grad()
    # 对损失函数进行反向传播，计算卷积层权重的梯度
    l.sum().backward()
    # 更新卷积层的权重参数，使用梯度下降法，学习率为3e-2
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad

    # 每2次迭代打印一次当前迭代次数和损失函数值
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')