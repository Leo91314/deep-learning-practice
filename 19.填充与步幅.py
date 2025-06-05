import torch
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1)+X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1,1,kernel_size=3,padding=1) #1个输入通道，1个输出通道，3x3卷积核
X = torch.rand(8,8)
Y = comp_conv2d(conv2d,X)
print(Y.shape)
conv2d = nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1))
Y = comp_conv2d(conv2d,X)
print(Y.shape)