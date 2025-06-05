import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))#卷积层数，输入通道数，输出通道数

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
                         nn.Linear(4096, 10))

net = vgg(conv_arch)
net.to(device)
X = torch.randn(size=(1, 1, 224, 224), device=device)
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

lr ,num_epochs = 0.05, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=64, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)