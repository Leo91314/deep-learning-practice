import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from d2l import torch as d2l

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(), nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten())

net.to(device)
X = torch.randn(size=(1, 1, 224, 224), device=device)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

lr, num_epochs = 0.05, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=64, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device) 
