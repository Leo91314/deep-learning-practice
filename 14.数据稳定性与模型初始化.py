import torch
from d2l import torch as d2l

x = torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))