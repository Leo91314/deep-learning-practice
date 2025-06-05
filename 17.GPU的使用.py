import torch
from torch import nn

print(torch.device('cpu'))
print(torch.device('cuda'))
print(torch.device('cuda:0'))

print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device('cuda:{}'.format(i))
    return torch.device('cpu')
def try_all_gpus():
    devices = [torch.device('cuda:{}'.format(i))
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

x = torch.tensor([1,2,3])
print(x.device)

x = torch.ones(2,3,device=try_gpu())
print(x)

Y = torch.rand(2,3,device=try_gpu(1))
print(Y.device)

Z = x.cuda()
print(x.device == Z.device)
print(x.device == Y.device)

#神经网络与GPU
net = nn.Sequential(nn.Linear(3,1))
net = net.to(device=try_gpu())
print(net[0].weight.data.device) 