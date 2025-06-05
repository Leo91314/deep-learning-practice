import torch
from torch import nn
from torch.nn import functional as F

net=nn.Sequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
#Sequential定义了一个特殊的module，可以嵌套module，并且可以自动执行module里面的操作
X = torch.rand(2,20)
ans = net(X)
print(X)
print(ans)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)

    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

net = MLP()
ans = net(X)
print(ans)

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self,x):
        for block in self._modules.values():
            x = block(x)
        return  x

net = MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))

ans = net(X)
print(ans)

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20,20),requires_grad=False)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        x = F.relu(torch.mm(x,self.rand_weight)+1)
        x = self.linear(x)
        while x.abs().sum() > 1:
            x /= 2
        return x.sum()
net = FixedHiddenMLP()
ans = net(X)
print(ans)

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20,64),nn.ReLU(),
                                 nn.Linear(64,32),nn.ReLU())
        self.linear = nn.Linear(32,16)

    def forward(self,x):
        return self.linear(self.net(x))

chimera = nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
ans = chimera(X)
print(ans)

