import torch
from torch import nn
from d2l import torch as d2l

def corr2d_multi_in(X,K):#多输入多输出
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))
X = torch.tensor([[[0.1,0.2],[0.3,0.4]],[[0.5,0.6],[0.7,0.8]]])
K = torch.tensor([[[0.1,0.1],[0.1,0.1]],[[0.1,0.1],[0.1,0.1]]])
print(corr2d_multi_in(X,K))

def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k) for k in K],0)
K = torch.stack([K,K+1,K+2],0)
print(corr2d_multi_in_out(X,K))


# corr2d_multi_in_out(X,K)