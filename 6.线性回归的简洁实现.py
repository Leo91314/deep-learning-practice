import numpy as mp
import torch
from torch.utils import data
import matplotlib.pyplot as plt
from d2l import synthetic_data
from torch import nn


true_p = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = synthetic_data(true_p, true_b, 1000)
def load_array(data_arrays, batch_size, is_train=True):
    #构造一个pytorch数据迭代器
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))
#Sequential表示顺序构建一个神经网络
net = nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01)#设置均值为0，方差为0.01的正态分布
print(net[0].bias.data.fill_(0))

loss = nn.MSELoss()#均方误差（平方范数）
trainer = torch.optim.SGD(net.parameters(),lr=0.03)#实例化优化器

num_epochs = 3#定义迭代次数
for epoch in range(num_epochs):
    for X,y in data_iter:
        l = loss(net(X),y)
        trainer.zero_grad()#先将梯度清零
        l.backward()
        trainer.step()#检测梯度更新
    l = loss(net(features),labels)
    print('epoch %d, loss %f' % (epoch + 1, l))
