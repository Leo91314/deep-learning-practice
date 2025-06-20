import torch

from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

#pytorch不会隐式调整输入的形状
# 所以，需要使用展平层flatten来重新调整输入的形状
net = nn.Sequential(nn.Flatten(),nn.Linear(784,10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss()#交叉熵损失函数
trainer = torch.optim.SGD(net.parameters(),lr=0.1)#优化器
num_epochs = 10


d2l.train_ch6(net,train_iter,test_iter,loss,num_epochs,trainer)
d2l.predict_ch3(net,test_iter)
d2l.plt.show()
