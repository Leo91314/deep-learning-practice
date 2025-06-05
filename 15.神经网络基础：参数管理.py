import torch
from torch import nn

torch.device = torch.device('cuda')
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X = torch.rand(size=(2,4))
print(X)
print(net(X))

print(net[2].state_dict())#打印权重weight

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
print(net[2].weight.grad==None)

print('-----'*10)
print(*[(name,param.shape) for name,param in net[0].named_parameters()])
print(*[(name,param.shape) for name,param in net.named_parameters()])#储存的都是数据
print(net.state_dict()['2.bias'].data)

def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}',block1())
    return net

rgnet = nn.Sequential(block2(),nn.Linear(4,1)) #Sequential定义了一个特殊的module，可以嵌套module，并且可以自动执行module里面的操作
ans = rgnet(X)

print(ans)
print(rgnet)

def init_normal_(m):#初始化权重
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)

net.apply(init_normal_)
print(net[0].weight.data[0],net[0].bias.data[0])

def init_constant_(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,10)
        nn.init.zeros_(m.bias)

net.apply(init_constant_)
print(net[0].weight.data[0],net[0].bias.data[0])

def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42_(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,42)

net[0].apply(xavier)
net[2].apply(init_42_)
print(net[0].weight.data[0],net[0].bias.data[0])
print(net[2].weight.data[0],net[2].bias.data[0])


def my_init(m):
    if type(m) == nn.Linear:
        print("Init",
        *[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *= m.weight.data.abs() > 5

net.apply(my_init)
print(net[0].weight[:2])


shared = nn.Linear(8,8)
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(),
                    shared,nn.ReLU(),
                    shared,nn.ReLU(),
                    nn.Linear(8,1))


#参数绑定
net(X)
print(net[2].weight.data[0] is net[4].weight.data[0])
net[2].weight.data[0,0] = 100
print(net[2].weight.data[0].sum() == net[4].weight.data[0].sum())

#自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X-X.mean()

layer = CenteredLayer()
print(layer(torch.FloatTensor([1,2,3,4,5])))

net = nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y = net(torch.rand(4,8))
print(Y.mean())

print(Y.device)