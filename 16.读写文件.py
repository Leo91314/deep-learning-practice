import torch
from torch import nn
from torch.nn import functional as F

x = torch.tensor([[1,2,3,4]])
torch.save(x,'x-file')

x2 = torch.load('x-file')
print(x2)

y = torch.zeros(4,4)
torch.save([x,y],'x-files')
x2,y2 = torch.load('x-files')
print(x2,y2)

mydict = {'x':x,'y':y}
torch.save(mydict,'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

#加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,x):
        return self.out(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2,20))
Y = net(X)

torch.save(net.state_dict(),'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
