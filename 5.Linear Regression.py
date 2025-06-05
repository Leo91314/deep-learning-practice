import torch
import random
import matplotlib.pyplot as plt
def synthetic_data(w,b,numm_examples):
    #生成y = wx + b + 噪声
    x = torch.normal(0,1,(numm_examples,len(w)))#定义x属于正太分布的随机数
    y = torch.matmul(x,w) + b#设置回归函数
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
print('features:',features[0],'\nlabel:',labels[0])

plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
plt.show()


def data_iter(batch_size,features,labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)#打乱数据,使数据没有特定的顺序
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size,num_examples)])
        yield features[batch_indices],labels[batch_indices]

batch_size = 10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)
def linreg(X,w,b):#定义线性回归函数
    return torch.matmul(X,w) + b
def squared_loss(y_hat,y):#定义回归损失函数
    return (y_hat-y.reshape(y_hat.shape))**2/2

def sgd(params,lr,batch_size):#小规模梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)#X与y的小批量损失
        # 因为形状是batch_size,1，不是一个标量，所以以l.sum()计算w,b的梯度


        l.sum().backward()
        sgd([w,b],lr,batch_size)#实时更新参数的梯度
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch+1},loss {float(train_l.mean()):f}')
        print(f'w的估计误差:{true_w-w.reshape(true_w.shape)}')