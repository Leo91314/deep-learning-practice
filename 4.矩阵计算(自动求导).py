#导数是切线的斜率
#亚导数:将导数拓展到不可微的函数
#导数拓展到向量的梯度

#自动求导
import torch
x = torch.arange(4.0)
print(x)
#在计算y关于x的梯度前，需要一个地方来存放梯度
x.requires_grad_(True)#隐式计算梯度
print(x.grad)

y = 2*torch.dot(x,x)
print(y)#隐式构造运算图
y.backward()#调用反向传播函数自动计算y关于x的每个分量的梯度
print(x.grad)#返回梯度向量

#默认情况下，每次调用backward函数都会累加梯度，我们需要清楚梯度的数值
x.grad.zero_()
y = x.sum()
y.backward()
print("sum",x.grad)

#深度学习中我们的目的不是计算微分矩阵，而是批量中每个样本单独计算的偏导数的和
x.grad.zero_()
y = x*x
y.sum().backward()
print(x.grad)

#将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x*x
u = y.detach()#将y从记录的计算图移除，设置为常数
z = u*x
z.sum().backward()
print(x.grad)


#即使构建函数的计算图需要python的控制流 ，我们仍然可以计算得到的变量梯度
def f(a):
    b = a*2
    while b.norm() < 1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100*b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad==d/a)