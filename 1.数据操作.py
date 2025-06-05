
import torch

#张量表示一个数组
x = torch.arange(12)
print(x)

#张量的形状
print(x.shape)

#张量的元素总个数
print(x.numel())

#改变张量的形状，不改变元素
x = x.reshape(3,4)

#使用全为0，全为一或者特定分布随机采样的数字
a = torch.zeros((2,3,4))
b = torch.ones((2,3,4))
print(a)
print(b)

#使用数值来创建python列表，为张量元素赋予特定的值
c = torch.tensor([[0,1,2],[3,4,5]])
print(c)

#算术运算符
x = torch.tensor([1,2,4,8])
y = torch.tensor([2,2,2,2])
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**y)
print('*'*30)
#将多个张量链接在一起
x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((x,y),dim=0))
#逻辑运算符
print(x<y)
#全部求和
print(x.sum())

#广播机制
a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))
print(a,b)
print(a+b)

#访问：[-1]选择最后一个元素，[1:3]访问2与3的元素
print(x[-1])
print(x[1:3])

#指定位置写入
x[1,2] = 9
print(x)

#运行一些操作可能会导致为新结构分配内存
before = id(x)
x=x+y
print(before == id(x))

#运行一些操作不会导致为新结构分配内存（执行原地操作）
z = torch.zeros_like(x)
print('id(z):',id(z))
z[:] = x+1
print('id(z):',id(z))
import numpy
#转化为numpy张量
A = x.numpy()
B = torch.from_numpy(A)
print(type(A),type(B))

#将大小为1的张量转换为python标量
a = torch.tensor([3.5])
print(a)
print(a.item())
print(type(a),type(a.item()))