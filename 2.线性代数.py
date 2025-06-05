import torch
from tensorboard.compat.tensorflow_stub.dtypes import double

#向量操作
x = torch.tensor([3,0])
y = torch.tensor([1,2])

print(x+y)
print(x*y)
print(x/y)
print(x**y)
#可以视为标量值组成的列表

print(len(x))
# 访问张量的长度

# 指定分量m,n创建一个m*n的矩阵
A = torch.arange(20).reshape(5,4)
print(A)
#转置
print(A.T)

#对称矩阵
B = torch.tensor([[1,2,3],[2,3,1],[3,1,2]])
# print(B)
# print(B.T)
print(B.T == B)

#三维矩阵
x = torch.arange(24).reshape(2,3,4)
print(x)

A = torch.arange(20,dtype=torch.float32).reshape(4,5)
B = A.clone()#分配新内存
print(B)

#注：axis代表轴，按维度的轴，例如axis = 0表示纵向积，axis = 1表示横向积，axis = 2表示三维的外向积
#axis = 1 shape:[m,1,n] axis = 0 shape:[1,m,n] axis = 2 shape:[m,n,1]
#指定求和汇总张量的轴
A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis0)
print(A_sum_axis1)
print(A_sum_axis0.shape)
print(A_sum_axis1.shape)
print(A.sum(axis=[0,1]))#总和

#求均值
A_mean_axis0 = A.mean(axis=0)
A_mean_axis1 = A.mean(axis=1)
print(A_mean_axis0)
print(A_mean_axis1)

sum_A = A.sum(axis = 1,keepdims=True)
print(sum_A)
print(A/sum_A)

cumsum_A = A.cumsum(axis=0)
print(cumsum_A)
print(A.cumsum(axis=1))

#点积
x = torch.tensor([1,2,3,4])
y = torch.tensor([5,6,7,8])
z = torch.dot(x,y)


#矩阵向量积
A = torch.tensor([[1,2,3],[4,5,6]])
x = torch.tensor([1,2,3])
z = torch.mv(A,x)
print(z)

#m次矩阵向量积
A = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype = torch.int32)
x = torch.ones(3,3,dtype = torch.int32)
z = torch.mm(A,x)
print(z)

#范数
# 创建一个张量
x = torch.tensor([3, 0, -4, 0],dtype=torch.float32)

# 使用张量的.norm()方法计算L2范数
print(x.norm())  # 这应该输出5.0

# 使用torch.norm()函数计算L2范数
z = torch.norm(x)
print(z)  # 这也应该输出5.0

#L1 范数 向量元素绝对值之和
z = torch.norm(x, p=1)
#或者torch.abs(x).sum()
print(z)

#弗洛贝尼乌斯范数
a = torch.norm(torch.ones(4,4))
#矩阵元素的平方和的平方根
