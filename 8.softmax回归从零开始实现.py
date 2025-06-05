import torch
from  IPython import display
from d2l import torch as d2l

batch_size = 256

train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10
#展平每个图像，视为长度784的向量，有十个类别，将网络维度视为10；
W = torch.normal(0,0.01,(num_inputs,num_outputs),requires_grad=True)
b = torch.zeros(num_outputs,requires_grad=True)

X = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])

print(X.sum(0,keepdim=True))
print(X.sum(1,keepdim=True))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp/partition

X = torch.normal(0,1,(2,5))
X_prob = softmax(X)
print(X_prob,X_prob.sum(1))

def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W) + b)#-1表示生成的矩阵的行数由输入决定，列数由W决定（批量大小）

y = torch.tensor([0,2])

y_hat = torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print(y_hat[[0,1],y])

def cross_entropy(y_hat,y):
    return -torch.log(y_hat[range(len(y_hat)),y])
print(cross_entropy(y_hat,y)) #交叉熵损失函数

def accuracy(y_hat,y):
    """计算预测的正确数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(accuracy(y_hat,y)/len(y))

class Accumulator:
    """在n个变量上累加"""
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data = [0.0]*len(self.data)
    def __getitem__(self, item):
        return self.data[item]
def main():
    def evaluate_accuracy(net,data_iter):
        """计算指定数据集上模型的精度"""
        if isinstance(net,torch.nn.Module):
            net.eval()#将模型设置为评估模式
        metric = Accumulator(2)#正确预测数、预测总数
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
        return metric[0]/metric[1]
    def train_epoch(net,train_iter,loss,updater):
        #训练一个周期
        if isinstance(net,torch.nn.Module):
            net.train()
        metric = Accumulator(3)
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y)
            if isinstance(updater,torch.optim.Optimizer):
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                l.sum().backward()
                updater(X.shape[0])
                metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
        return metric[0]/metric[2],metric[1]/metric[2]

    def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
        animator = d2l.Animator(xlabel='epoch',xlim=[1,num_epochs],legend=['train loss','train acc','test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch(net,train_iter,loss,updater)
            test_acc = evaluate_accuracy(net,test_iter)
            animator.add(epoch+1,train_metrics+(test_acc,))
        train_loss,train_acc = train_metrics

    lr = 0.1
    num_epochs = 10
    updater = lambda batch_size:d2l.sgd([W,b],lr,batch_size)
    d2l.plt.show(train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater))

    print(evaluate_accuracy(net,test_iter))



if __name__ == '__main__':
    main()
