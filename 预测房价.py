import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
#@save
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)# 将所有特征转换为独热编码，包括处理缺失值
print(all_features.shape)

n_train = train_data.shape[0]
# 遍历每一列，检查并转换
for col in all_features.columns:
    if all_features[col].dtype == 'object':
        # 尝试将字符串类型转换为数值类型
        all_features[col] = pd.to_numeric(all_features[col], errors='coerce')

# 删除仍然包含NaN的行
all_features = all_features.dropna()

# 转换为Tensor
train_features = torch.tensor(all_features[:n_train].values.astype('float32'))

# 转换为 float32 类型
test_features = torch.tensor(all_features[n_train:].values.astype('float32'))

train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
infeatures = train_features.shape[1]
def get_net():
    net = nn.Sequential(nn.Linear(infeatures,1))
    return net


def log_rmse(net, features, labels):
    """
    计算模型预测的均方根对数误差（Root Mean Square Logarithmic Error, RMSLE）。

    该函数首先使用模型对特征进行预测，并对预测值进行裁剪，以避免取对数时出现未定义值。
    然后，它计算预测值的对数与真实标签的对数之间的均方根误差。
    这种方法特别适用于预测值和标签值范围广泛的回归问题，因为取对数可以将大范围的值压缩到较小的范围内。

    参数:
    - net (nn.Module): 用于预测的神经网络模型。
    - features (Tensor): 输入特征数据。
    - labels (Tensor): 真实标签数据。

    返回:
    - float: 均方根对数误差。
    """
    # 使用模型预测并裁剪预测值，避免取对数时出现未定义值
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    # 计算预测值的对数与真实标签的对数之间的均方根误差
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    # 返回均方根对数误差
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """
    训练神经网络并评估其性能。

    参数:
    net: 模型对象，用于训练和预测。
    train_features: 训练集特征。
    train_labels: 训练集标签。
    test_features: 测试集特征。
    test_labels: 测试集标签。
    num_epochs: 训练的轮数。
    learning_rate: 学习率，用于优化算法。
    weight_decay: 权重衰减，用于正则化。
    batch_size: 批次大小，每次训练处理的样本数。

    返回:
    train_ls: 训练集的损失列表。
    test_ls: 测试集的损失列表，如果测试标签不为空。
    """
    # 初始化训练和测试的损失列表
    train_ls, test_ls = [], []
    # 加载训练数据集
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    # 开始训练循环
    for epoch in range(num_epochs):
        # 遍历训练数据集
        for X, y in train_iter:
            # 清除梯度
            optimizer.zero_grad()
            # 计算损失
            l = loss(net(X), y)
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
        # 记录训练集的损失
        train_ls.append(log_rmse(net, train_features, train_labels))
        # 如果测试标签存在，记录测试集的损失
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    # 返回训练和测试的损失列表
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """
    获取k折交叉验证的训练数据和验证数据。

    参数:
    k - int, k折交叉验证的k值。
    i - int, 当前折的索引，用于指定哪一折作为验证集。
    X - Tensor, 输入特征数据。
    y - Tensor, 输入标签数据。

    返回:
    X_train - Tensor, 训练用的输入特征数据。
    y_train - Tensor, 训练用的标签数据。
    X_valid - Tensor, 验证用的输入特征数据。
    y_valid - Tensor, 验证用的标签数据。
    """
    # 确保k的值大于1，因为k折交叉验证需要至少两折
    assert k > 1

    # 计算每折数据的大小
    fold_size = X.shape[0] // k  # 每折数据的大小

    # 初始化训练数据和标签为None，稍后将根据条件赋值
    X_train, y_train = None, None

    # 遍历每一折
    for j in range(k):
        # 计算当前折的索引范围
        idx = slice(j * fold_size, (j + 1) * fold_size)

        # 提取当前折的特征数据和标签
        X_part, y_part = X[idx, :], y[idx]

        # 如果当前折是第i折，则将其作为验证集
        if j == i:
            X_valid, y_valid = X_part, y_part
        # 如果训练数据还未初始化，则将当前折作为训练集的初始值
        elif X_train is None:
            X_train, y_train = X_part, y_part
        # 如果训练数据已经初始化，则将当前折追加到训练集中
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)

    # 返回训练数据和验证数据
    return X_train, y_train, X_valid, y_valid
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls,valid_ls = train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}, 训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred(train_features, test_feature, train_labels, test_data,
                  num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将测试数据的特征和标签转换为张量
    preds = net(test_features).detach().numpy()
    # 将预测结果转换为DataFrame
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])

    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
train_and_pred(train_features, test_features, train_labels, test_data,num_epochs, lr, weight_decay, batch_size)
