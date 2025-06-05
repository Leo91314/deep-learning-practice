import os
#创建文件夹
import pandas as pd
import torch
import numpy as np
os.makedirs(os.path.join('../PythonProject15','data'),exist_ok=True)
#创建一个人工数据集，储存在csv（逗号分隔值）文件
data_file = os.path.join('../PythonProject15','data','house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas


data = pd.read_csv(data_file)
print(data)

#处理缺失的数据，有插值与删除，首先考虑插值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
print(inputs)#注：这里如果用视频里面的代码无法执行，因为alley是string类型，无法用平均数填充

#对于inputs中的类别值或者离散值，我们将“NaN”视为一个类别
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 转换为 NumPy 数组并确保类型是支持的
X_np = inputs.to_numpy(dtype=np.float32)  # 选择一个适当的 NumPy 类型
y_np = outputs.to_numpy(dtype=np.float32)
#注：以上代码也是原文中所没有的，因为原来的布尔值不可以转化为浮点值输出，需要手动强转
print(X_np,y_np)
