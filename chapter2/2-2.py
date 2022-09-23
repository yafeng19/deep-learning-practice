import os
import torch
import numpy
import pandas as pd

"""2-2 数据预处理"""

data_path = os.path.join('.', 'data')  # 目录拼接
os.makedirs(data_path, exist_ok=True)  # 创建目录，目录不存在时创建
data_file = os.path.join(data_path, 'house_tiny.csv')
# 创建csv文件
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    # 插入几条记录
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 根据位置索引获取元素
inputs = inputs.fillna(inputs.mean())  # 数值型属性用均值填充空值
# 字符串型将空值当做一个类别作为一个属性，其他类别分别作为一个属性，所有类别属性转化为0/1
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)  # values取出dataframe中的数值
print(X, y)
