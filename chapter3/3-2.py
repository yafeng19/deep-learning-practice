import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

"""3-2 线性回归（调包实现）"""
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 构造PyTorch数据迭代器
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)  # 元组拆包
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# 定义网络模型
net = nn.Sequential(nn.Linear(2, 1))  # 输入维度为2，输出维度为1
# 初始化权重和偏差
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数：均方误差
loss = nn.MSELoss()
# 定义优化器，传入所有的参数和学习率
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练和测试
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss={l:f}')
