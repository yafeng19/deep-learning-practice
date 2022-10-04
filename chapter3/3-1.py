import random
import torch
from d2l import torch as d2l

"""3-1 线性回归"""

# 随机生成一些点 y = Xw + b + 噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 正态分布，n个样本，w个属性
    y = torch.matmul(X, w) + b  # y = X*w + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape(-1, 1)  # y是列向量，k行1列，k自动计算

# 生成真实系数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

d2l.set_figsize()
# 绘制所有样本第零个特征和标签的散点图
d2l.plt.scatter(features[:, 0].detach().numpy(),
                labels.detach().numpy(), 1)  # 从计算图中分离出来并转为numpy数组
d2l.plt.show()
# 绘制所有样本第一个特征和标签的散点图
d2l.plt.scatter(features[:, 1].detach().numpy(),
                labels.detach().numpy(), 1)  # 从计算图中分离出来并转为numpy数组
d2l.plt.show()


# 生成小批量数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # shape:(1000,2)，len:1000，shape最外边的维度
    indices = list(range(num_examples))  # 从0到n-1转为list
    random.shuffle(indices)  # 打乱顺序
    for i in range(0, num_examples, batch_size):
        # 获取每个batch所需调用的标号，最后一个batch不满batch_size则取到最后
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)]
        )
        # yield类似迭代器，不断依次返回
        yield features[batch_indices], labels[batch_indices]


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数，均方误差
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 有可能行列向量不匹配，所以reshape一下


# 定义优化算法，小批量随机梯度下降
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
