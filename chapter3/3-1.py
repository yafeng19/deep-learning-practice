import random
import torch
from d2l import torch as d2l

"""3-1 线性回归（手动实现）"""

'''定义模型'''


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


# 定义损失函数：平方误差和
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 有可能行列向量不匹配，所以reshape一下


# 定义优化算法，小批量随机梯度下降
def sgd(params, lr, batch_size):
    # 更新参数的时候不需要参与梯度计算
    with torch.no_grad():
        # 对于每一个参数，都进行随机梯度下降
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  # 下次计算梯度与上一次不相关


'''训练模型'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    # 用每一组X和y优化参数
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # 对每一组X求出预测的y值，与真实y值求损失值
        # l.shape是(batch_size, 1)，求和后求梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    # 优化完一轮之后，计算所有数据点的损失值均值
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}: loss={float(train_l.mean()):f}')

print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')
