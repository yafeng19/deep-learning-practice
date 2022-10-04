import torch

"""2-4 自动求导"""

'''1. 标量对向量求导'''
A = torch.arange(4.0)
A.requires_grad_(True)  # 需要对A求导，并且A的梯度存放在A.grad中
# 上面两句等价于以下这一句
# A = torch.arange(4.0, requires_grad=True)
print(A.grad)

B = torch.dot(A, A)  # 定义一个B关于A的函数
print(B)  # 一个标量
B.backward()  # B对A中每个分量求导
print(A.grad)  # 经过Backward已经求解出梯度并存储
print(A.grad == 2 * A)  # B对A求导等于2倍向量A

# 在默认情况下，pytorch会累积梯度，因此需要清除之前的值
A.grad.zero_()  # 将A.grad中存储的梯度清零

B = A.sum()  # 定义另一个B关于A的函数
print(B)  # 一个标量
B.backward()  # B对A中每个分量求导
print(A.grad)  # 经过Backward已经求解出梯度并存储
print(A.grad == torch.ones_like(A))  # B对A求导等于全1向量

'''2. 向量对向量求导'''
# 在机器学习中往往只做标量对向量的求导
C = torch.arange(4.0, requires_grad=True)
D = C * C  # 哈达玛积
# 对非标量调用backward要传一个gradient参数
D.sum().backward()
print(C.grad)

'''3. 将某些计算移到计算图之外'''
E = torch.arange(4.0, requires_grad=True)
F = E * E
G = F.detach()  # G等于F并且将G不再看做E的函数，而是看做常数，将其移到计算图之外
H = G * E  # H是一个向量

H.sum().backward()  # H对E求导，结果其实等于常数G
print(E.grad)
print(E.grad == F)

E.grad.zero_()
F.sum().backward()  # F仍然是E的函数，还可以对E求导
print(E.grad)
print(E.grad == 2 * E)
