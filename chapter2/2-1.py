import torch
import numpy

"""2-1 基础数据操作"""

'''1. 生成向量和矩阵'''
A = torch.arange(12)  # 0到11
print(A)
print(A.shape)  # x的大小
print(A.numel())  # tensor中的元素数量
B = A.reshape(3, 4)  # 改为3行4列
print(B)

'''2. 全零和全一矩阵'''
C = torch.zeros(2, 3, 4)
print(C)
C1 = torch.zeros_like(C)  # 生成和C一样维度的零矩阵C1
print(C1)
D = torch.ones(2, 3, 4)
print(D)

'''3. 自定义矩阵'''
E = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2]])
print(E)
print(E.shape)

'''4. 矩阵算数运算'''
F = torch.tensor([1, 2, 4, 8])
G = torch.tensor([2, 2, 2, 2])
print(F + G, F - G, F * G, F / G, F ** G)  # 均为对应元素进行运算
print(torch.exp(F))

'''5. 矩阵逻辑运算'''
F = torch.tensor([1, 2, 4, 8])
G = torch.tensor([2, 2, 2, 2])
print(F == G, F >= G, F > G)  # 均为对应元素进行运算

'''6. 矩阵拼接'''
H = torch.arange(12, dtype=torch.float32).reshape(3, 4)  # shape:(3, 4)， dim=0代表行，dim=1代表列
I = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2]])
print(torch.cat((H, I), dim=0))  # 在列方向拼接（第0维）
print(torch.cat((H, I), dim=0).shape)   # shape:(6, 4)
print(torch.cat((H, I), dim=1))  # 在行方向拼接（第1维）
print(torch.cat((H, I), dim=1).shape)   # shape:(3, 8)

# H = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) # shape:(2,3,4)
# I = torch.tensor([[[12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12]],
#                  [[12, 12, 12, 12], [12, 12, 12, 12], [12, 12, 12, 12]]])
# print(torch.cat((H, I), dim=0)) # 在最外层tensor拼接（第0维）
# print(torch.cat((H, I), dim=1)) # 在中间层tensor拼接（第1维）
# print(torch.cat((H, I), dim=2)) # 在最里层tensor拼接（第2维）

'''7. 矩阵元素求和'''
J = torch.arange(1, 11).reshape(2, 5)  # 从1到10
print(J.sum())

'''8. 广播机制'''
K = torch.arange(3).reshape(3, 1)
L = torch.arange(2).reshape(1, 2)
print(K + L)

'''9. 元素访问与赋值'''
M = torch.arange(12).reshape(3, 4)
print(M[-1], M[-1, :])  # 倒数第一行
print(M[:, -1])  # 倒数第一列
print(M[:2, 2:])  # 前两行并且后两列
print(M[::2, ::3])  # 每2行取第一行，且每3列取第一列

M[0, 0] = 12  # 第一个元素赋值为12
M[:, -1] = 13  # 最后一列元素均赋值为13
print(M)
M[:] = -1  # 所有元素赋值为-1
print(M)

'''10. 新变量分配内存'''
N = torch.arange(12).reshape(3, 4)
before = id(N)
N = N + N
after = id(N)
print(before == after)  # False，意味着原来的N和现在的N不是同一变量，开辟了新的内存
print(N)

# 矩阵执行原地操作，适用于大矩阵，避免复制次数太多占用内存
N = torch.arange(12).reshape(3, 4)
before = id(N)
N += N  # 所有元素的值更新为相应计算结果
after = id(N)
print(before == after)  # True， 意味着原来的N和现在的N是同一变量，没有开辟新内存
print(N)

'''11. numpy张量和torch张量相互转化'''
# torch转numpy
O = torch.arange(12).reshape(3, 4)
P = O.numpy()
print(P, type(P))
# numpy转torch
O = numpy.arange(12).reshape(3, 4)
Q = torch.tensor(O)
print(Q, type(Q))
# 单元素张量转python标量
O = torch.tensor([3.5])
print(O, O.item(), float(O))
