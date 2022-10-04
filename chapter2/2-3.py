import torch

"""2-3 线性代数基础"""

'''1. 标量运算'''
# 向量由只含一个元素的tensor表示
A = torch.tensor(3)  # 视为3，int类型
B = torch.tensor(2.)  # 视为2.0，float类型
# 算术运算按照类型兼容规则
print(A + B, A - B, A * B, A / B, A ** B)

'''2. 矩阵转置'''
C = torch.arange(20).reshape(5, 4)
print(C, C.T, sep='\n')  # 原矩阵和转置矩阵，用回车分割两个输出项

'''3. 矩阵数乘、矩阵哈达玛积'''
D = torch.arange(12).reshape(3, 4)
E = D.clone()
# 矩阵数乘：标量乘以矩阵的每个元素
print(2 * D)
# 哈达玛积：同型矩阵对应元素相乘
print(E * D)

'''4. 矩阵求和'''
F = torch.arange(12).reshape(3, 4)
print(F)
print(F.sum(), F.shape)  # 全部元素求和
print(F.sum(axis=0), F.sum(axis=0).shape)  # 第0维对应位置元素求和，shape:(4)
print(F.sum(axis=1), F.sum(axis=1).shape)  # 第1维对应位置元素求和，shape:(3)

G = torch.arange(24).reshape(2, 3, 4)
print(G)
print(G.sum(axis=0), G.sum(axis=0).shape)  # 第0维对应位置元素求和，shape:(3, 4)
print(G.sum(axis=1), G.sum(axis=1).shape)  # 第1维对应位置元素求和，shape:(2, 4)
print(G.sum(axis=2), G.sum(axis=2).shape)  # 第2维对应位置元素求和，shape:(2, 3)
print(G.sum(axis=[0, 1]), G.sum(axis=[0, 1]).shape)  # 第0、1维对应位置元素求和，shape:(4)
print(G.sum(axis=[1, 2]), G.sum(axis=[1, 2]).shape)  # 第1、2维对应位置元素求和，shape:(2)
print(G.sum(), G.shape)  # 全部元素求和

'''5. 矩阵求均值'''
H = torch.arange(12, dtype=float).reshape(3, 4)
print(H)
print(H.mean(), H.shape)  # 全部元素求均值
print(H.mean(axis=0), H.mean(axis=0).shape)  # 第0维对应位置元素求均值，shape:(4)
print(H.mean(axis=1), H.mean(axis=1).shape)  # 第1维对应位置元素求均值，shape:(3)

'''6. 矩阵求和、求均值（保留维度）'''
I = torch.arange(12, dtype=float).reshape(3, 4)
print(I.sum(axis=0, keepdims=True), I.sum(axis=0, keepdims=True).shape)  # 第0维对应位置元素求和，shape:(1, 4)
print(I.mean(axis=0, keepdims=True), I.mean(axis=0, keepdims=True).shape)  # 第0维对应位置元素求均值，shape:(1, 4)
print(I / I.sum(axis=0, keepdims=True))  # 维度个数相同可以广播
print(I / I.sum(axis=0))  # 高版本python在维度个数不同的时候也可以广播

'''7. 向量乘法、向量矩阵乘法、矩阵乘法'''
# 向量乘法：对应位置相乘，得到一个标量
J = torch.arange(4, dtype=float)
K = torch.ones(4, dtype=float) * 2
print(torch.dot(J, K))

# 向量矩阵乘法
L = torch.arange(4, dtype=float)
M = torch.arange(12, dtype=float).reshape(3, 4)
print(torch.mv(M, L), torch.mv(M, L).shape)  # 矩阵的每一行乘以向量，shape:(4)

# 矩阵矩阵乘法
N = torch.arange(6).reshape(2, 3)
O = torch.arange(12).reshape(3, 4)
print(torch.mm(N, O), torch.mm(N, O).shape)  # N的每一行乘以O的每一列形成新的矩阵，shape:(2, 4)

'''8. 向量范数、矩阵范数'''
P = torch.arange(4, dtype=float)
Q = torch.arange(6, dtype=float).reshape(2, 3)
# L1范数（向量）
print(torch.abs(P).sum())
# L2范数（向量）
print(torch.norm(P))
# F范数（矩阵）
print(torch.norm(Q))

