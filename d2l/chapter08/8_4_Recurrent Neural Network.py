import torch
from d2l import torch as d2l
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))


import torch
from d2l import torch as d2l
# X：输入特征，形状(3,1)，3个样本，每个样本1维输入
# W_xh：输入到隐藏层的权重矩阵，形状(1,4)，把1维输入映射到4维隐藏状态
X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
# H：上一时刻的隐藏状态，形状(3,4)，3个样本，隐藏状态维度为4
# W_hh：隐藏状态到隐藏状态的权重矩阵，形状(4,4)，隐藏状态之间的变换权重
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
# RNN隐藏层计算公式：H_new = X @ W_xh + H @ W_hh
# 分开做两次矩阵乘法再相加，是RNN公式原始写法
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# 等价合并写法：把X和H在列维度拼接，权重W_xh、W_hh在行维度拼接
# 一次矩阵乘法，结果和上面完全相等，是工程上常用的合并加速技巧
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
