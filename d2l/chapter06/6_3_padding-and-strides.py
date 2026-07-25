import torch
from torch import nn
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])
# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape


import torch
from torch import nn
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小(batch_size)=1、通道数(in_channels)=1
    # 将二维矩阵X (H,W) 扩充为四维张量 (N, C, H, W)，满足Conv2d输入格式
    X = X.reshape((1, 1) + X.shape)
    # 执行卷积运算
    Y = conv2d(X)
    # 去掉前两个维度：批量大小和通道，只返回输出特征图的高、宽
    return Y.reshape(Y.shape[2:])
# 案例1：3×3卷积核，上下左右每一侧填充1行/1列，总共四周增加2行2列
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))  # 构造8×8二维输入矩阵
print(comp_conv2d(conv2d, X).shape)
# 案例2：矩形卷积核(5行3列)，上下填充2，左右填充1
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)
# 案例3：3×3卷积核，padding=1，步幅stride=2，输出尺寸会减半
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)
# 案例4：矩形卷积核(3行5列)；上下无填充、左右填充1；高方向步幅3，宽方向步幅4
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
