import torch
from torch import nn
from d2l import torch as d2l
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
  X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
pool2d(X, (2, 2), 'avg')
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
pool2d = nn.MaxPool2d(3)
pool2d(X)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
X = torch.cat((X, X + 1), 1)
X
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)


# 导入PyTorch核心库
import torch
# 导入神经网络模块
from torch import nn
# 导入d2l学习工具库
from d2l import torch as d2l
def pool2d(X, pool_size, mode='max'):
    """
    手动实现二维池化层
    :param X: 输入二维张量
    :param pool_size: 池化窗口大小 (ph, pw)
    :param mode: 池化模式 'max'最大池化 / 'avg'平均池化
    :return: 池化输出二维张量
    """
    # 解包池化窗口高度、宽度
    p_h, p_w = pool_size
    # 计算输出特征图尺寸：(H - ph + 1, W - pw + 1)，无填充、步幅为1
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历输出特征图每一行
    for i in range(Y.shape[0]):
        # 遍历输出特征图每一列
        for j in range(Y.shape[1]):
            if mode == 'max':
                # 最大池化：取窗口内最大值
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                # 平均池化：取窗口内平均值
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
# 构造3×3二维输入矩阵
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 使用2×2窗口执行手动最大池化，默认步幅1、无填充
pool2d(X, (2, 2))
# 使用2×2窗口执行手动平均池化
pool2d(X, (2, 2), 'avg')
# 构造4×4特征图，形状：(batch=1, channel=1, height=4, width=4) NCHW格式
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
# 使用PyTorch内置最大池化，窗口3×3，默认stride=3，padding=0
pool2d = nn.MaxPool2d(3)
pool2d(X)
# 内置最大池化：3×3窗口，填充1，步幅2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
# 非正方形池化窗口：高2、宽3；步幅(2,3)；填充(高度方向0，宽度方向1)
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
# 在通道维度拼接特征，通道数由1变为2：[1,2,4,4]
X = torch.cat((X, X + 1), 1)
X
# 双通道输入下，3×3最大池化（池化层独立作用于每个通道）
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
