import torch
from torch import nn
from d2l import torch as d2l
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
blk = transition_block(23, 10)
blk(Y).shape
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
# num_channels为当前的通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


import torch
from torch import nn
from d2l import torch as d2l

def conv_block(input_channels, num_channels):
    """
    稠密网络里的基础卷积块（BN -> ReLU -> 3×3卷积）
    :param input_channels: 输入特征图通道数
    :param num_channels: 卷积输出通道数（增长率growth_rate）
    :return: 序列化网络模块
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),  # 批量归一化，加速收敛
        nn.ReLU(),                       # 激活函数
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)  # 3×3卷积，padding=1保持高宽不变
    )


class DenseBlock(nn.Module):
    """
    DenseBlock 稠密块：每一层卷积都把前面所有层的输出在通道维度拼接
    """
    def __init__(self, num_convs, input_channels, num_channels):
        """
        :param num_convs: 稠密块内部包含多少个conv_block
        :param input_channels: 稠密块的输入通道数
        :param num_channels: 每个conv_block新增输出通道（增长率）
        """
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 第i个卷积块的输入通道 = 原始输入 + 前面i个卷积新增的通道
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        """前向传播：稠密连接核心，通道拼接"""
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出（dim=1对应通道维度 NCHW）
            X = torch.cat((X, Y), dim=1)
        return X


# 测试稠密块：2层卷积，输入通道3，每层新增10通道
blk = DenseBlock(2, 3, 10)
# 构造测试张量：batch=4, channel=3, H=8, W=8
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape  # 输出通道：3 + 2*10 = 23


def transition_block(input_channels, num_channels):
    """
    转换层（过渡块）：用来降低通道数、压缩特征图尺寸，控制模型复杂度
    BN -> ReLU -> 1×1卷积降维 -> 平均池化下采样
    :param input_channels: 输入通道
    :param num_channels: 1×1卷积压缩后的通道数
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),  # 1×1卷积减少通道
        nn.AvgPool2d(kernel_size=2, stride=2)  # 2×2平均池化，高宽减半
    )


# 测试过渡块：输入通道23，压缩到10通道，尺寸减半
blk = transition_block(23, 10)
blk(Y).shape


# b1：DenseNet最开始的初始模块，和ResNet开头类似
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 7×7大卷积，下采样
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)       # 最大池化继续下采样
)

# num_channels：记录当前通道总数；growth_rate：稠密块每层新增通道数
num_channels, growth_rate = 64, 32
# 4个稠密块，每个稠密块内部有4个卷积
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    # 添加一个稠密块
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 稠密块输出通道 = 原有通道 + 卷积个数 × 增长率
    num_channels += num_convs * growth_rate
    # 稠密块之间插入transition_block，最后一个稠密块后面不加
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2  # 更新通道数为压缩后的通道


# 搭建完整DenseNet网络
net = nn.Sequential(
    b1, *blks,                                 # 初始模块 + 4组稠密块+过渡块
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),              # 自适应平均池化，不管输入多大，输出1×1特征图
    nn.Flatten(),                              # 展平，去掉空间维度
    nn.Linear(num_channels, 10)                # 全连接分类头，Fashion-MNIST共10类
)

# 超参数设置：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 0.1, 10, 256
# 加载Fashion-MNIST数据集，图片resize到96×96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l封装好的训练函数，在GPU上训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
