import torch
from torch import nn
from d2l import torch as d2l
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


import torch
# 导入神经网络模块
from torch import nn
# 导入d2l工具库，提供数据集、训练、绘图等辅助函数
from d2l import torch as d2l
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """
    NiN块：网络中的网络基础模块
    :param in_channels: 输入特征图通道数
    :param out_channels: 输出特征图通道数
    :param kernel_size: 第一个卷积层的卷积核大小
    :param strides: 第一个卷积层的步幅
    :param padding: 第一个卷积层的填充
    :return: 组装好的NiN块序列
    """
    return nn.Sequential(
        # 主卷积层，完成特征提取
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        # 1×1卷积，等价于全连接层，在通道维度做特征变换，不改变高宽
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        # 第二个1×1卷积，进一步对通道特征做非线性变换
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
# 搭建完整NiN网络
net = nn.Sequential(
    # 第一个NiN块，输入单通道灰度图，输出96通道特征
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # 最大池化，窗口3×3，步幅2，降低特征图尺寸
    nn.MaxPool2d(3, stride=2),
    # 第二个NiN块，输入96通道，输出256通道
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    # 第三个NiN块，输入256通道，输出384通道
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    # Dropout层，概率0.5随机置零神经元，防止过拟合
    nn.Dropout(0.5),
    # 标签类别数是10，最后一个NiN块输出通道等于分类类别数
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 自适应平均池化，不管输入高宽，输出固定(1,1)，替代全连接层
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维(batch, channel, H, W)输出转成二维(batch, 10)，用于分类输出
    nn.Flatten())
# 创建测试张量：批量1，通道1，高224，宽224，模拟输入图片
X = torch.rand(size=(1, 1, 224, 224))
# 逐层前向传播，打印每一层输出shape，观察特征图尺寸变化
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
# 超参数设置：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 0.1, 10, 128
# 加载Fashion‑MNIST数据集，图片resize到224×224，得到训练、测试迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 使用GPU(有就用，没有用CPU)训练模型，d2l.train_ch6封装训练、验证、绘图逻辑
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
