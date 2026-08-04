import torch
from torch import nn
from d2l import torch as d2l
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))
net = vgg(conv_arch)
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 导入PyTorch主库
import torch
# 导入神经网络模块，包含卷积、全连接、激活等网络层
from torch import nn
# 导入d2l工具库，提供数据集、训练函数、可视化等工具
from d2l import torch as d2l
def vgg_block(num_convs, in_channels, out_channels):
    """
    构建VGG的一个卷积块
    :param num_convs: 当前块内卷积层的数量
    :param in_channels: 输入特征图通道数
    :param out_channels: 输出特征图通道数
    :return: 封装好的卷积块Sequential容器
    """
    # 用来存放该块内所有网络层
    layers = []
    # 循环添加num_convs个卷积+ReLU组合
    for _ in range(num_convs):
        # 2维卷积：3×3卷积核，padding=1保证卷积后图像尺寸不变
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        # 添加ReLU激活函数，引入非线性
        layers.append(nn.ReLU())
        # 下一层卷积的输入通道等于当前输出通道
        in_channels = out_channels
    # 添加最大池化层：2×2池化核，步幅2，高宽各缩小为原来1/2
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    # 将层列表解包，封装为Sequential顺序模型返回
    return nn.Sequential(*layers)
# VGG网络结构配置：(该块卷积层数,输出通道数)，一共5个卷积块
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    """
    根据配置构建完整VGG网络
    :param conv_arch: 卷积块配置列表，每个元素(卷积层数,输出通道)
    :return: 完整VGG顺序网络模型
    """
    # 存储所有卷积块
    conv_blks = []
    # Fashion‑MNIST是单通道灰度图，初始输入通道为1
    in_channels = 1
    # 卷积层部分：循环创建每一个VGG块
    for (num_convs, out_channels) in conv_arch:
        # 调用vgg_block生成块，加入列表
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        # 更新下一个块的输入通道
        in_channels = out_channels
    return nn.Sequential(
        # 放入全部卷积块
        *conv_blks,
        # Flatten：把4维特征图展平为2维，用于后续全连接输入
        nn.Flatten(),
        # 全连接层部分
        # 经过5次池化，224→7，特征图大小7×7；out_channels为最后块输出通道
        nn.Linear(out_channels * 7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),  # dropout0.5随机失活一半神经元，抑制过拟合
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        # 输出层：10分类，对应Fashion‑MNIST10个类别
        nn.Linear(4096, 10)
    )
# 按照原始配置实例化VGG网络
net = vgg(conv_arch)
# 构造测试输入：batch=1，通道1，图像224×224，随机张量
X = torch.randn(size=(1, 1, 224, 224))
# 逐层遍历网络，打印每一块输出shape，观察尺寸变化
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
# 缩放系数4，用来缩小通道数，减小模型参数量，方便训练
ratio = 4
# 生成缩小版VGG配置：卷积层数不变，通道数除以4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# 重新实例化轻量化VGG网络
net = vgg(small_conv_arch)
# 超参数：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 0.05, 10, 128
# 加载Fashion‑MNIST数据集，resize把图片放大到224×224适配VGG输入要求
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# d2l封装好的训练函数，使用GPU(如有)训练，输出loss、训练/测试精度
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
