import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 导入PyTorch核心库
import torch
# 导入神经网络模块
from torch import nn
# 导入常用函数，如relu
from torch.nn import functional as F
# 导入d2l工具库，提供数据集、训练函数等
from d2l import torch as d2l

# 残差块类，ResNet的基础模块  #@save表示d2l文档标记
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        # 继承父类nn.Module初始化
        super().__init__()
        # 第一个3×3卷积：改变通道数、可以下采样
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第二个3×3卷积：不改变尺寸和通道，padding=1保证尺寸不变
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 是否使用1×1卷积做捷径分支（用来匹配X和Y的通道/尺寸）
        if use_1x1conv:
            # 1×1卷积：调整通道数+配合stride下采样
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            # 不使用捷径卷积，直接恒等映射
            self.conv3 = None
        # 第一个卷积后的批量归一化
        self.bn1 = nn.BatchNorm2d(num_channels)
        # 第二个卷积后的批量归一化
        self.bn2 = nn.BatchNorm2d(num_channels)

    # 前向传播函数
    def forward(self, X):
        # 卷积1 -> BN -> ReLU
        Y = F.relu(self.bn1(self.conv1(X)))
        # 卷积2 -> BN（这里先不加激活，等残差相加后再激活）
        Y = self.bn2(self.conv2(Y))
        # 如果有1×1捷径卷积，对输入X做变换，匹配Y的shape
        if self.conv3:
            X = self.conv3(X)
        # 残差连接：主分支输出 + 捷径分支输入
        Y += X
        # 相加之后再ReLU激活
        return F.relu(Y)

# 构造残差块：输入通道3，输出通道3，不做下采样、不使用1×1卷积
blk = Residual(3,3)
# 构造测试张量：batch=4, channel=3, H=6, W=6
X = torch.rand(4, 3, 6, 6)
# 前向传播
Y = blk(X)
# 输出形状，这里尺寸通道都不变
Y.shape

# 残差块：输入3通道，输出6通道，开启1×1卷积，stride=2（高宽减半）
blk = Residual(3,6, use_1x1conv=True, strides=2)
# 查看输出shape：通道翻倍，高宽缩小一半
blk(X).shape

# ResNet第一部分：输入层
# 7×7卷积 + BN + ReLU + 3×3最大池化，初步提取特征、降采样
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 构造一组残差块的函数
# input_channels：输入通道；num_channels：本组输出通道；num_residuals：残差块数量；first_block：是否是网络第一个残差块组
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        # 组内第一个块，并且不是整个网络第一个残差组：需要下采样+升通道，开启1×1卷积
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 组内其余残差块：恒等捷径，尺寸通道不变
            blk.append(Residual(num_channels, num_channels))
    return blk

# 第2组残差块，2个残差块，是第一个残差组，不做下采样
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# 第3组残差块，2个残差块，通道64→128，高宽减半
b3 = nn.Sequential(*resnet_block(64, 128, 2))
# 第4组残差块，2个残差块，通道128→256，高宽减半
b4 = nn.Sequential(*resnet_block(128, 256, 2))
# 第5组残差块，2个残差块，通道256→512，高宽减半
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 搭建完整ResNet-18（简化版，适配Fashion-MNIST）
net = nn.Sequential(b1, b2, b3, b4, b5,
                    # 自适应平均池化，不管输入多大，输出固定(1,1)
                    nn.AdaptiveAvgPool2d((1,1)),
                    # 展平，去掉多余维度，准备送入全连接层
                    nn.Flatten(),
                    # 全连接分类层，512特征映射到10类
                    nn.Linear(512, 10))

# 测试网络前向传播，单张灰度图：batch=1, channel=1, H=224, W=224
X = torch.rand(size=(1, 1, 224, 224))
# 逐层跑一遍，打印每一层输出形状
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

# 超参数：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 0.05, 10, 256
# 加载Fashion-MNIST数据集，图片resize到96×96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# d2l封装好的训练函数，在GPU上训练
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
