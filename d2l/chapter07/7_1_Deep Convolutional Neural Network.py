import torch
from torch import nn
from d2l import torch as d2l
net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10))
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 导入PyTorch核心库
import torch
# 导入神经网络模块
from torch import nn
# 导入d2l学习工具库，封装数据集、训练、可视化等工具
from d2l import torch as d2l
# 搭建AlexNet网络，使用Sequential有序容器堆叠网络层
net = nn.Sequential(
    # 卷积层1：输入通道1(灰度图Fashion-MNIST)，输出96通道
    # kernel_size=11：11×11大卷积核，大范围捕获图像特征
    # stride=4：步幅4，大幅缩小特征图尺寸
    # padding=1：边缘填充1
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
    nn.ReLU(),  # 激活函数ReLU，替代Sigmoid缓解梯度消失
    # 最大池化层：3×3池化窗口，步幅2，下采样降低特征图尺寸
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 卷积层2：输入96通道，输出256通道
    # kernel_size=5，padding=2：保持输出特征图宽高不变
    nn.Conv2d(96, 256, kernel_size=5, padding=2),
    nn.ReLU(),
    # 最大池化下采样
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 卷积层3：3×3小卷积核，padding=1，特征图尺寸维持不变
    nn.Conv2d(256, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    # 卷积层4：继续提取高维特征，通道数保持384
    nn.Conv2d(384, 384, kernel_size=3, padding=1),
    nn.ReLU(),
    # 卷积层5：通道数回落至256
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    # 最后一次池化，完成卷积部分下采样
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Flatten展平层：把四维[N,C,H,W]特征图转为二维[N, feature_num]送入全连接层
    nn.Flatten(),
    # 全连接层1：输入特征维度6400，输出4096个神经元
    nn.Linear(6400, 4096),
    nn.ReLU(),
    # Dropout正则化，概率0.5随机置零神经元，抑制过拟合
    nn.Dropout(p=0.5),
    # 全连接层2
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    # 输出层：最终分为10个类别(Fashion-MNIST共10类服饰)
    # 原始AlexNet论文输出1000类ImageNet
    nn.Linear(4096, 10))
# 创建测试张量：批量1，通道1(灰度图)，224×224输入图像
X = torch.randn(1, 1, 224, 224)
# 逐层前向传播，打印每一层输出张量形状，检验网络维度是否正确
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 设置批次大小
batch_size = 128
# 加载Fashion-MNIST数据集，图像resize到224×224适配AlexNet输入
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
# 学习率、训练轮数超参数
lr, num_epochs = 0.01, 10
# d2l封装的训练函数：使用GPU(如有)训练AlexNet，输出训练损失、测试精度
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
