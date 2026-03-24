import torch# 导入PyTorch核心库
from torch import nn# 导入PyTorch的神经网络模块，用于构建模型层
from d2l import torch as d2l# 导入动手学深度学习库的PyTorch版本，提供数据加载和训练工具
batch_size = 256# 定义批量大小，即每次训练时输入模型的样本数量
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 加载Fashion-MNIST数据集，返回训练迭代器和测试迭代器
# 迭代器会按batch_size分批返回数据和标签
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# PyTorch不会自动调整输入形状，因此在全连接层前添加展平层(Flatten)
# 模型结构：先将28x28的图像展平为784维向量，再通过线性层映射到10个类别
def init_weights(m):# 定义权重初始化函数，用于初始化模型参数
     if type(m) == nn.Linear:# 判断当前层是否为线性层(Linear)
        nn.init.normal_(m.weight, std=0.01) #对线性层的权重进行正态分布初始化，均值为0，标准差为0.01
net.apply(init_weights);# 对模型net的所有层应用权重初始化函数
loss = nn.CrossEntropyLoss(reduction='none')# 定义交叉熵损失函数，reduction='none'表示不自动对损失求和或取平均，保留每个样本的损失值
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# 定义随机梯度下降(SGD)优化器，用于更新模型参数
# 传入模型的可训练参数，设置学习率lr=0.1
num_epochs = 10# 定义训练的轮数(epoch)，即完整遍历训练集的次数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 调用d2l库的训练函数，执行模型训练和测试
# 参数依次为：模型、训练迭代器、测试迭代器、损失函数、训练轮数、优化器
