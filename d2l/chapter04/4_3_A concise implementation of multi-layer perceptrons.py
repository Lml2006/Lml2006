import torch # 导入PyTorch深度学习框架核心库

from torch import nn# 导入PyTorch的神经网络模块，用于构建模型层

from d2l import torch as d2l# 导入d2l库的PyTorch接口，提供数据加载、训练可视化等辅助功能
net = nn.Sequential # 构建顺序式神经网络模型
( 

    nn.Flatten(),  # 展平层：将28x28的图像数据转换为784维的一维向量
    nn.Linear(784, 256),  # 全连接层：输入784维，输出256维
    nn.ReLU(),  # ReLU激活函数：引入非线性，解决梯度消失问题
    nn.Linear(256, 10)  # 输出层：输入256维，输出10维（对应Fashion-MNIST的10个类别）
)

def init_weights(m): # 定义权重初始化函数

    if type(m) == nn.Linear: # 仅对全连接层（nn.Linear）进行权重初始化
    
        nn.init.normal_(m.weight, std=0.01) # 用正态分布（均值0，标准差0.01）初始化权重
        

net.apply(init_weights) # 应用权重初始化函数到模型的所有层


batch_size, lr, num_epochs = 256, 0.1, 10 # 超参数设置：批量大小、学习率、训练轮数

loss = nn.CrossEntropyLoss(reduction='none') # 定义交叉熵损失函数（reduction='none'表示不自动求平均，保留每个样本的损失值）

trainer = torch.optim.SGD(net.parameters(), lr=lr) # 定义优化器：随机梯度下降（SGD），传入模型参数和学习率


train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) #加载Fashion-MNIST数据集，生成训练和测试迭代器

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 调用d2l库的训练函数，执行模型训练与评估
