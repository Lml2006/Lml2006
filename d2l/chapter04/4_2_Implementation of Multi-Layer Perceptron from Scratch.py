import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.predict_ch3(net, test_iter)



# ===================== 1. 导入工具包 =====================
import torch                  # PyTorch核心库：处理数据、自动算梯度
from torch import nn          # PyTorch的神经网络模块：用来定义参数、层
from d2l import torch as d2l  # 教材配套工具包：提供数据集、训练、可视化等现成函数


# ===================== 2. 准备训练数据 =====================
batch_size = 256  # 批量大小：每次训练同时处理256张图片
# 加载Fashion-MNIST数据集（衣服/鞋子/包等10类图片），返回训练集和测试集的数据迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# ===================== 3. 定义网络结构的维度和参数 =====================
# 输入维度：28×28的图片展平成784个像素
# 输出维度：10个类别（对应10种服饰）
# 隐藏层维度：256个神经元（中间层，用来提取更复杂的特征）
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 第一层权重：输入层→隐藏层，随机初始化（乘以0.01是为了让初始值别太大）
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 第一层偏置：隐藏层的偏置项，初始化为0
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

# 第二层权重：隐藏层→输出层，同样随机初始化并缩小
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# 第二层偏置：输出层的偏置项，初始化为0
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 把所有可训练的参数打包起来，方便优化器一起更新
params = [W1, b1, W2, b2]


# ===================== 4. 定义激活函数（ReLU） =====================
def relu(X):
    """ReLU激活函数：小于0的值直接变成0，大于0的值保持不变
    作用：给网络加入非线性能力，不然多层网络也和一层没区别"""
    a = torch.zeros_like(X)  # 生成和X形状一样、全是0的张量
    return torch.max(X, a)  # 每个位置取X和0里的最大值，实现ReLU


# ===================== 5. 定义前向传播的网络 =====================
def net(X):
    """前向传播：把数据从输入层传到输出层，得到预测结果"""
    # 把28×28的图片展平成一维向量，方便后续计算（-1表示自动适配批量维度）
    X = X.reshape((-1, num_inputs))
    # 第一层计算：矩阵乘法（@是矩阵乘法符号）+偏置，再经过ReLU激活
    H = relu(X @ W1 + b1)
    # 第二层计算：隐藏层的结果再经过一次矩阵乘法+偏置，得到最终预测分数
    return H @ W2 + b2


# ===================== 6. 定义训练的核心组件 =====================
loss = nn.CrossEntropyLoss(reduction='none')  # 损失函数：交叉熵，用来衡量预测和真实标签的差距
num_epochs, lr = 10, 0.1                     # 训练超参数：训练10轮，学习率0.1（控制参数更新的步长）
updater = torch.optim.SGD(params, lr=lr)     # 优化器：随机梯度下降（SGD），用来更新参数，让损失越来越小


# ===================== 7. 开始训练模型 =====================
# 调用教材提供的训练函数：自动循环训练+验证，还会画损失和准确率的图
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


# ===================== 8. 用训练好的模型做预测 =====================
# 调用教材提供的预测函数：在测试集上看模型分类效果
d2l.predict_ch3(net, test_iter)
