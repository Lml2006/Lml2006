import math 
import numpy as np 
import torch 
from torch import nn 
from d2l import torch as d2l
max_degree = 20  # 多项式的最大阶数 
n_train, n_test = 100, 100  # 训练和测试数据集大小 
true_w = np.zeros(max_degree)  # 分配大量的空间 
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6]) 
 features = np.random.normal(size=(n_train + n_test, 1)) 
np.random.shuffle(features) 
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1)) 
for i in range(max_degree):     
  poly_features[:, i] /= math.gamma(i + 1)  
# gamma(n)=(n-1)! # labels的维度:(n_train+n_test,) 
  labels = np.dot(poly_features, true_w) 
  labels += np.random.normal(scale=0.1, size=labels.shape)
  # NumPy ndarray转换为tensor 
  true_w, features, poly_features, labels = [torch.tensor(x, dtype=     
                                                          torch.float32) for x in [true_w, features, poly_features, labels]]
  features[:2], poly_features[:2, :], labels[:2]
  (tensor([[-0.7408],
         [ 0.9021]]),
 tensor([[ 1.0000e+00, -7.4078e-01,  2.7438e-01, -6.7751e-02,  1.2547e-02,
          -1.8589e-03,  2.2951e-04, -2.4288e-05,  2.2490e-06, -1.8511e-07,
           1.3713e-08, -9.2346e-10,  5.7007e-11, -3.2484e-12,  1.7188e-13,
          -8.4884e-15,  3.9300e-16, -1.7125e-17,  7.0478e-19, -2.7478e-20],
         [ 1.0000e+00,  9.0208e-01,  4.0687e-01,  1.2234e-01,  2.7591e-02,
           4.9777e-03,  7.4838e-04,  9.6443e-05,  1.0875e-05,  1.0900e-06,
           9.8325e-08,  8.0633e-09,  6.0614e-10,  4.2061e-11,  2.7101e-12,
           1.6298e-13,  9.1889e-15,  4.8759e-16,  2.4436e-17,  1.1602e-18]]),
 tensor([2.9074, 5.2200]))
def evaluate_loss(net, data_iter, loss):  #@save     
    """评估给定数据集上模型的损失"""     
    metric = d2l.Accumulator(2)  # 损失的总和,样本数量     
  for X, y in data_iter:         
    out = net(X)         
    y = y.reshape(out.shape)         
    l = loss(out, y)         
    metric.add(l.sum(), l.numel())     
  return metric[0] / metric[1]
def train(train_features, test_features, train_labels, test_labels,           
          num_epochs=400):     
        loss = nn.MSELoss(reduction='none')     
        input_shape = train_features.shape[-1]     
        # 不设置偏置，因为我们已经在多项式中实现了它     
        net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))     
        batch_size = min(10, train_labels.shape[0])     
        train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),                                 
                                        batch_size)     
        test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),                                
                                       batch_size, is_train=False)     
        trainer = torch.optim.SGD(net.parameters(), lr=0.01)     
        animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',                             
                                    xlim=[1, num_epochs], ylim=[1e-3, 1e2],                             
                                    legend=['train', 'test'])     
        for epoch in range(num_epochs):         
            d2l.train_epoch_ch3(net, train_iter, loss, trainer)         
            if epoch == 0 or (epoch + 1) % 20 == 0:             
                animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),                                      
                                         evaluate_loss(net, test_iter, loss)))     
        print('weight:', net[0].weight.data.numpy())
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3! 
train(poly_features[:n_train, :4], poly_features[n_train:, :4],       
      labels[:n_train], labels[n_train:])
# 从多项式特征中选择前2个维度，即1和x 
train(poly_features[:n_train, :2], poly_features[n_train:, :2],       
      labels[:n_train], labels[n_train:])
# 从多项式特征中选取所有维度 
train(poly_features[:n_train, :], poly_features[n_train:, :],       
      labels[:n_train], labels[n_train:], num_epochs=1500)

# ===================== 1. 导入依赖工具库 =====================
import math                     # Python标准数学库，用于计算阶乘（gamma函数）
import numpy as np              # NumPy数值计算库，用于生成人工数据、矩阵运算
import torch                    # PyTorch核心库
from torch import nn            # PyTorch神经网络模块
from d2l import torch as d2l    # 教材配套工具库，提供训练、绘图等封装函数


# ===================== 2. 人工构造多项式数据集 =====================
max_degree = 20                 # 多项式的最高阶数（0~19阶，共20维特征）
n_train, n_test = 100, 100     # 训练集样本数100，测试集样本数100

# 初始化真实权重向量：真实模型只有前4阶非零，其余高阶系数全为0
# 即真实函数为：y = 5 + 1.2*x - 3.4*(x²/2!) + 5.6*(x³/3!) （3次多项式）
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 随机生成自变量x：总共200个样本（训练+测试），每个样本1个特征，服从标准正态分布
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)    # 打乱数据顺序，避免分布不均

# 生成多项式特征矩阵：shape为(200, 20)
# 每一行对应一个样本的各阶次幂：[x^0, x^1, x^2, ..., x^19]
# np.arange(max_degree).reshape(1, -1) 生成[0,1,...,19]，利用广播机制逐元素求幂
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))

# 对每一阶特征除以 i!（阶乘），做数值缩放
# math.gamma(i+1) = i!，例如 gamma(4)=3!=6
# 目的：避免高次项数值过大导致数值不稳定，让不同阶特征的数值尺度保持在相近范围
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)

# 计算标签（真实值）：多项式特征与真实权重做矩阵乘法，得到干净的y值
labels = np.dot(poly_features, true_w)
# 给标签加入高斯噪声（标准差0.1），模拟真实数据中的随机误差
labels += np.random.normal(scale=0.1, size=labels.shape)

# 将NumPy数组统一转换为PyTorch的float32张量，供后续模型训练使用
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]
]

# 查看前2个样本的原始特征、多项式特征、标签（验证数据生成是否正确）
# features[:2], poly_features[:2, :], labels[:2]


# ===================== 3. 定义数据集损失评估函数 =====================
def evaluate_loss(net, data_iter, loss):
    """
    计算模型在给定数据集上的平均损失
    参数：
        net: 神经网络模型
        data_iter: 数据迭代器
        loss: 损失函数
    返回：
        所有样本的平均损失值
    """
    metric = d2l.Accumulator(2)  # 累加器：累计「总损失值」和「样本总数」两个量
    for X, y in data_iter:
        out = net(X)             # 前向传播，得到模型预测值
        y = y.reshape(out.shape) # 统一标签和预测值的形状，避免广播错误
        l = loss(out, y)         # 计算每个样本的损失
        metric.add(l.sum(), l.numel())  # 累加总损失 + 累加样本数量
    return metric[0] / metric[1] # 返回平均损失 = 总损失 / 样本总数


# ===================== 4. 定义训练函数（含可视化） =====================
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    """
    训练线性回归模型，并实时绘制训练/测试损失曲线
    参数：
        train_features: 训练集特征
        test_features: 测试集特征
        train_labels: 训练集标签
        test_labels: 测试集标签
        num_epochs: 训练总轮数
    """
    # 使用均方误差MSE作为损失函数（回归任务的标准损失）
    # reduction='none'：返回每个样本的独立损失，不自动求和/平均
    loss = nn.MSELoss(reduction='none')
    
    input_shape = train_features.shape[-1]  # 输入特征的维度（多项式阶数）
    
    # 定义模型：单层线性网络，不设置偏置bias
    # 原因：多项式的0阶项 x^0 = 1，已经作为第一维特征，等价于偏置项
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    
    # 批量大小：取10和训练集样本数中的较小值，避免样本太少时报错
    batch_size = min(10, train_labels.shape[0])
    
    # 构造训练集和测试集的数据迭代器
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    
    # 优化器：随机梯度下降SGD，更新模型所有可训练参数
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    # 动画绘图器：实时绘制损失曲线
    # yscale='log'：纵坐标用对数刻度，方便观察不同量级的损失变化
    animator = d2l.Animator(
        xlabel='epoch', ylabel='loss', yscale='log',
        xlim=[1, num_epochs], ylim=[1e-3, 1e2],
        legend=['train', 'test']
    )
    
    # 开始训练循环
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)  # 完成一轮训练
        
        # 第1轮结束 + 每20轮结束，记录并绘制一次损失
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(
                epoch + 1,
                (
                    evaluate_loss(net, train_iter, loss),  # 训练集平均损失
                    evaluate_loss(net, test_iter, loss)    # 测试集平均损失
                )
            )
    
    # 训练结束后，打印模型学到的权重参数
    print('weight:', net[0].weight.data.numpy())


# ===================== 5. 实验1：匹配真实模型阶数（4阶特征）→ 正常拟合 =====================
# 只使用前4维特征：x^0, x^1, x^2/2!, x^3/3!，和真实数据生成的阶数完全一致
# 预期效果：训练损失和测试损失都很低，模型泛化能力好，属于「正常拟合」
train(
    poly_features[:n_train, :4],   # 训练集：前100个样本，前4阶特征
    poly_features[n_train:, :4],   # 测试集：后100个样本，前4阶特征
    labels[:n_train],              # 训练集标签
    labels[n_train:]               # 测试集标签
)


# ===================== 6. 实验2：低阶特征（2阶）→ 欠拟合 =====================
# 只使用前2维特征：x^0, x^1，模型复杂度远低于真实数据的3次多项式
# 预期效果：训练损失和测试损失都很高，模型连训练数据都学不好，属于「欠拟合」
train(
    poly_features[:n_train, :2],
    poly_features[n_train:, :2],
    labels[:n_train],
    labels[n_train:]
)


# ===================== 7. 实验3：全阶特征（20阶）→ 过拟合 =====================
# 使用全部20维特征，模型复杂度远高于真实数据的3次多项式
# 预期效果：训练损失非常低，但测试损失很高，模型记住了训练噪声、泛化能力差，属于「过拟合」
# 增加训练轮数到1500，让模型充分拟合训练集
train(
    poly_features[:n_train, :],
    poly_features[n_train:, :],
    labels[:n_train],
    labels[n_train:],
    num_epochs=1500
)
