%matplotlib inline 
import torch  
from torch import nn 
from d2l import torch as d2l
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5 
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05 
train_data = d2l.synthetic_data(true_w, true_b, n_train) 
train_iter = d2l.load_array(train_data, batch_size) 
test_data = d2l.synthetic_data(true_w, true_b, n_test) 
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
def init_params():     
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)     
    b = torch.zeros(1, requires_grad=True)     
    return [w, b]
def l2_penalty(w):     
    return torch.sum(w.pow(2)) / 2
def train(lambd):     
    w, b = init_params()     
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss     
    num_epochs, lr = 100, 0.003     
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',                             
                          xlim=[5, num_epochs], legend=['train', 'test'])     
  for epoch in range(num_epochs):         
      for X, y in train_iter:             
      # 增加了L2范数惩罚项，            
      # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量             
      l = loss(net(X), y) + lambd * l2_penalty(w)             
      l.sum().backward()             
      d2l.sgd([w, b], lr, batch_size)         
    if (epoch + 1) % 5 == 0:             
        animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),                                      
                                 d2l.evaluate_loss(net, test_iter, loss)))     
  print('w的L2范数是：', torch.norm(w).item())
train(lambd=0)
train(lambd=3)
def train_concise(wd):     
  net = nn.Sequential(nn.Linear(num_inputs, 1))     
  for param in net.parameters():         
      param.data.normal_()     
  loss = nn.MSELoss(reduction='none')     
  num_epochs, lr = 100, 0.003     
# 偏置参数没有衰减     
  trainer = torch.optim.SGD([         
      {"params":net[0].weight,'weight_decay': wd},         
      {"params":net[0].bias}], lr=lr)     
  animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',                             
                            xlim=[5, num_epochs], legend=['train', 'test'])     
  for epoch in range(num_epochs):         
      for X, y in train_iter:             
          trainer.zero_grad()             
          l = loss(net(X), y)             
          l.mean().backward()             
          trainer.step()         
      if (epoch + 1) % 5 == 0:             
          animator.add(epoch + 1,                          
                       (d2l.evaluate_loss(net, train_iter, loss),                           
                        d2l.evaluate_loss(net, test_iter, loss)))     
  print('w的L2范数：', net[0].weight.norm().item())
  train_concise(0)
  train_concise(3)

# Jupyter Notebook 魔法命令：让 matplotlib 绘制的图表直接内嵌在 notebook 单元格中显示
%matplotlib inline
# 导入 PyTorch 核心库
import torch
# 导入 PyTorch 神经网络模块，封装了层、损失函数、优化器等组件
from torch import nn
# 导入《动手学深度学习》配套工具库 d2l，封装了数据生成、绘图、通用训练工具等
from d2l import torch as d2l
# ========== 1. 构造过拟合实验数据集 ==========
# 超参数定义：训练样本20个，测试样本100个，输入特征维度200，批量大小5
# 训练样本数远小于特征维度，刻意构造「容易过拟合」的实验场景
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
# 定义真实线性模型的参数：所有权重为0.01，偏置为0.05
# 真实权重数值很小，模拟现实中多数特征对标签影响微弱的场景
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
# 生成训练集：基于真实线性模型 + 高斯噪声生成合成样本
train_data = d2l.synthetic_data(true_w, true_b, n_train)
# 构造训练集批量迭代器：按batch_size加载数据，训练模式默认打乱样本顺序
train_iter = d2l.load_array(train_data, batch_size)
# 生成测试集：同分布的合成数据，用于评估模型泛化能力
test_data = d2l.synthetic_data(true_w, true_b, n_test)
# 构造测试集批量迭代器：is_train=False 表示测试时不打乱样本顺序
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
# ========== 2. 从零实现：手动定义参数与L2正则 ==========
def init_params():
    """初始化线性回归模型的可训练参数 w 和 b"""
    # 权重w：形状(输入维度, 1)，用均值0、标准差1的正态分布随机初始化
    # requires_grad=True 表示需要追踪梯度，用于反向传播更新参数
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    # 偏置b：初始化为标量0，同样开启梯度计算
    b = torch.zeros(1, requires_grad=True)
    return [w, b]
ef l2_penalty(w):
    """计算权重 w 的 L2 范数惩罚项，公式：(1/2) * ||w||²
    除以2是为了求导后形式更简洁，与权重衰减公式完全等价
    """
    # pow(2)对每个权重元素平方，sum()对所有权重平方求和
    return torch.sum(w.pow(2)) / 2
def train(lambd):
    """
    从零实现带L2正则化的线性回归训练
    :param lambd: L2正则化系数λ，控制惩罚强度，λ越大权重约束越强
    """
    # 初始化模型参数
    w, b = init_params()
    # 定义线性回归网络（匿名函数封装）和平方损失函数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss

    # 训练超参数：总训练轮数100，学习率0.003
    num_epochs, lr = 100, 0.003
    # 初始化动画绘制器：实时绘制训练/测试损失曲线
    # yscale='log'：y轴使用对数刻度，更清晰观察损失下降过程
    # xlim：x轴显示范围（跳过前5轮波动）；legend：曲线图例
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    # 遍历每一轮训练
    for epoch in range(num_epochs):
        # 遍历训练集的每个批量
        for X, y in train_iter:
            # 总损失 = 原始平方损失 + L2正则化惩罚项
            # 广播机制会让标量惩罚项自动扩展为与损失同形状的向量，实现逐样本加罚
            l = loss(net(X), y) + lambd * l2_penalty(w)
            # 对批量内所有样本的损失求和后反向传播，计算参数梯度
            l.sum().backward()
            # 使用小批量随机梯度下降(SGD)更新参数 w 和 b
            d2l.sgd([w, b], lr, batch_size)
        # 每5轮训练评估一次损失，更新动画图表
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    # 训练结束，打印最终权重的L2范数：范数越小，说明权重被约束得越紧凑
    print('w的L2范数是：', torch.norm(w).item())
# 实验1：不使用正则化（λ=0），模型会发生严重过拟合
train(lambd=0)
# 实验2：使用λ=3的L2正则化，通过惩罚大权重缓解过拟合
train(lambd=3)
# ========== 3. 简洁实现：调用PyTorch官方API实现权重衰减 ==========
def train_concise(wd):
    """
    PyTorch 简洁版带权重衰减的线性回归训练
    :param wd: weight_decay 权重衰减系数，等价于手动实现中的正则化系数λ
    """
    # 使用Sequential快速搭建单层线性层，等价于线性回归模型
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    # 遍历网络所有参数，用正态分布初始化，与手动实现保持一致
    for param in net.parameters():
        param.data.normal_()
    # 定义均方误差损失，reduction='none' 表示返回每个样本的独立损失，不做聚合
    loss = nn.MSELoss(reduction='none')
    # 训练超参数与手动实现对齐
    num_epochs, lr = 100, 0.003
    # 定义SGD优化器，分别为权重和偏置设置不同的更新规则
    # 权重参数启用 weight_decay（权重衰减），底层等价于L2正则化
    # 偏置参数不使用权重衰减，符合深度学习常规实践
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}
    ], lr=lr)
    # 初始化动画绘制器，配置与手动版一致
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    # 遍历每一轮训练
    for epoch in range(num_epochs):
        # 遍历每个批量数据
        for X, y in train_iter:
            # 清零优化器中累积的梯度，避免上一批梯度累加
            trainer.zero_grad()
            # 前向传播：计算模型预测值，并得到每个样本的损失
            l = loss(net(X), y)
            # 对批量损失求均值后反向传播，计算参数梯度
            l.mean().backward()
            # 优化器执行一步参数更新（含权重衰减逻辑）
            trainer.step()
        # 每5轮评估一次损失，更新图表
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                          d2l.evaluate_loss(net, test_iter, loss)))
    # 打印最终权重的L2范数
    print('w的L2范数：', net[0].weight.norm().item())
# 简洁版实验1：不使用权重衰减
train_concise(0)
# 简洁版实验2：使用权重衰减系数3
train_concise(3)
