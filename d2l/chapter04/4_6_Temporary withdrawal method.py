import torch 
from torch import nn 
from d2l import torch as d2l 
 
def dropout_layer(X, dropout):     
     assert 0 <= dropout <= 1     
      # 在本情况中，所有元素都被丢弃     
     if dropout == 1:         
         return torch.zeros_like(X)     
      # 在本情况中，所有元素都被保留     
     if dropout == 0:         
         return X     
      mask = (torch.rand(X.shape) > dropout).float()     
      return mask * X / (1.0 - dropout)
X= torch.arange(16, dtype = torch.float32).reshape((2, 8)) 
print(X) 
print(dropout_layer(X, 0.)) 
print(dropout_layer(X, 0.5)) 
print(dropout_layer(X, 1.))
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5 
class Net(nn.Module):     
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,                  
                 is_training = True):         
                   super(Net, self).__init__()         
                   self.num_inputs = num_inputs         
                   self.training = is_training         
                   self.lin1 = nn.Linear(num_inputs, num_hiddens1)         
                   self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)         
                   self.lin3 = nn.Linear(num_hiddens2, num_outputs)         
                   self.relu = nn.ReLU() 
    def forward(self, X):         
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))         
        # 只有在训练模型时才使用dropout         
        if self.training == True:             
            # 在第一个全连接层之后添加一个dropout层             
            H1 = dropout_layer(H1, dropout1)         
        H2 = self.relu(self.lin2(H1))         
        if self.training == True:             
            # 在第二个全连接层之后添加一个dropout层             
            H2 = dropout_layer(H2, dropout2)         
        out = self.lin3(H2)         
        return out 
 
 net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
num_epochs, lr, batch_size = 10, 0.5, 256 
loss = nn.CrossEntropyLoss(reduction='none') 
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
trainer = torch.optim.SGD(net.parameters(), lr=lr) 
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
net = nn.Sequential(nn.Flatten(),         
                    nn.Linear(784, 256),         
                    nn.ReLU(),         
                    # 在第一个全连接层之后添加一个dropout层         
                    nn.Dropout(dropout1),         
                    nn.Linear(256, 256),         
                    nn.ReLU(),         
                    # 在第二个全连接层之后添加一个dropout层         
                    nn.Dropout(dropout2),         
                    nn.Linear(256, 10)) 
def init_weights(m):     
    if type(m) == nn.Linear:         
        nn.init.normal_(m.weight, std=0.01) 
net.apply(init_weights);
trainer = torch.optim.SGD(net.parameters(), lr=lr) 
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


# 导入PyTorch基础库与神经网络模块
import torch 
from torch import nn 
# 导入d2l工具库，封装了数据集加载、训练流程等常用函数
from d2l import torch as d2l 
# -------------------------- 1. 手动实现Dropout层 --------------------------
# 自定义Dropout计算函数：输入张量X，丢弃概率dropout（0~1）
def dropout_layer(X, dropout):     
    # 断言校验：丢弃概率必须在合法区间内，否则直接报错
    assert 0 <= dropout <= 1     
    # 边界情况1：丢弃概率=1 → 所有神经元全部被置0丢弃
    if dropout == 1:         
        return torch.zeros_like(X)     
    # 边界情况2：丢弃概率=0 → 所有神经元全部保留，直接返回原张量
    if dropout == 0:         
        return X        
    # 生成随机掩码：生成与X形状完全一致的随机数（0~1均匀分布）
    # 大于dropout的位置为True（保留神经元），否则为False（丢弃神经元）
    # 转为float后，保留位置=1，丢弃位置=0
    mask = (torch.rand(X.shape) > dropout).float()         
    # 核心计算：掩码点乘原张量实现随机丢弃，同时除以 (1-dropout) 做"倒置缩放"
    # 原理：随机丢弃后，输出的数学期望会变为原来的 (1-dropout) 倍
    # 除以(1-dropout)可保证训练时输出期望与测试时（全保留）一致，维持数值尺度稳定
    return mask * X / (1.0 - dropout)
# -------------------------- 2. Dropout效果测试 --------------------------
# 构造测试输入：0~15的一维浮点张量，reshape为 2行×8列 的矩阵
X = torch.arange(16, dtype=torch.float32).reshape((2, 8)) 
print("原始张量：\n", X) 
print("dropout=0（全保留）：\n", dropout_layer(X, 0.)) 
print("dropout=0.5（50%概率丢弃）：\n", dropout_layer(X, 0.5)) 
print("dropout=1（全丢弃）：\n", dropout_layer(X, 1.))
# -------------------------- 3. 定义网络超参数 --------------------------
# 输入维度784（Fashion-MNIST图片为28×28，展平后784维）
# 输出维度10（对应10个服饰类别）
# 两个隐藏层的神经元数量均为256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
# 两个隐藏层后的丢弃概率：通常靠近输入的层丢弃概率更小
dropout1, dropout2 = 0.2, 0.5 
# -------------------------- 4. 自定义带Dropout的网络类 --------------------------
class Net(nn.Module):     
    # 初始化函数：传入维度参数 + 训练模式标记
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,                  
                 is_training=True):         
        super(Net, self).__init__()         # 调用父类nn.Module的构造函数
        self.num_inputs = num_inputs        # 保存输入维度，用于前向传播时展平图片
        self.training = is_training         # 标记当前是否为训练模式，控制Dropout是否生效        
        # 定义三层全连接层
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)   # 输入层 → 隐藏层1
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2) # 隐藏层1 → 隐藏层2
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)  # 隐藏层2 → 输出层
        self.relu = nn.ReLU()                             # ReLU激活函数
    # 前向传播逻辑
    def forward(self, X):         
        # 第一层：将图片展平为一维 → 全连接计算 → ReLU激活
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))                 
        # 关键：只有训练模式才启用Dropout，推理/测试时必须关闭
        if self.training == True:             
            H1 = dropout_layer(H1, dropout1)  # 第一个隐藏层后加Dropout     
        # 第二层：全连接计算 → ReLU激活
        H2 = self.relu(self.lin2(H1))         
        if self.training == True:             
            H2 = dropout_layer(H2, dropout2)  # 第二个隐藏层后加Dropout       
        # 输出层：直接输出分类logits（交叉熵损失内置了softmax）
        out = self.lin3(H2)         
        return out 
# 实例化自定义网络
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
# -------------------------- 5. 训练配置与手动实现版训练 --------------------------
num_epochs, lr, batch_size = 10, 0.5, 256  # 训练轮数、学习率、批次大小
# 交叉熵损失函数，reduction='none'表示返回每个样本的独立损失（不自动平均/求和）
loss = nn.CrossEntropyLoss(reduction='none') 
# 加载Fashion-MNIST训练集与测试集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 
# SGD随机梯度下降优化器，优化网络所有可学习参数
trainer = torch.optim.SGD(net.parameters(), lr=lr) 
# 调用d2l封装的训练函数，自动完成前向传播、反向传播、参数更新与测试
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# -------------------------- 6. PyTorch内置API简洁实现 --------------------------
# 使用nn.Sequential顺序堆叠层，代码更简洁
net = nn.Sequential(
    nn.Flatten(),         # 展平层：将 [batch, 28, 28] 转为 [batch, 784]
    nn.Linear(784, 256),  # 第一个全连接层
    nn.ReLU(),            # ReLU激活
    # PyTorch内置Dropout层：会自动根据网络.train()/.eval()切换状态
    # 训练时随机丢弃，推理时自动关闭，无需手动判断
    nn.Dropout(dropout1), 
    nn.Linear(256, 256),  # 第二个全连接层
    nn.ReLU(),            # ReLU激活
    nn.Dropout(dropout2), # 第二个Dropout层
    nn.Linear(256, 10)    # 输出分类层
) 
# 权重初始化函数
def init_weights(m):     
    # 仅对全连接层的权重做正态分布初始化，标准差0.01
    if type(m) == nn.Linear:         
        nn.init.normal_(m.weight, std=0.01) 
# 对网络所有层递归应用初始化函数
net.apply(init_weights);
# 重新定义优化器（对应新网络的参数）
trainer = torch.optim.SGD(net.parameters(), lr=lr) 
# 使用简洁实现的网络再次训练
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
