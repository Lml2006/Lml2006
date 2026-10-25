%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))
batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net
# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')
net = get_net()
train(net, train_iter, loss, 5, 0.01)
onestep_preds = net(features)
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)
  steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))


# %matplotlib inline 是jupyter魔法命令，开启单元格内绘图，不用单独弹出图片窗口
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l  # 导入d2l工具库，提供绘图、数据加载等辅助函数

T = 1000  # 总共产生1000个时间步的时序数据点
time = torch.arange(1, T + 1, dtype=torch.float32)  # 生成时间轴：1,2,...,1000
# 构造原始时序：正弦信号 + 高斯噪声，模拟带噪声的时间序列
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# 绘制原始时序曲线，x轴time，y轴x，横坐标范围1~1000，画布大小6×3
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

tau = 4  # 时间窗口大小：用前面tau=4个时刻的值，预测下一个时刻的值
# 构造特征矩阵：行数=样本数量(T-tau)，列数=时间窗口tau；每行是一个长度为4的历史序列
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 依次填充每一列：第i列为x[i : i + (T-tau)]，滑动窗口采样
    features[:, i] = x[i: T - tau + i]
# 构造标签：每个样本对应的下一个时刻真值，reshape成(N,1)的二维张量
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train=600个样本用于训练集；加载数据集，生成迭代器，is_train=True做打乱
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# 初始化网络权重的函数：对Linear层使用Xavier均匀初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 搭建一个简单的多层感知机MLP：输入4维，隐藏层10个神经元，ReLU激活，输出1维预测值
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)  # 对网络所有层应用权重初始化
    return net

# 平方损失MSE，reduction='none'：不自动求和/求均值，保留每个样本单独的损失值
loss = nn.MSELoss(reduction='none')

# 训练函数：输入网络、数据迭代器、损失函数、迭代轮数、学习率
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)  # 使用Adam优化器
    for epoch in range(epochs):
        for X, y in train_iter:  # 遍历每一个batch
            trainer.zero_grad()  # 梯度清零，防止累积
            l = loss(net(X), y)  # 前向传播，计算每个样本损失
            l.sum().backward()    # 损失求和，反向传播计算梯度
            trainer.step()        # 根据梯度更新网络参数
        # 每一轮结束，在训练集上评估损失并打印
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()  # 实例化网络
train(net, train_iter, loss, 5, 0.01)  # 训练5轮，学习率0.01

# 单步预测：已知真实历史观测值，预测下一个时刻（one-step prediction）
onestep_preds = net(features)
# 绘图对比：原始数据 & 单步预测结果
d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))

# 多步预测（迭代预测）：用前面预测出来的值，继续作为输入往后预测
multistep_preds = torch.zeros(T)
# 训练集以及前面tau个真实观测直接填入，这部分不用预测
multistep_preds[: n_train + tau] = x[: n_train + tau]
# 从训练集结束位置开始迭代预测：每次取最近tau个值（含预测值）送入网络
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))
# 绘图对比：原始数据、单步预测、多步迭代预测
d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))

max_steps = 64  # 最大预测步数，用来生成1~64步不同步长预测结果
# 构造特征矩阵：列分为两部分，前tau列是真实观测，后面max_steps列是逐步预测结果
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# 列i（i<tau）是来自x的观测，滑动截取原始时序作为初始窗口
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]
# 列i（i>=tau）：利用前面窗口的预测结果，迭代生成更长步长的预测
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)  # 对比 1步、4步、16步、64步预测效果
# 绘制不同步长的多步预测曲线，可以观察到步数越大误差累积越严重
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
