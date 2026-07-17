%matplotlib inline
import torch
from d2l import torch as d2l

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)


# Jupyter Notebook 魔法命令：让 matplotlib 绘制的图表直接内嵌在单元格输出中，无需手动调用 plt.show()
%matplotlib inline
# 导入 PyTorch 深度学习核心库
import torch
# 导入《动手学深度学习》(d2l) 配套工具库的 PyTorch 版本，封装了绘图等常用教学工具
from d2l import torch as d2l
# ========== 第一部分：Sigmoid 激活函数及其梯度可视化 ==========
# 生成从 -8.0 到 8.0、步长为 0.1 的一维张量
# requires_grad=True 表示开启梯度追踪，后续可对该张量进行自动求导
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# 对 x 中每个元素计算 Sigmoid 激活函数值，得到输出张量 y
y = torch.sigmoid(x)
# 执行反向传播，计算 y 对 x 的梯度
# 由于 y 是向量而非标量，backward() 需要传入与 y 形状一致的权重张量
# torch.ones_like(x) 生成全1张量，表示上游梯度全部为1，等价于直接求 y 对 x 的导数
y.backward(torch.ones_like(x))
# 使用 d2l 封装的绘图函数绘制曲线
# 参数1：x 轴数据，需 detach() 解除梯度追踪后转为 numpy 数组
# 参数2：y 轴数据列表，分别为 sigmoid 函数曲线、sigmoid 的梯度曲线
# legend：设置图例标签；figsize：设置图表宽高尺寸（单位：英寸）
d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
# ========== 第二部分：矩阵连乘的数值爆炸演示 ==========
# 生成一个 4×4 的矩阵，元素服从均值为 0、标准差为 1 的标准正态分布
M = torch.normal(0, 1, size=(4,4))
# 打印初始随机矩阵
print('一个矩阵 \n',M)
# 循环 100 次，每次让当前矩阵 M 右乘一个新的 4×4 标准正态随机矩阵
# 用于模拟深度学习中多层网络的连乘效应，演示数值爆炸（梯度爆炸）现象
for i in range(100):
    # torch.mm 执行矩阵乘法，要求两个张量维度匹配（m×n 与 n×p）
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))
# 打印连续相乘 100 次后的矩阵，数值会急剧放大，体现数值爆炸
print('乘以100个矩阵后\n', M)
