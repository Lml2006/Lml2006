#激活函数
%matplotlib inline 
import torch 
from d2l import torch as d2l
#ReLU函数
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True) 
y = torch.relu(x) 
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
y.backward(torch.ones_like(x), retain_graph=True) 
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
#sigmoid函数
y = torch.sigmoid(x) 
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# 清除以前的梯度 
x.grad.data.zero_() 
y.backward(torch.ones_like(x),retain_graph=True) 
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
#tanh函数
y = torch.tanh(x) 
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# 清除以前的梯度 
x.grad.data.zero_() 
y.backward(torch.ones_like(x),retain_graph=True) 
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

# 设置内联绘图（Jupyter Notebook 中显示图像）
%matplotlib inline

import torch
from d2l import torch as d2l  # 动手学深度学习工具库，提供绘图等辅助功能

# ==================== 1. ReLU 激活函数 ====================
# 生成从 -8 到 8 的连续值，步长 0.1，并开启自动求导（requires_grad=True）
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)

# 计算 ReLU 函数值：y = max(0, x)
y = torch.relu(x)

# 绘制 ReLU 函数曲线（x 需 detach() 断开计算图，仅取数值）
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))

# 对 y 关于 x 求导（使用全 1 的梯度权重，因为 y 是标量？这里实际是向量对向量求导，需 retain_graph 保留计算图以便后续复用）
y.backward(torch.ones_like(x), retain_graph=True)

# 绘制 ReLU 的导数（梯度）曲线：x<0 时导数为 0，x>0 时导数为 1
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

# ==================== 2. Sigmoid 激活函数 ====================
# 计算 Sigmoid 函数值：y = 1 / (1 + exp(-x))
y = torch.sigmoid(x)

# 绘制 Sigmoid 曲线（S 形，值域 (0,1)）
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 清除之前累积的梯度（ReLU 的梯度），避免影响本次求导
x.grad.data.zero_()

# 再次求导，计算 Sigmoid 的梯度（dy/dx = y*(1-y)）
y.backward(torch.ones_like(x), retain_graph=True)

# 绘制 Sigmoid 导数曲线（在 0 附近最大，两侧趋近于 0）
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

# ==================== 3. tanh 激活函数 ====================
# 计算 tanh 函数值：y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))，值域 (-1,1)
y = torch.tanh(x)

# 绘制 tanh 曲线（类似 Sigmoid 但中心对称，过原点）
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# 清除之前 Sigmoid 的梯度
x.grad.data.zero_()

# 求 tanh 的梯度（dy/dx = 1 - y^2）
y.backward(torch.ones_like(x), retain_graph=True)

# 绘制 tanh 导数曲线（在 0 处最大为 1，两侧趋近于 0）
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
