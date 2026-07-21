import torch
import torch.nn.functional as F
from torch import nn
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X):
        return X - X.mean()
      layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
linear = MyLinear(5, 3)
linear.weight
linear(torch.rand(2, 5))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))


# 导入PyTorch核心库
import torch
# 导入神经网络函数库，包含激活函数等
import torch.nn.functional as F
# 导入神经网络模块基类
from torch import nn
# 自定义层：中心化层，将输入数据减去均值，使输出均值趋近于0
class CenteredLayer(nn.Module):
    def __init__(self):
        # 调用父类nn.Module构造函数，必须书写
        super().__init__()
    def forward(self, X):
        # 前向传播逻辑：输入减去自身均值，实现中心化
        return X - X.mean()
# 实例化自定义中心化层
layer = CenteredLayer()
# 创建浮点张量输入，送入自定义层测试
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
# 构建串行网络：线性层 + 自定义中心化层
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# 生成4行8列随机输入，送入网络前向传播
Y = net(torch.rand(4, 8))
# 打印网络输出均值，理论上接近0（浮点微小误差）
Y.mean()
# 自定义全连接线性层，手动创建权重与偏置参数
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # 定义权重参数：in_units输入维度，units输出维度，注册为可训练参数
        self.weight = nn.Parameter(torch.randn(in_units, units))
        # 定义偏置参数，维度与输出units一致，注册为可训练参数
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        # 矩阵乘法 X @ weight + bias，使用.data获取张量数据
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        # 使用ReLU激活函数返回结果
        return F.relu(linear)
# 实例化自定义线性层：输入5维，输出3维
linear = MyLinear(5, 3)
# 查看自定义层中的权重参数
linear.weight
# 传入2行5列随机张量进行前向计算
linear(torch.rand(2, 5))
# 搭建多层网络，堆叠两层自定义线性层
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
# 输入2个样本，每个样本64维，通过完整网络计算
net(torch.rand(2, 64))
