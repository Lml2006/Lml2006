import torch
from torch import nn
from torch.nn import functional as F
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')
x2
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))
net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
Y_clone = clone(X)
Y_clone == Y


# 导入PyTorch核心库
import torch
# 导入神经网络模块
from torch import nn
# 导入神经网络常用函数（激活函数等）
from torch.nn import functional as F
# ========== 1. 保存与加载单个张量 ==========
# 创建一维张量 [0,1,2,3]
x = torch.arange(4)
# 将张量x保存到本地文件 x-file
torch.save(x, 'x-file')
# 从文件加载张量到x2
x2 = torch.load('x-file')
# 打印加载得到的张量
x2
# ========== 2. 保存与加载张量列表 ==========
# 创建全0张量
y = torch.zeros(4)
# 保存由多个张量组成的列表
torch.save([x, y],'x-files')
# 加载列表并解包赋值
x2, y2 = torch.load('x-files')
# 输出两个张量
(x2, y2)
# ========== 3. 保存与加载字典（最常用，方便管理多个变量） ==========
# 构造字典，存储多个张量
mydict = {'x': x, 'y': y}
# 保存字典对象
torch.save(mydict, 'mydict')
# 加载字典
mydict2 = torch.load('mydict')
# 打印加载后的字典
mydict2
# ========== 4. 保存和加载神经网络模型参数（重点！深度学习模型保存标准方式） ==========
# 定义多层感知机MLP网络
class MLP(nn.Module):
    def __init__(self):
        super().__init__()  # 继承父类构造函数，必须调用
        self.hidden = nn.Linear(20, 256)   # 隐藏层：输入20维，输出256维
        self.output = nn.Linear(256, 10)   # 输出层：输入256维，输出10维

    # 前向传播逻辑，定义网络计算流程
    def forward(self, x):
        # x经过隐藏层→ReLU激活→输出层，返回预测结果
        return self.output(F.relu(self.hidden(x)))
# 实例化神经网络
net = MLP()
# 构造输入样本：2个样本，每个样本20维特征
X = torch.randn(size=(2, 20))
# 网络前向传播，得到原始模型输出
Y = net(X)
# 只保存网络参数（state_dict：模型权重、偏置等参数字典，不保存网络结构！推荐方式）
torch.save(net.state_dict(), 'mlp.params')
# 先新建一个结构完全相同的网络
clone = MLP()
# 从文件加载参数，赋值给新网络
clone.load_state_dict(torch.load('mlp.params'))
# 设置模型为评估模式（关闭Dropout、BatchNorm训练特有行为）
clone.eval()
# 使用加载参数后的网络进行预测
Y_clone = clone(X)
# 判断两套网络输出结果是否全部相等，验证参数加载成功
Y_clone == Y
