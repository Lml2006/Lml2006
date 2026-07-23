!nvidia-smi
import torch
from torch import nn
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
torch.cuda.device_count()
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
try_gpu(), try_gpu(10), try_all_gpus()
x = torch.tensor([1, 2, 3])
x.device
X = torch.ones(2, 3, device=try_gpu())
X
Y = torch.rand(2, 3, device=try_gpu(1))
Y
Z = X.cuda(1)
print(X)
print(Z)
Y + Z
Z.cuda(1) is Z
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
net(X)
net[0].weight.data.device


# !nvidia-smi 命令：查看当前服务器显卡信息（在Jupyter/Colab等notebook环境执行）
!nvidia-smi
# 导入PyTorch核心库、神经网络模块
import torch
from torch import nn
# 创建设备对象：cpu设备、默认cuda显卡、1号显卡
torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
# 查询本机可用GPU数量
torch.cuda.device_count()
def try_gpu(i=0):  #@save
    """
    如果存在，则返回gpu(i)，否则返回cpu()
    :param i: GPU编号，默认0号GPU
    :return: torch设备对象（cuda:i 或 cpu）
    """
    # 判断是否存在第i号GPU
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    # 没有对应GPU则使用CPU
    return torch.device('cpu')
def try_all_gpus():  #@save
    """
    返回所有可用的GPU，如果没有GPU，则返回[cpu(),]
    :return: 设备对象列表，包含全部可用GPU；无GPU时列表仅含CPU
    """
    # 遍历所有显卡编号，批量创建设备
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    # 有GPU返回GPU列表，无GPU返回CPU列表
    return devices if devices else [torch.device('cpu')]
# 测试设备函数：获取0号GPU、10号GPU(不存在会回落CPU)、全部可用设备
try_gpu(), try_gpu(10), try_all_gpus()
# 在CPU上创建张量
x = torch.tensor([1, 2, 3])
# 查看张量所在设备
x.device
# 在可用的0号GPU上创建2行3列全1张量
X = torch.ones(2, 3, device=try_gpu())
X
# 在1号GPU上创建2行3列随机张量；若无1号GPU则自动分配到CPU
Y = torch.rand(2, 3, device=try_gpu(1))
Y
# 将张量X复制迁移到1号GPU，生成新张量Z
Z = X.cuda(1)
print(X)  # X仍然在原0号GPU
print(Z)  # Z位于1号GPU
# 张量运算要求两个张量在同一设备上，Y和Z同在cuda:1，可以运算
Y + Z
# cuda(1)迁移到相同设备时，不会创建新对象，直接返回自身
Z.cuda(1) is 
# 搭建简单网络：单层全连接层，输入3维，输出1维
net = nn.Sequential(nn.Linear(3, 1))
# 将整个网络移动到目标设备（0号GPU，无GPU则CPU）
net = net.to(device=try_gpu())
# 将GPU上的张量X送入网络前向传播
net(X)
# 查看网络第一层权重参数存储在哪一个设备
net[0].weight.data.device
