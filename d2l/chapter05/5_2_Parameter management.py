import torch
from torch import nn
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
print(net[2].state_dict())
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
net[2].weight.grad == None
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
net.state_dict()['2.bias'].data
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)
rgnet[0][1][0].bias.data
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5
net.apply(my_init)
net[0].weight[:2]
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])


# 导入PyTorch核心库与神经网络模块
import torch
from torch import nn
# ====================== 1. 构建基础简单Sequential网络 ======================
# nn.Sequential：有序容器，按传入顺序堆叠层
# 网络结构：输入4维 -> 全连接(4→8) -> ReLU激活 -> 全连接(8→1)输出
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
# 构造测试输入X：2个样本，每个样本4个特征，取值0~1随机浮点数
X = torch.rand(size=(2, 4))
# 前向传播，将输入送入网络计算输出
net(X)
# ====================== 2. 访问网络层参数state_dict() ======================
# net[2]是第三个层：第二层全连接层Linear(8,1)
# state_dict()：返回该层所有参数的字典，key为参数名，value为参数张量
print(net[2].state_dict())
# 查看偏置bias的类型：nn.Parameter类型（可自动求导的可训练参数）
print(type(net[2].bias))
# 直接打印bias参数对象（包含值+是否需要梯度信息）
print(net[2].bias)
# .data 获取纯张量数值，剥离梯度计算相关信息
print(net[2].bias.data)
# 判断权重的梯度是否为None：网络未反向传播，梯度未计算，因此为True
print(net[2].weight.grad == None)
# named_parameters()：返回迭代器，输出(参数名,参数张量)
# 遍历第0层（第一层全连接Linear(4,8)）的所有参数，*解包打印
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# 遍历整个网络所有层的全部参数
print(*[(name, param.shape) for name, param in net.named_parameters()])
# 从整个网络的总参数字典中，取出第2层bias的数值张量
print(net.state_dict()['2.bias'].data)
# ====================== 3. 嵌套Sequential复杂网络（多层块嵌套） ======================
# 定义基础子块block1：4→8→ReLU→4→ReLU
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())
# 定义block2：循环4次，将4个block1嵌套进同一个Sequential容器
def block2():
    net = nn.Sequential()
    for i in range(4):
        # add_module(层名, 模块)：手动添加子模块，支持自定义层名，方便索引
        net.add_module(f'block {i}', block1())
    return net
# 组装深层网络：block2(4个嵌套子块) + 最后一层全连接输出1维
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# 测试输入前向传播
rgnet(X)
# 打印完整网络层级结构，直观看到嵌套关系
print(rgnet)
# 多层索引取参数：rgnet[0]=block2；[1]=第1个block1；[0]=block1第一层Linear，取其bias数值
print(rgnet[0][1][0].bias.data)
# ====================== 4. 网络参数初始化 apply() 全局遍历初始化 ======================
# 初始化函数1：正态分布初始化权重，偏置置0
def init_normal(m):
    # 判断当前模块是否为全连接层Linear
    if type(m) == nn.Linear:
        # normal_原地操作：权重均值0，标准差0.01正态分布填充
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # zeros_原地操作：偏置全部置0
        nn.init.zeros_(m.bias)
# net.apply(函数)：递归遍历网络所有子模块，每个模块都会传入init_normal执行初始化
net.apply(init_normal)
# 打印第一层权重第一行、第一层偏置第一个值查看初始化结果
print(net[0].weight.data[0], net[0].bias.data[0])
# 初始化函数2：所有权重固定为常数1，偏置0
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
# 全局重新初始化
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
# 初始化函数3：Xavier均匀初始化（适配激活函数，防止梯度消失爆炸）
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
# 初始化函数4：权重全部固定常数42
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
# 局部初始化：只对第0层（第一层全连接）执行Xavier初始化
net[0].apply(init_xavier)
# 局部初始化：只对第2层（输出全连接）权重设为42
net[2].apply(init_42)
# 分别打印两层权重验证局部初始化效果
print(net[0].weight.data[0])
print(net[2].weight.data)
# 自定义初始化：均匀分布采样+掩码过滤
def my_init(m):
    if type(m) == nn.Linear:
        # 打印当前层参数名称与形状
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        # uniform_：权重在[-10,10]均匀随机初始化
        nn.init.uniform_(m.weight, -10, 10)
        # 掩码：只保留绝对值≥5的权重，其余置0
        m.weight.data *= m.weight.data.abs() >= 5
# 全局执行自定义初始化
net.apply(my_init)
# 打印第一层前两行权重查看掩码效果
print(net[0].weight[:2])
# ====================== 5. 手动直接修改参数数值（跳过初始化API） ======================
# 所有权统一+1
net[0].weight.data[:] += 1
# 指定位置权重手动赋值42：第0行第0列
net[0].weight.data[0, 0] = 42
# 打印修改后的权重
print(net[0].weight.data[0])
# ====================== 6. 共享层/共享参数（同一模块多次复用，参数完全同步） ======================
# 定义可复用共享全连接层，必须单独创建变量才能重复使用
shared = nn.Linear(8, 8)
# 搭建网络，两次使用同一个shared模块，两层完全共享一套参数
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
# 前向传播
net(X)
# 判断两处共享层对应位置权重数值是否相等，输出全True
print(net[2].weight.data[0] == net[4].weight.data[0])
# 修改其中一处共享层权重
net[2].weight.data[0, 0] = 100
# 再次对比，另一处同步改变，证明二者共用同一份内存参数
print(net[2].weight.data[0] == net[4].weight.data[0])
