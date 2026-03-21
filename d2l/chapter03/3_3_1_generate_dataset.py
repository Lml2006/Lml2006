import numpy as np #导入数值计算库Numpy，简写为np
import torch #导入PyTorch深度学习框架
from torch.utils import data #从PyTorch工具库导入数据处理模块
from d2l import torch as d2l #导入《动手学深度学习》配套工具库，简写为d2l

true_w = torch.tensor([2, -3.4]) #定义真实权重w（2维特征的权重）
true_b = 4.2 #定义真实偏置顶b
features, labels = d2l.synthetic_data(true_w, true_b, 1000) #调用d2l的合成数据函数，生成1000个样本的线性回归数据集
