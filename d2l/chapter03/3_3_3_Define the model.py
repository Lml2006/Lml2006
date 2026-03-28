# nn是神经网络的缩写 
from torch import nn 
 net = nn.Sequential(nn.Linear(2, 1)) #创建模型，输入两个特征，一个结果
