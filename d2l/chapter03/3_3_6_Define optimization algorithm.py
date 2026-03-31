trainer = torch.optim.SGD(net.parameters(), lr=0.03) 
# 1. 定义优化器（训练器）：使用PyTorch的SGD（随机梯度下降）优化器
# net.parameters()：传入要训练的模型（神经网络）的所有参数（权重、偏置）
# lr=0.03：学习率（步长），控制每次参数更新的幅度，数值越小更新越慢
