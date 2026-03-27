num_epochs = 3  # 定义训练的总轮数（epoch），这里设置为3轮
for epoch in range(num_epochs):  # 外层循环：遍历每一轮训练
       for X, y in data_iter:
        # 内层循环：遍历数据加载器中的每一个批次（batch）数据
        # X：输入特征数据，y：对应的标签（真实结果）
            l = loss(net(X), y)  # 1. 前向传播：将数据X输入网络net，计算预测值并与真实值y对比，得到损失值loss
            trainer.zero_grad()  # 2. 梯度清零：优化器清空上一轮的梯度，防止梯度累加        
            l.backward()  # 3. 反向传播：根据损失值l，计算网络中每个参数的梯度（求导）
            trainer.step()  # 4. 更新参数：优化器根据计算出的梯度，更新网络权重
        l = loss(net(features), labels)  #一轮训练结束后，在整个数据集上计算最终损失（用于观察训练效果）
        print(f'epoch {epoch + 1}, loss {l:f}')  # 打印当前轮数和损失值，{l:f}表示以浮点数格式输出损失
    
