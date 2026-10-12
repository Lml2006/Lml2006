import tensorflow as tf
from d2l import tensorflow as d2l
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # 计算移动方差元平方根的倒数
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 缩放和移位
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # 非模型参数的变量初始化为0和1
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)
    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)
    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
    # 回想一下，这个函数必须传递给d2l.train_ch6。
# 或者说为了利用我们现有的CPU/GPU设备，需要在strategy.scope()建立模型
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# 导入TensorFlow框架
import tensorflow as tf
# 导入d2l工具库（动手学深度学习配套工具，封装数据加载、训练绘图等）
from d2l import tensorflow as d2l
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    """
    批量归一化前向计算函数（推理阶段使用）
    :param X: 输入特征
    :param gamma: 拉伸参数γ，可学习
    :param beta: 偏移参数β，可学习
    :param moving_mean: 全局滑动均值（训练时更新，推理时固定）
    :param moving_var: 全局滑动方差（训练时更新，推理时固定）
    :param eps: 极小值，防止除0
    :return: 批量归一化后的输出Y
    """
    # rsqrt: 1/sqrt(x)，计算 1 / sqrt(滑动方差 + eps)
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # 缩放：inv * gamma 等价 gamma / sqrt(var + eps)
    inv *= gamma
    # 完整BN公式：Y = gamma * (X - mean)/sqrt(var+eps) + beta
    Y = X * inv + (beta - moving_mean * inv)
    return Y
class BatchNorm(tf.keras.layers.Layer):
    """自定义批量归一化层，继承Keras基础层"""
    def __init__(self, **kwargs):
        # 调用父类Layer的构造函数
        super(BatchNorm, self).__init__(**kwargs)
    def build(self, input_shape):
        """build：在第一次拿到输入形状后创建权重，仅执行一次"""
        # 参数形状：通道维度，BN对每个通道单独一组gamma/beta
        weight_shape = [input_shape[-1], ]
        # 可训练参数gamma（拉伸系数），初始化为1
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        # 可训练参数beta（偏移系数），初始化为0
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # 滑动均值moving_mean，非可训练，训练时滑动更新，推理使用，初始0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        # 滑动方差moving_variance，非可训练，初始化为1
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        # 调用父类build方法，标记层构建完成
        super(BatchNorm, self).build(input_shape)
    def assign_moving_average(self, variable, value):
        """滑动平均更新：momentum动量更新全局均值/方差"""
        momentum = 0.9  # 动量系数，一般取0.9/0.99
        # 滑动平均公式：new_val = old * momentum + new_batch_val * (1-momentum)
        delta = variable * momentum + value * (1 - momentum)
        # 将计算结果赋值给variable，返回更新操作
        return variable.assign(delta)
    @tf.function  # 编译为计算图，加速执行
    def call(self, inputs, training):
        """前向传播核心逻辑，training标记训练/推理模式"""
        if training:
            # axes：除最后一维通道外，其余维度求均值方差（N,H,W维度）
            axes = list(range(len(inputs.shape) - 1))
            # 在batch、高、宽维度求均值，keepdims保留维度方便广播
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            # 求批量方差：E[(X-μ)^2]，stop_gradient防止均值参与方差求导
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            # 挤压多余维度，形状变为[C]通道维度
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            # 使用滑动平均更新全局moving_mean
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            # 使用滑动平均更新全局moving_variance
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            # 注册更新操作，训练时自动执行
            self.add_update(mean_update)
            self.add_update(variance_update)
            # 训练阶段使用当前batch的均值方差做归一化
            mean, variance = batch_mean, batch_variance
        else:
            # 推理阶段：使用训练得到的滑动均值、方差，不用当前batch统计量
            mean, variance = self.moving_mean, self.moving_variance
        # 调用批量归一化函数，得到输出
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
# 构建LeNet网络，使用上面【自定义BatchNorm层】
def net():
    return tf.keras.models.Sequential([
        # 卷积层：6个5×5卷积核，输入28×28单通道FashionMNIST图像
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),                  # 自定义BN层
        tf.keras.layers.Activation('sigmoid'), # sigmoid激活
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2), # 2×2平均池化
        tf.keras.layers.Conv2D(filters=16, kernel_size=5), # 第二层卷积
        BatchNorm(),                  # 自定义BN层
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),    # 展平，把二维特征转为一维向量
        tf.keras.layers.Dense(120),   # 全连接层 120神经元
        BatchNorm(),                  # 自定义BN层
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),    # 全连接层 84神经元
        BatchNorm(),                  # 自定义BN层
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]    # 输出层，10分类（FashionMNIST）
    )
# 超参数设置：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 1.0, 10, 256
# 加载Fashion-MNIST数据集，生成训练集、测试集迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# 训练网络，d2l.train_ch6：训练+评估+绘图，自动选择GPU/CPU
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# 取出第一个BN层的gamma和beta参数，reshape方便查看
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
# 重新定义net，使用TensorFlow【内置BatchNormalization层】做对比实验
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(), # TF官方BN层
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
# 使用内置BN层重新训练，对比自定义BN效果
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
