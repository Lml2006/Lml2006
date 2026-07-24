import tensorflow as tf
from d2l import tensorflow as d2l
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
  X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)
    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
      X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
K = tf.constant([[1.0, -1.0]])
Y = corr2d(X, K)
Y
corr2d(tf.transpose(X), K)
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)
# 这个二维卷积层使用四维输入和输出格式（批量大小、高度、宽度、通道），
# 其中批量大小和通道数都为1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # 学习率
Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # 迭代卷积核
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {tf.reduce_sum(l):.3f}')
          tf.reshape(conv2d.get_weights()[0], (1, 2))


import tensorflow as tf
from d2l import tensorflow as d2l
def corr2d(X, K):  #@save
    """
    计算二维互相关运算（卷积运算核心操作，无填充、步幅为1）
    参数：
        X: 输入二维张量，特征矩阵
        K: 卷积核（核矩阵）
    返回：
        Y: 互相关运算输出矩阵
    """
    # 获取卷积核的高度h、宽度w
    h, w = K.shape
    # 计算输出特征图尺寸：(H-h+1, W-w+1)，创建可变张量用于保存结果
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    # 遍历输出矩阵每一行
    for i in range(Y.shape[0]):
        # 遍历输出矩阵每一列
        for j in range(Y.shape[1]):
            # 取出X对应窗口，与卷积核逐元素相乘后求和，赋值给Y[i,j]
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
# 构造输入特征矩阵X
X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 构造2×2卷积核K
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
# 执行二维互相关运算
corr2d(X, K)
# 自定义二维卷积层类，基于tf.keras.layers.Layer实现
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def build(self, kernel_size):
        """
        初始化层参数
        kernel_size: 卷积核尺寸 (kh, kw)
        """
        # 正态分布初始化器
        initializer = tf.random_normal_initializer()
        # 创建卷积核权重参数
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        # 创建偏置参数，单通道偏置shape=(1,)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)
    def call(self, inputs):
        """前向传播：输入执行互相关运算 + 偏置"""
        return corr2d(inputs, self.weight) + self.bias
# 构造6行8列输入矩阵，初始全部为1
X = tf.Variable(tf.ones((6, 8)))
# 将第2~5列置0，构造带有垂直边缘的图像
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
# 定义1×2边缘检测卷积核：检测水平相邻像素差值
K = tf.constant([[1.0, -1.0]])
# 互相关运算得到边缘检测结果Y
Y = corr2d(X, K)
Y
# 将输入矩阵转置后再做互相关，验证该卷积核只能检测水平边缘，无法检测垂直边缘
corr2d(tf.transpose(X), K)
# ========== 使用内置卷积层，通过训练学习卷积核（边缘检测核） ==========
# 构建内置二维卷积层：输出通道1，卷积核(1,2)，不使用偏置
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)
# TensorFlow卷积层输入格式：(批量大小, 高度, 宽度, 通道) NHWC
# 转换维度：增加batch维度与通道维度
X = tf.reshape(X, (1, 6, 8, 1))
# 标签Y同样扩充为四维格式
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # 设置学习率
Y_hat = conv2d(X)
# 迭代训练，自动拟合卷积核
for i in range(10):
    # 梯度带，手动监控卷积核权重
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        # 前向传播得到预测输出
        Y_hat = conv2d(X)
        # 损失函数：预测值与真实标签绝对值平方
        l = (abs(Y_hat - Y)) ** 2
    # 计算梯度，手动实现梯度下降更新
    grad = g.gradient(l, conv2d.weights[0])
    update = tf.multiply(lr, grad)
    weights = conv2d.get_weights()
    weights[0] = conv2d.weights[0] - update
    conv2d.set_weights(weights)
    # 每两轮打印损失
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {tf.reduce_sum(l):.3f}')
# 将训练完成的四维卷积核reshape为(1,2)，观察是否逼近 [[1,-1]]
tf.reshape(conv2d.get_weights()[0], (1, 2))
