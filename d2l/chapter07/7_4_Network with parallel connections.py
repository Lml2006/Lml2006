import tensorflow as tf
from d2l import tensorflow as d2l
class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')
    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
    def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
  def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
  def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
  def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])
# “net”必须是一个将被传递给“d2l.train_ch6（）”的函数。
# 为了利用我们现有的CPU/GPU设备，这样模型构建/编译需要在“strategy.scope()”
def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),
                                tf.keras.layers.Dense(10)])
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


import tensorflow as tf
from d2l import tensorflow as d2l
class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1：单1×1卷积层，用于直接提取特征，改变通道数
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')

        # 线路2：1×1卷积降维，再接3×3卷积提取中等尺度特征
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',
                                           activation='relu')

        # 线路3：1×1卷积降维，再接5×5卷积提取大尺度特征
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',
                                           activation='relu')

        # 线路4：3×3最大池化下采样，再用1×1卷积调整通道
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')
    def call(self, x):
        # 分支1前向传播
        p1 = self.p1_1(x)
        # 分支2：先1×1卷积，再3×3卷积
        p2 = self.p2_2(self.p2_1(x))
        # 分支3：先1×1卷积，再5×5卷积
        p3 = self.p3_2(self.p3_1(x))
        # 分支4：先池化，再1×1卷积
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度(axis=-1)拼接4条分支输出，得到Inception模块最终输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])
def b1():
    # GoogLeNet第一块：7×7大卷积 + 最大池化，初步提取底层特征，降分辨率
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
def b2():
    # 第二块：两个卷积进一步提取特征，再池化降维
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),    # 1×1降维
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
def b3():
    # 第三块：堆叠2个Inception模块，多尺度特征融合，最后池化
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
def b4():
    # 第四块：堆叠5个Inception模块，网络主体，大量多尺度特征提取
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
def b5():
    # 第五块：2个Inception，全局平均池化替代全连接降参，展平
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),   # 全局平均池化，每个通道输出一个值
        tf.keras.layers.Flatten()            # 展平为一维向量，送入输出层
    ])
# net包装为函数，适配d2l.train_ch6，方便在GPU策略scope内构建模型
def net():
    return tf.keras.Sequential([
        b1(), b2(), b3(), b4(), b5(),
        tf.keras.layers.Dense(10)   # 输出层，Fashion‑MNIST共10个类别
    ])
# 构造随机输入张量，模拟1张96×96单通道图片，遍历每一层打印输出shape，调试网络维度
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 超参数设置：学习率、训练轮数、批次大小
lr, num_epochs, batch_size = 0.1, 10, 128
# 加载Fashion‑MNIST数据集，图片resize到96×96
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 调用d2l训练函数，使用可用GPU训练GoogLeNet
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
