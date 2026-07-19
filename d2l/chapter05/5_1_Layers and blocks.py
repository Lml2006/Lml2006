import tensorflow as tf
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
X = tf.random.uniform((2, 20))
net(X)
class MLP(tf.keras.Model):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Model的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        # Hiddenlayer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Outputlayer
    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def call(self, X):
        return self.out(self.hidden((X)))
net = MLP()
net(X)
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # 这里，block是tf.keras.layers.Layer子类的一个实例
            self.modules.append(block)
    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
      net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # 使用tf.constant函数创建的随机权重参数在训练期间不会更新（即为常量参数）
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)
    def call(self, inputs):
        X = self.flatten(inputs)
        # 使用创建的常量参数以及relu和matmul函数
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.dense(X)
        # 控制流
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
net = FixedHiddenMLP()
net(X)
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)
    def call(self, inputs):
        return self.dense(self.net(inputs))
chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)


# 导入TensorFlow深度学习框架，约定别名为tf
import tensorflow as tf
# ===================== 方式1：使用内置Sequential序贯容器构建模型 =====================
# Sequential 是Keras提供的线性堆叠模型容器，数据会按顺序依次流经每一层
net = tf.keras.models.Sequential([
    # 全连接层（Dense）：隐藏层，输出维度256，激活函数使用ReLU
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # 全连接层：输出层，输出维度10（对应10分类任务的输出），默认无激活函数
    tf.keras.layers.Dense(10),
])
# 构造测试输入：形状为(2, 20)的均匀随机张量
# 2代表batch_size（批次样本数），20代表每个样本的特征维度
X = tf.random.uniform((2, 20))
# 调用模型执行前向传播，传入输入X，得到模型输出
net(X)
# ===================== 方式2：继承Model基类自定义MLP模型 =====================
# 自定义多层感知机类，继承 tf.keras.Model 是Keras自定义模型的标准方式
# 相比Sequential，自定义类可以实现更灵活的前向逻辑、控制流、参数复用
class MLP(tf.keras.Model):
    # 构造函数：实例化模型时调用，用于定义模型包含的层和可训练参数
    def __init__(self):
        # 调用父类Model的构造函数，完成模型基类的内部初始化
        # 父类会统一管理模型的所有权重、层、训练状态等
        super().__init__()
        # 定义隐藏层：全连接层，输出单元数256，激活函数为ReLU
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        # 定义输出层：全连接层，输出单元数10
        self.out = tf.keras.layers.Dense(units=10)  
    # 定义前向传播逻辑：模型被调用时自动执行此方法
    # 输入X先经过隐藏层计算，再传入输出层，最终返回输出结果
    def call(self, X):
        return self.out(self.hidden((X)))
# 实例化自定义MLP模型
net = MLP()
# 执行前向传播，验证模型输出
net(X)
# ===================== 方式3：自定义MySequential，模拟官方Sequential功能 =====================
# 手动实现一个序列容器，理解Sequential的底层原理：按顺序依次调用各层
class MySequential(tf.keras.Model):
    # *args 接收任意数量的层/模块作为入参
    def __init__(self, *args):
        super().__init__()
        # 用列表存储所有传入的网络层/模块
        self.modules = []
        # 遍历所有传入的模块，依次存入列表
        for block in args:
            # block 是 tf.keras.layers.Layer 或 Model 的子类实例
            self.modules.append(block)
    # 前向传播：按列表顺序依次调用每个模块，数据逐层向后传递
    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
# 使用自定义序列容器构建模型，传入两个全连接层
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
# 执行前向传播验证
net(X)
# ===================== 方式4：带固定权重、参数复用与控制流的特殊模型 =====================
# 演示自定义模型的灵活能力：常量权重、层复用、循环控制流
# 这些复杂逻辑无法通过Sequential直接实现
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 展平层：将多维输入展平为二维（batch_size, 特征总数）
        self.flatten = tf.keras.layers.Flatten()
        # 创建形状为(20, 20)的常量权重矩阵
        # tf.constant 创建的是常量张量，训练过程中不会被梯度更新
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        # 定义一个可训练的全连接层，输出维度20，ReLU激活
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)
    def call(self, inputs):
        # 第一步：将输入展平为二维张量
        X = self.flatten(inputs)
        # 第二步：使用常量权重做矩阵乘法 + 偏置1，再经过ReLU激活
        # 这部分权重固定不变，不参与模型训练
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 第三步：传入全连接层计算
        # 该层可被多次调用，实现参数复用（共享同一组权重）
        X = self.dense(X)
        # 第四步：自定义循环控制流
        # 当X所有元素的绝对值之和大于1时，将X整体除以2
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        # 返回X所有元素的和（最终输出为标量）
        return tf.reduce_sum(X)
# 实例化并验证前向传播
net = FixedHiddenMLP()
net(X)
# ===================== 方式5：模型嵌套与混合组合 =====================
# 自定义嵌套模型：内部包含一个Sequential子网络，再外接一个全连接层
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 内部定义一个序贯子网络
        self.net = tf.keras.Sequential()
        # 向子网络添加第一个全连接层，输出64维，ReLU激活
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        # 向子网络添加第二个全连接层，输出32维，ReLU激活
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        # 定义外部全连接层，输出16维，ReLU激活
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)
    # 前向传播：输入先经过子网络，再经过外部全连接层
    def call(self, inputs):
        return self.dense(self.net(inputs))
# 构建混合组合模型：将自定义模型、普通层、特殊模型堆叠到一个Sequential中
chimera = tf.keras.Sequential()
# 第一层：加入自定义嵌套模型NestMLP
chimera.add(NestMLP())
# 第二层：普通全连接层，输出20维
chimera.add(tf.keras.layers.Dense(20))
# 第三层：加入带固定权重和控制流的FixedHiddenMLP
chimera.add(FixedHiddenMLP())
# 执行前向传播，验证整个嵌套组合模型
chimera(X)
