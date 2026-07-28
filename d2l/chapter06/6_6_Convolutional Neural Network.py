import tensorflow as tf
from d2l import tensorflow as d2l
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """一个以可视化的训练进展的回调"""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')
#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# 导入TensorFlow主库
import tensorflow as tf
# 导入d2l学习工具库（配套《动手学深度学习》代码）
from d2l import tensorflow as d2l
def net():
    """构建LeNet卷积神经网络模型"""
    return tf.keras.models.Sequential([
        # 第一层卷积：6个5×5卷积核，sigmoid激活，padding=same保持输入输出尺寸一致
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        # 第一层平均池化：窗口2×2，步幅2，下采样
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        # 第二层卷积：16个5×5卷积核，无padding，sigmoid激活
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        # 第二层平均池化：窗口2×2，步幅2
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        # 展平层：将二维特征图转为一维向量，接入全连接层
        tf.keras.layers.Flatten(),
        # 第一个全连接层，120个神经元，sigmoid激活
        tf.keras.layers.Dense(120, activation='sigmoid'),
        # 第二个全连接层，84个神经元，sigmoid激活
        tf.keras.layers.Dense(84, activation='sigmoid'),
        # 输出层：10个神经元，输出原始logits，不添加softmax
        tf.keras.layers.Dense(10)])
# 创建测试输入张量：batch=1，高度28，宽度28，单通道灰度图
X = tf.random.uniform((1, 28, 28, 1))
# 逐层前向传播，打印每一层输出形状，用于调试网络维度
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
# 设置批次大小
batch_size = 256
# 加载Fashion-MNIST数据集，生成训练迭代器、测试迭代器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
class TrainCallback(tf.keras.callbacks.Callback):  #@save
    """自定义Keras回调类，实现训练过程可视化、计时与指标打印"""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()          # 计时器，统计训练速度
        # 绘图动画器：绘制损失、训练精度、测试精度曲线
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net                     # 网络模型
        self.train_iter = train_iter       # 训练集迭代器
        self.test_iter = test_iter         # 测试集迭代器
        self.num_epochs = num_epochs       # 总训练轮数
        self.device_name = device_name     # 训练设备名称
    def on_epoch_begin(self, epoch, logs=None):
        """每一轮epoch开始时触发，启动计时器"""
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        """每一轮epoch结束时触发"""
        self.timer.stop()
        # 在测试集评估模型，得到测试准确率
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        # 组装指标：训练损失、训练精度、测试精度
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        # 更新绘图曲线
        self.animator.add(epoch + 1, metrics)
        # 训练全部轮次结束后打印最终结果与训练速度
        if epoch == self.num_epochs - 1:
            # 获取一个批次样本数量
            batch_size = next(iter(self.train_iter))[0].shape[0]
            # 计算训练集样本总数
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            # 输出每秒处理样本数量以及使用设备
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')
#@save
def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """
    第六章训练封装函数：使用指定设备(GPU/CPU)训练卷积网络
    net_fn: 构建网络的函数
    train_iter: 训练数据集迭代器
    test_iter: 测试数据集迭代器
    num_epochs: 训练轮数
    lr: 学习率
    device: 训练硬件设备
    """
    device_name = device._device_name
    # 单设备分布式策略，指定在选定GPU/CPU上执行模型构建与训练
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        # 随机梯度下降优化器
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        # 损失函数：稀疏分类交叉熵，输入为logits（内部自带softmax）
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()                     # 实例化网络
        # 模型编译：绑定优化器、损失函数、评估指标
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    # 创建自定义训练回调实例
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    # 开始训练，关闭内置日志输出，使用自定义回调
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net
# 超参数：学习率0.9，训练10轮
lr, num_epochs = 0.9, 10
# 获取可用GPU，启动训练
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
