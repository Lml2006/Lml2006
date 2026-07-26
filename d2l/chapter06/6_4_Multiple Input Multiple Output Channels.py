import tensorflow as tf
from d2l import tensorflow as d2l
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
corr2d_multi_in(X, K)
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return tf.stack([corr2d_multi_in(X, k) for k in K], 0)
K = tf.stack((K, K + 1, K + 2), 0)
K.shape
corr2d_multi_in_out(X, K)
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X, (c_i, h * w))
    K = tf.reshape(K, (c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))
X = tf.random.normal((3, 3, 3), 0, 1)
K = tf.random.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6


import tensorflow as tf
from d2l import tensorflow as d2l
def corr2d_multi_in(X, K):
    """
    多输入通道的二维互相关运算（卷积核心操作，无偏置）
    参数：
        X: 输入张量，形状 (输入通道数 c_i, 高度 h, 宽度 w)
        K: 卷积核张量，形状 (输入通道数 c_i, 核高 kh, 核宽 kw)
    返回：
        单通道输出特征图，形状 (h-kh+1, w-kw+1)
    原理：每个输入通道分别和对应通道卷积核做互相关，再逐元素求和融合通道信息
    """
    # zip(X,K) 按通道配对；每个通道独立执行互相关；所有通道结果相加
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
# 构造测试输入：2个输入通道，3×3特征图
X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
# 构造卷积核：2个输入通道，2×2卷积核
K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
# 执行多输入通道卷积运算
corr2d_multi_in(X, K)
def corr2d_multi_in_out(X, K):
    """
    多输入通道 + 多输出通道 二维互相关运算
    参数：
        X: 输入特征图 (c_i, h, w)
        K: 卷积核组 (输出通道数 c_o, 输入通道数 c_i, kh, kw)
    返回：
        输出特征图 (c_o, h_out, w_out)
    原理：一组卷积核对应一个输出通道，循环每组核调用多输入通道卷积，最后堆叠所有输出通道
    """
    # 遍历输出通道维度，每组卷积核生成一张输出特征图
    # tf.stack 在第0维堆叠，组合成多通道输出
    return tf.stack([corr2d_multi_in(X, k) for k in K], 0)
# 扩充卷积核：原有K、K+1、K+2 三组卷积核，得到3个输出通道
K = tf.stack((K, K + 1, K + 2), 0)
# 打印卷积核形状：(输出通道,输入通道,核高,核宽)
K.shape
# 执行多输入多输出通道卷积
corr2d_multi_in_out(X, K)
def corr2d_multi_in_out_1x1(X, K):
    """
    1×1卷积的手动实现，等价矩阵乘法形式
    1×1卷积：不改变空间尺寸，只用来跨通道特征融合
    参数：
        X: 输入特征图 (c_i, h, w)
        K: 1×1卷积核 (c_o, c_i, 1, 1)
    返回：
        输出特征图 (c_o, h, w)
    核心思想：把空间维度展平，1×1卷积退化为通道之间的全连接矩阵乘法
    """
    c_i, h, w = X.shape   # 输入通道、特征图高、宽
    c_o = K.shape[0]      # 输出通道数量
    # 将特征图空间维度展平：(c_i, h*w)，每个空间位置视为一个样本
    X = tf.reshape(X, (c_i, h * w))
    # 卷积核去除1×1空间维度：(c_o, c_i)，等价权重矩阵
    K = tf.reshape(K, (c_o, c_i))
    # 矩阵乘法：完成不同通道之间特征加权融合
    Y = tf.matmul(K, X)
    # 恢复空间维度，还原成特征图格式 (c_o, h, w)
    return tf.reshape(Y, (c_o, h, w))
# 随机生成测试数据：3输入通道，3×3特征图
X = tf.random.normal((3, 3, 3), 0, 1)
# 2个输出通道，3输入通道，1×1卷积核
K = tf.random.normal((2, 3, 1, 1), 0, 1)
# 方式1：矩阵乘法实现1×1卷积
Y1 = corr2d_multi_in_out_1x1(X, K)
# 方式2：通用多输入多输出卷积实现1×1卷积
Y2 = corr2d_multi_in_out(X, K)
# 断言验证两种实现结果完全等价，误差小于极小值
assert float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6
