import collections
import re 
from d2l import torch as d2l
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):  # 未知词元的索引为0
        return 0
    @property
    def token_freqs(self):
        return self._token_freqs
def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
  vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
  def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)


# -*- coding: utf-8 -*-
"""
《动手学深度学习》(d2l) 第 8 章：文本预处理 —— 时光机器 (The Time Machine) 数据集
本文件的处理流水线：
    原始文本文件
        -> read_time_machine()：读取 + 清洗（只保留字母、转小写）
        -> tokenize()：切分成词元（单词级或字符级）
        -> Vocab：建立词表，把词元映射成从 0 开始的整数索引
        -> load_corpus_time_machine()：得到最终用于训练的"索引序列 + 词表"
"""
# collections 提供 Counter，用于高效统计词元出现频率
import collections
# re 提供正则表达式，用于清洗文本（去掉标点、数字等非字母字符）
import re
# d2l 中的 torch 版本模块，提供数据集下载、路径管理等工具函数
from d2l import torch as d2l
#@save
# 向 d2l 的全局数据集字典 DATA_HUB 注册数据集：
#   key   = 数据集名称 'time_machine'
#   value = (下载地址, SHA-1 校验值)
# 注册后，d2l.download('time_machine') 就能自动下载、缓存并校验该文件；
# 其中 DATA_URL 是 d2l 默认的数据集镜像地址前缀。
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    # d2l.download('time_machine') 返回本地缓存文件路径（首次调用时会自动下载）
    # 以只读文本模式打开，f.readlines() 把文件按行读成字符串列表
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    # 逐行做文本清洗：
    #   re.sub('[^A-Za-z]+', ' ', line) —— 把所有"连续的非字母字符"替换成一个空格
    #                                      （即去掉标点、数字、换行符等）
    #   .strip()  —— 去掉首尾多余的空格
    #   .lower()  —— 统一转成小写，避免 'The' 和 'the' 被当成两个不同的词
    # 返回：清洗后的文本行列表，每个元素是一行字符串
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
# ---- 演示：读取并查看前几行 ----
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')  # 打印数据集总行数
print(lines[0])    # 打印第 1 行，确认清洗效果
print(lines[10])   # 打印第 11 行
def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        # 单词级（word-level）分词：按空白字符切分，一行 -> 若干单词
        # 例如 'i love you' -> ['i', 'love', 'you']
        return [line.split() for line in lines]
    elif token == 'char':
        # 字符级（char-level）分词：把每个字符单独作为一个词元
        # 例如 'abc' -> ['a', 'b', 'c']
        return [list(line) for line in lines]
    else:
        # 传入未知的切分粒度时给出错误提示（注意：这里只打印，未抛异常）
        print('错误：未知词元类型：' + token)
# ---- 演示：按单词切分，并打印前 11 行的词元 ----
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])  # 每行输出一个单词列表
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # tokens：二维词元列表（如 [['i','love'], ['you']]）或一维列表；
        #         为 None 时表示构造空词表
        # min_freq：最小词频阈值，出现次数低于该值的词元会被丢弃（视为未知词元）
        # reserved_tokens：需要预留的专用词元（如 <pad>、<bos>、<eos>），
        #                  它们不参与统计，但会被固定分配到靠前的索引
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        # count_corpus 返回 Counter（词元 -> 频次）；
        # .items() 得到 (词元, 频次) 对，按频次从高到低排序，
        # 结果保存在 self._token_freqs 中，形如 [('the', 100), ('a', 80), ...]
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        # 词表第 0 位固定是 <unk>（unknown，未知词元）；
        # 其后再接上调用方指定的预留词元
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 反向映射：词元 -> 索引，方便 O(1) 查询；由 idx_to_token 构造而来
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 按频次从高到低依次把词元加入词表（高频词因此获得较小的索引）
        for token, freq in self._token_freqs:
            if freq < min_freq:
                # 因为已按频次降序排列，一旦发现频次低于阈值，后面的只会更低，
                # 所以可以直接 break 提前结束
                break
            if token not in self.token_to_idx:
                # 去重：跳过已经存在的词元（例如预留词元本身也出现在语料中时）
                self.idx_to_token.append(token)
                # 新词元的索引 = 添加前列表长度 - 1，即追加后的末位下标
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    def __len__(self):
        # 支持 len(vocab)，返回词表大小（即不同词元的个数）
        return len(self.idx_to_token)
    def __getitem__(self, tokens):
        # 支持 vocab[token]（单个词元）和 vocab[token_list]（词元列表）两种用法
        if not isinstance(tokens, (list, tuple)):
            # 单个词元：查字典返回索引；查不到时返回 unk（即 0），
            # 这就是"未登录词/未知词"统一映射到 0 的原因
            return self.token_to_idx.get(tokens, self.unk)
        # 列表/元组：逐个转换，返回索引列表
        return [self.__getitem__(token) for token in tokens]
    def to_tokens(self, indices):
        # __getitem__ 的逆操作：把索引还原成词元
        if not isinstance(indices, (list, tuple)):
            # 单个索引 -> 单个词元
            return self.idx_to_token[indices]
        # 索引列表 -> 词元列表
        return [self.idx_to_token[index] for index in indices]
    @property
    def unk(self):  # 未知词元的索引为0
        # 用 property 装饰，可以像属性一样访问 vocab.unk 而不用加括号；
        # 约定 <unk> 永远位于索引 0
        return 0
    @property
    def token_freqs(self):
        # 以只读属性形式暴露按频次降序排列的 (词元, 频次) 列表
        return self._token_freqs
def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        # 情况一：tokens 为空，无需展平
        # 情况二：tokens 是二维列表（每行一个词元列表），需要展平成一维
        #         [['a','b'], ['c']] -> ['a','b','c']
        tokens = [token for line in tokens for token in line]
    # 用 Counter 统计每个词元出现的次数，返回类似 {'the': 100, 'a': 80} 的字典
    return collections.Counter(tokens)
# ---- 演示：基于单词级词元构建词表 ----
vocab = Vocab(tokens)
# 打印词表 (词元 -> 索引) 映射的前 10 项，验证 <unk> 位于索引 0
print(list(vocab.token_to_idx.items())[:10])
# 对第 1 行和第 11 行，展示"词元列表 <-> 索引列表"的双向转换
for i in [0, 10]:
    print('文本:', tokens[i])   # 原始词元
    print('索引:', vocab[tokens[i]])  # 对应的整数索引
def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    # 1) 读取并清洗文本
    lines = read_time_machine()
    # 2) 这里采用"字符级"分词（不按单词切），以减少词表规模、降低训练难度
    tokens = tokenize(lines, 'char')
    # 3) 用字符级词元构建词表
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # 即把二维的 [每行字符列表] 展平成一维的索引序列，供语言模型按顺序建模
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        # 只保留前 max_tokens 个词元，便于快速实验
        corpus = corpus[:max_tokens]
    # 返回：索引序列（供模型训练）与词表（供索引/词元互转）
    return corpus, vocab
# ---- 演示：加载完整的时光机器语料 ----
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)  # 输出语料中的词元总数 与 词表大小
