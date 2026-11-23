import random  
import torch  
from d2l import torch as d2l
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
  my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
  def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


import random  
import torch  
from d2l import torch as d2l
# 读取《时光机器》文本并分词，得到按行分词后的结果（每行是token列表）
tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
# 展平成一维列表corpus，corpus是全部token组成的语料序列
corpus = [token for line in tokens for token in line]
# 根据语料构建词表Vocab，自动统计每个token出现频次、分配索引
vocab = d2l.Vocab(corpus)
# 打印频次最高的前10个token
vocab.token_freqs[:10]
# 提取所有token的频次，用于绘制词频曲线（齐普夫定律验证）
freqs = [freq for token, freq in vocab.token_freqs]
# 绘制单gram词频图，双对数坐标
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
# 构造二元语法(bi-gram)：相邻两个token组成词对
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
# 基于二元词对构建词表，统计词对频次
bigram_vocab = d2l.Vocab(bigram_tokens)
# 打印频次最高的前10个二元词对
bigram_vocab.token_freqs[:10]
# 构造三元语法(tri-gram)：连续三个token组成三元组
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
# 基于三元组构建词表，统计三元组频次
trigram_vocab = d2l.Vocab(trigram_tokens)
# 打印频次最高的前10个三元词组
trigram_vocab.token_freqs[:10]
# 提取二元、三元语法的频次
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
# 在一张图中同时绘制单/双/三元语法的词频对比曲线，双对数坐标
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列
    参数：
        corpus: 一维token语料列表
        batch_size: 每个批次样本数量
        num_steps: 每个子序列的长度（时间步）
    返回：
        迭代器，每次yield (X,Y)，X是输入序列，Y是X右移一位的标签序列
    """
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    # 随机截断开头，避免每次划分起点固定
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签（标签是输入后移一位，最后一个token没有标签）
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引列表
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    # 打乱起始位置，实现随机采样
    random.shuffle(initial_indices)
    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]
    # 计算一共有多少个batch，丢弃不足一个batch的数据
    num_batches = num_subseqs // batch_size
    # 按batch_size分批取出起始索引
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        # X：每个起始位置取长度num_steps的序列
        X = [data(j) for j in initial_indices_per_batch]
        # Y：标签，是X每个序列向右移动一位（语言模型预测下一个token）
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 转成tensor返回
        yield torch.tensor(X), torch.tensor(Y)
# 测试随机采样迭代器：构造0~34共35个数字的测试序列
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列（顺序采样，适合RNN保持状态）
    特点：同一个batch内部样本在原始序列上连续，批次之间接续
    参数同上
    """
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    # 计算可用token总数，保证能被batch_size整除，舍弃末尾多余token
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    # Xs：输入序列tensor；Ys：标签序列，整体右移一位
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # reshape成 [batch_size, 总长度]：每一行是一条长序列
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    # 沿着时间维度切分成num_steps长度的片段
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        # 按时间步切片得到一批X,Y
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
# 测试顺序分区迭代器
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
class SeqDataLoader:  #@save
    """加载序列数据的迭代器封装类
    作用：统一管理语料、词表、采样方式，支持迭代
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        # 根据标志选择随机采样 / 顺序采样函数
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        # 加载时光机器数据集，返回处理好的语料和词表，限制最大token数量
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        # 返回迭代器实例
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表
    对外接口：
        batch_size：批次大小
        num_steps：时间步长度
        use_random_iter：True随机采样，False顺序采样
        max_tokens：语料最大token数
    返回：data_iter（迭代器）, vocab（词表）
    """
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
