#! usr/sbin/python3
# -*- coding:utf-8-*-

import numpy as np
import re


def load_data_set():
    """创造实验样本
    :return:
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', '', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit',  'buying',  'worthless',  ' dog ', 'food',  'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    """创建一个包含在所有文档中出现的不重复词的列表
    :param data_set: 数据集
    :return:
    """
    vocab_set = set([])
    for document in data_set:
        # 将每篇文档返回的新词集合添加到该集合中
        # 操作符|用于求两个集合的并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words_2_vec(vocab_list, input_set):
    """将单词转换为向量(词集模型)
    :param vocab_list: 词汇表
    :param input_set: 某个文档
    :return:
    """
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        # 遍历文档
        # 如果单词在词汇表中，将词汇表向量中对应位置置为1
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print("该单词不在我的词汇表中:%s" % word)
    return return_vec


def train_nb_0(train_matrix, train_category):
    """朴素贝叶斯分类器
    :param train_matrix: 文档矩阵,由set_of_words_2_vec()转换来
    :param train_category: 每篇文档类别标签构成的向量
    :return:
    """
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    # 计算文档归属为(class=1)的概率
    pa_busive = sum(train_category)/float(num_train_docs)
    # 初始化分子变量和分母变量
    p0_num, p1_num = np.ones(num_words), np.ones(num_words)
    pO_denom, p1_denom = 2.0, 2.0
    for i in range(num_train_docs):
        # 遍历所有文档
        # 某个词在文档中出现该词对应的个数加1，该文档的总次数也加1
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            pO_denom += sum(train_matrix[i])
    # 对每个元素除以该类别中的总词数
    p1_vect = np.log(p1_num/p1_denom)
    p0_vect = np.log(p0_num/pO_denom)
    return p0_vect, p1_vect, pa_busive


def classify_nb(vec_2_classify, p0_vec, p1_vec, p_class1):
    """朴素贝叶斯分类函数
    :param vec_2_classify: 需要分类的向量
    :param p0_vec: 分类为0的概率向量
    :param p1_vec: 分类为1的概率向量
    :param p_class1: 文档归属为(class=1)的概率
    :return:
    """
    # 对应元素相乘
    # 将词汇表中所有词的对应值相加
    # 加上类别的对数概率
    p1 = sum(vec_2_classify*p1_vec)+np.log(p_class1)
    p0 = sum(vec_2_classify*p0_vec)+np.log(1.0-p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    """测试朴素贝叶斯分类器
    :return:
    """
    list_o_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_o_posts)
    train_mat = []
    for list_o_post in list_o_posts:
        train_mat.append(set_of_words_2_vec(my_vocab_list, list_o_post))
    p0_v, p1_v, pa_b = train_nb_0(train_mat, list_classes)
    test_entry = ['love', 'my', 'dalmation']
    this_doc = np.array(set_of_words_2_vec(my_vocab_list, test_entry))
    print('test_entry 分类为：', classify_nb(this_doc, p0_v, p1_v, pa_b))
    test_entry = ['stupid', 'garbage']
    this_doc = np.array(set_of_words_2_vec(my_vocab_list, test_entry))
    print('test_entry 分类为：', classify_nb(this_doc, p0_v, p1_v, pa_b))


def bag_of_words_2_vec_mn(vocab_list, input_set):
    """将单词转换为向量(词袋模型)
    :param vocab_list: 词汇表
    :param input_set: 某个文档
    :return:
    """
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            # 没遇到一个单词增加词向量中的对应值
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def text_parse(big_string):
    """文件解析
    将大字符串解析为字符串列表
    :param big_string: 大字符串
    :return:
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower()for tok in list_of_tokens if len(tok) > 2]


def spam_test():
    """垃圾邮件分类测试
    :return:
    """
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        # 导入文件夹spam和ham下的文本文件并将它们解析为词列表
        word_list = text_parse(open('data/email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(doc_list)
        class_list.append(1)
        word_list = text_parse(open('data/email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(doc_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    training_set = list(range(50))
    test_set = []
    for i in range(10):
        # 随机选择10个文件作为测试集并将其从训练集中剔除
        rand_index = int(np.random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_class = []
    for doc_index in training_set:
        # 训练所有文档
        train_mat.append(set_of_words_2_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb_0(train_mat, train_class)
    error_count = 0
    for doc_index in test_set:
        # 验证文档
        word_vec = set_of_words_2_vec(vocab_list, doc_list[doc_index])
        if classify_nb(word_vec, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('错误率为：', float(error_count)/len(test_set))


def calc_most_freq(vocab_list, full_text):
    """去除高频词汇
    :param vocab_list: 词汇表
    :param full_text: 文本
    :return:
    """
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    """RSS源分类测试
    :param feed1: RSS源1
    :param feed0: RSS源0
    :return:
    """
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)
    top30words = calc_most_freq(vocab_list, full_text)
    for pair_w in top30words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    traning_set = list(range(2*min_len))
    test_set = []
    for i in range(20):
        rand_index = int(np.random.uniform(0, len(traning_set)))
        test_set.append(traning_set[rand_index])
        del(traning_set[rand_index])
    traning_mat = []
    traning_class = []
    for doc_index in traning_set:
        traning_mat.append(bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index]))
        traning_class.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb_0(traning_mat, traning_class)
    error_count = 0
    for doc_index in test_set:
        word_vec = bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index])
        if classify_nb(word_vec, p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('错误率为：', float(error_count)/len(test_set))
    return vocab_list, p0_v, p1_v

