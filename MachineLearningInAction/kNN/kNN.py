#! usr/sbin/Python3
# -*- coding:utf-8 -*-

import numpy as np
import operator
import os
"""
1. 计算已知类别数据集中的点与当前点之间的距离
2. 按照距离递增次序排序
3. 选取与当前距离最小的k个点
4. 确定前k个点所在类别的出现频率
5. 返回前k个点出现频率最高的类别作为当前点的预测分类
"""


def classify(in_x, data_set, labels, k):
    """k-近邻算法
    :param in_x: 分类的输入向量
    :param data_set: 输入的训练样本集
    :param labels: 标签向量，元素数目和矩阵dataSet的行数相同
    :param k: 用于选择最近邻居的数目
    :return:
    """
    data_set_size = data_set.shape[0]
    # 计算输入向量与样本的差值
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    # 计算欧式距离
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    # 排序
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        # 获取第i个元素的label
        vote_i_label = labels[sorted_dist_indicies[i]]
        # 计算该类别的数目
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    # 对类别按值进行排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


"""在约会网站上使用k-近邻算法
1. 收集数据：提供文本文件
2. 准备数据：使用Python解析文本文件
3. 分析数据：使用Matplotlib画二维扩散图
4. 训练算法：不适用与k-近邻算法
5. 测试算法：使用提供的部分数据作为测试样本
    测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误
6. 使用算法：产生简单的命令行程序，输入特征数据判断
"""


def file2matrix(filename):
    """将文本记录转换为NumPy
    :param filename: 文件名
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as file:
        # 读取文本计算样本数量
        array_o_lines = file.readlines()
        number_of_lines = len(array_o_lines)
        # 生成样本举证
        return_mat = np.zeros((number_of_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_o_lines:
            # 处理每一个样本
            line = line.strip()
            list_from_line = line.split('\t')
            # 获取数据
            return_mat[index, :] = list_from_line[0:3]
            # 获取标签
            class_label_vector.append(int(list_from_line[-1]))
            index += 1
        return return_mat, class_label_vector


def auto_norm(data_set):
    """归一化特征值
    :param data_set: 数据集
    :return:
    """
    # 计算最小值和最大值及两者的差值
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    m = data_set.shape[0]
    # 归一化数据集
    norm_data_set = data_set - np.tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def dating_class_test():
    """分类器针对约会网站的测试代码
    :return:
    """
    ho_radtio = 0.10
    dating_data_mat, dating_labels = file2matrix('dataSet/datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_radtio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                     dating_labels[num_test_vecs:m], 3)
        print('预测值为：%d,真实值为：%d' % (classifier_result, dating_labels[i]))
        if(classifier_result != dating_labels[i]):
            error_count += 1.0
    print("错误率为：%f" % (error_count/float(num_test_vecs)))


"""手写识别系统
1. 收集数据：提供文本文件
2. 准备数据：编写函数img2vector()，将图像格式转换为分类器使用的向量格式
3. 分析数据：在Python命令提示符中检查数据，确保它符合要求
4. 训练算法：不适用k-近邻算法
5. 测试算法：使用提供的部分数据作为测试样本
    测试样本和非测试样本的区别在于：测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误
6. 使用算法：产生简单的命令行程序，输入特征数据判断
"""


def img2vector(filename):
    """将图像转换为测试向量
    将测试数据中32*32的二进制图像矩阵转换为1*1024的向量
    :param filename: 文件名
    :return:
    """
    return_vect = np.zeros((1, 1024))
    with open(filename, 'r', encoding='utf-8') as file:
        for i in range(32):
            line_str = file.readline()
            for j in range(32):
                return_vect[0, 32*i+j] = int(line_str[j])
    return return_vect


def handwriting_class_test():
    """使用k-近邻算法识别手写数字
    :return:
    """
    hw_labels = []
    training_file_list = os.listdir('dataSet/digits/trainingDigits')
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        # 加载数据集并添加标签
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        training_mat[i, :] = img2vector('dataSet/digits/trainingDigits/%s' % file_name_str)

    test_file_list = os.listdir('dataSet/digits/testDigits')
    error_account = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        # 预测训练集
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        hw_labels.append(class_num_str)
        vector_under_test = img2vector('dataSet/digits/testDigits/%s' % file_name_str)
        classifier_result = classify(vector_under_test, training_mat, hw_labels, 3)
        print('预测值为：%d,真实值为：%d' % (classifier_result, hw_labels[i]))
        if classifier_result != class_num_str:
            error_account += 1.0
    print("预测错误个数为:%d" % error_account)
    print("错误率为：%f" % (error_account/float(m_test)))
