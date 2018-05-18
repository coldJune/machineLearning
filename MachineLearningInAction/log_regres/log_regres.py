#! usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np


def load_data_set():
    """加载测试数据集
    打开文本逐行读取，设置x_0为1，每行前两个值为x_1,x_2
    第三个值对应类别标签
    :return:
    """
    data_mat = []
    label_mat = []
    with open('data/testSet.txt', 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split()
            data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
            label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    """定义阶跃函数
    :param in_x: 线性假设函数theta'*X(weight'*X)
    :return:
    """
    return np.longfloat(1.0/(1+np.exp(-in_x)))


def grad_ascent(data_mat_in, class_labels):
    """梯度上升函数
    :param data_mat_in: 数据集m*n的矩阵，m行表示m个训练样本，n列表式n个特征值(x_0为1)
    :param class_labels: 类别标签,为1*m的行向量，m对应m个训练样本
    :return:
    """
    # 将数据装换为NumPy矩阵
    # 将类别标签行向量转换为列向量
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_matrix)
    # 初始化梯度上升算法的一些值
    # alpha为移动步长
    # max_cycles为迭代次数
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1))
    for k in range(max_cycles):
        # 计算梯度上升算法
        h = sigmoid(data_matrix*weights)
        error = label_mat - h
        weights = weights + alpha*data_matrix.transpose()*error
    return weights


def plot_best_fit(weight):
    """ 画出分界线
    :param weight: 数据训练出来的参数值
    :return:
    """
    import matplotlib.pyplot as plt
    # 将矩阵转换为数组
    weights = weight.getA()
    data_mat, label_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(n):
        # 区分数据类别
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 使sigmoid函数值为0.5即weight'*X=0
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    """ 随机梯度上升算法
    :param data_matrix: 训练数据集
    :param class_labels:  训练数据对应的分类标签
    :return:
    """
    m, n = np.shape(data_matrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        # 每一次只对一个样本运行梯度上升算法
        h = sigmoid(data_matrix[i]*weights)
        error = class_labels[i] - h
        weights = weights + alpha*error*data_matrix[i]
    return weights


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    """改进后的随机梯度上升算法
    :param data_matrix: 训练数据集
    :param class_labels: 训练数据对应的分类标签
    :param num_iter: 迭代次数
    :return:
    """
    m, n = np.shape(data_matrix)
    # 将数据转换为矩阵
    weights = np.ones((n, 1))
    data_matrix = np.mat(data_matrix)
    class_labels = np.mat(class_labels).transpose()
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            # 每次迭代中调整步进值
            alpha = 4/(1.0+j+i)+0.01
            # 随机选取样本更新回归系数
            # 然后从列表中删除对应的值
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(data_matrix[rand_index]*weights)
            error = class_labels[rand_index] - h
            weights = weights + alpha*data_matrix[rand_index].transpose()*error
            del(data_index[rand_index])
    return weights

"""从疝气病症预测病马的死亡率
1. 收集数据：给定数据文件
2. 准备数据：用Python解析文本文件并填充缺失值
3. 分析数据：可视化并观察数据
4. 训练算法：使用优化算法，找到最佳系数
5. 测试算法：观察错误率，通过改变迭代的次数和步长等菜蔬来得到更好的回归系数
6. 使用算法：
"""


def classify_vector(in_x, weights):
    """分类函数
    计算对应的Sigmoid值来对数据进行分类
    :param in_x: 数据集
    :param weights: 回归系数
    :return:
    """
    prob = sigmoid(in_x*weights)
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    """训练函数
    :return:
    """
    train_set = []
    train_labels = []
    with open('data/horseColicTraining.txt', 'r', encoding='utf-8') as fr_train:
        # 训练训练集，训练回归系数
        for line in fr_train.readlines():
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            train_set.append(line_arr)
            train_labels.append(float(curr_line[21]))
    train_weight = stoc_grad_ascent1(train_set, train_labels, 500)
    error_count = 0.0
    num_test_vec = 0.0
    with open('data/horseColicTest.txt', 'r', encoding='utf-8') as fr_test:
        # 使用测试集测试
        for line in fr_test.readlines():
            num_test_vec += 1.0
            curr_line = line.strip().split('\t')
            line_arr = []
            for i in range(21):
                line_arr.append(float(curr_line[i]))
            if int(classify_vector(np.array(line_arr), train_weight)) != int(curr_line[21]):
                error_count += 1
    error_rate = (float(error_count)/num_test_vec)
    print('错误率为：%f' % error_rate)
    return error_rate


def multi_test():
    """多次测试函数
    :return:
    """
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print('迭代%d次后平均误差为%f' % (num_tests, error_sum/float(num_tests)))