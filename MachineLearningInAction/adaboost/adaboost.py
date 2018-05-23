#! /usr/sbin/python3
# -*- coding:utf-8 -*-
import numpy as np


def load_simple_data():
    """添加一个简单数据集
    :return:
    """
    data_mat = np.matrix([[1., 2.1],
                          [2., 1.1],
                          [1.3, 1.],
                          [1., 1.],
                          [2., 1.]])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, class_labels


def stump_classify(data_matrix, dimen, thresh_val, thresh_ineq):
    """通过阈值分类
    :param data_matrix: 数据集
    :param dimen: 特征下标
    :param thresh_val: 阈值
    :param thresh_ineq: 符号
    :return:
    """
    ret_array = np.ones((np.shape(data_matrix)[0], 1))
    if thresh_ineq == 'lt':
        # 将小于阈值的置为-1类
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        # 将大于阈值的置为-1类
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, class_labels, d):
    """生成单层决策树
    :param data_arr: 数据集
    :param class_labels: 类别标签
    :param d: 权重
    :return:
    """
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = np.shape(data_mat)
    # 在特征的所有可能值上进行遍历
    num_steps = 10.0
    # 存储给定权重向量d时所得到的最佳单层决策树的相关信息
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    # 设为无穷大，用于寻找可能的最小错误率
    min_error = np.inf
    for i in range(n):
        # 在数据的所有特征上遍历
        # 通过特征的最大最小值来确定步长
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max-range_min)/num_steps
        for j in range(-1, int(num_steps)+1):
            for inequal in ['lt', 'gt']:
                thresh_val = (range_min + float(j) * step_size)
                # 计算预测的分类
                predicted_vals = stump_classify(data_mat, i, thresh_val, inequal)
                # 设置预测分类和真实类别不同的值为1
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                # 计算错误向量的权重和
                weight_error = d.T * err_arr
                if weight_error < min_error:
                    # 更新最小错误和最佳单层数
                    min_error = weight_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def adaboost_train_ds(data_arr, class_labels, num_it=40):
    """基于单层决策树训练AdaBoost
    :param data_arr: 数据集
    :param class_labels: 类别标签
    :param num_it: 迭代次数
    :return:
    """
    weak_class_arr = []
    m = np.shape(data_arr)[0]
    # 为每一条数据初始化权重
    d = np.mat(np.ones((m, 1))/m)
    # 记录每个数据点的类别估计累计值
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(num_it):
        # 建立单层决策树
        best_stump, error, class_est = build_stump(data_arr, class_labels, d)
        # print("D:", d.T)
        # 计算每一个单层决策树的权重
        # max(error,1e-16)保证在没有错误时不会除0异常
        alpha = float(0.5*np.log((1-error)/np.longfloat(max(error, 1e-16))))
        # 保存决策树权重和单层决策树
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)
        # print("class_est:", class_est.T)
        # 计算新的权重向量d
        expon = np.multiply(-1*alpha*np.mat(class_labels).T, class_est)
        d = np.multiply(d, np.exp(expon))
        d = d/d.sum()
        # 类别估计值
        agg_class_est += alpha*class_est
        # print('agg_class_est', agg_class_est.T)
        # 获取错误率
        agg_errors = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m, 1)))
        # print(agg_errors)
        error_rate = agg_errors.sum()/m
        # print('total error', error_rate, '\n')
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classify(dat_to_class, classifier_arr):
    """分类函数
    :param dat_to_class: 待分类的数据
    :param classifier_arr: 弱分类器数组
    :return:
    """
    data_matrix = np.mat(dat_to_class)
    m = np.shape(data_matrix)[0]
    # 记录每个数据点的类别估计累计值
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        # 计算类别估计值
        class_est = stump_classify(data_matrix,
                                   classifier_arr[i]['dim'],
                                   classifier_arr[i]['thresh'],
                                   classifier_arr[i]['ineq'])
        agg_class_est += classifier_arr[i]['alpha']*class_est
        print(agg_class_est)
    return np.sign(agg_class_est)


def load_data_set(file_name):
    """自适应数据加载函数
    :param file_name: 文件名
    :return:
    """
    # 计算数据列数
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat-1):
                # 读取特征
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            # 读取类别
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def plot_roc(pred_strengths, class_labels):
    """画出ROC曲线
    :param pred_strengths: 行向量组成的矩阵，表示分类器的预测强度
    :param class_labels: 类别标签
    :return:
    """
    import matplotlib.pyplot as plt
    # 保留绘制光标的位置
    cur = (1.0, 1.0)
    # 用于计算AUC的值
    y_sum = 0.0
    # 计算正例的数目
    num_pos_class = sum(np.array(class_labels) == 1.0)
    # 确定x轴和y轴上的步长
    y_step = 1/float(num_pos_class)
    x_step = 1/float(len(class_labels)-num_pos_class)
    # 得到排序索引
    # 因为索引时按照最小到最大的顺序排列
    # 所以从点<1.0,1.0>绘到<0,0>
    sorted_indicies = pred_strengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111)
    for index in sorted_indicies.tolist()[0]:
        # 在排序值上进行循环
        if class_labels[index] == 1.0:
            # 得到一个标签为1.0的类，在y轴上下降一个步长
            # 即不断降低真阳率
            del_x = 0
            del_y = y_step
        else:
            # 其他类别的标签，按x轴方向上倒退一个步长
            # 假阴率方向
            del_x = x_step
            del_y = 0
            # 对矩形的高度进行累加
            y_sum += cur[1]
        # 在当前点和新点之间画一条线
        ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], color='b')
        # 更新当前点的位置
        cur = (cur[0]-del_x, cur[1]-del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('假阳率')
    plt.ylabel('真阳率')
    plt.title('AdaBoost马疝病监测系统的ROC曲线')
    ax.axis([0, 1, 0, 1])
    plt.show()
    # 乘以x_step得到总面积
    print("曲线下面积为:", y_sum*x_step)

