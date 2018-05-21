#! /usr/sbin/python3
# -*- coding:utf-8 -*-


import numpy as np


def load_data_set(file_name):
    """加载数据
    :param file_name: 文件名
    :return:
    """
    data_mat = []
    label_mat = []
    with open(file_name, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line_arr = line.strip().split('\t')
            data_mat.append([float(line_arr[0]), float(line_arr[1])])
            label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    """随机选择一个整数
    :param i: alpha的下标
    :param m: 所有alpha的数目
    :return:
    """
    j = i
    while j == i:
        # 只要函数值不等于输入值i,就进行随机选择
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    """调整数值
    :param aj: alpha值
    :param h: 上限
    :param l: 下限
    :return:
    """
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    """简化版SMO算法
    :param data_mat_in: 数据集
    :param class_labels: 类别标签
    :param c: 常数C
    :param toler: 容错率
    :param max_iter: 取消前最大的循环次数
    :return:
    """
    data_matrix = np.mat(data_mat_in)
    label_mat = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_matrix)
    alphas = np.mat(np.zeros((m, 1)))
    # 在没有任何alpha改变的情况下遍历数据集的次数
    ite = 0
    while ite < max_iter:
        # 记录alpha是否进行优化
        alpha_pairs_changed = 0
        for i in range(m):
            # 计算预测的类别
            forecast_x_i = float(np.multiply(alphas, label_mat).T *
                                 (data_matrix*data_matrix[i, :].T)) + b
            # 计算预测类别和真实类别的误差
            error_i = forecast_x_i - float(label_mat[i])
            if (((label_mat[i]*error_i < -toler) and (alphas[i] < c)) or
                    ((label_mat[i]*error_i > toler) and (alphas[i] > 0))):
                # 如果误差超出范围(不等于0或C,正负间隔值)
                # 选取第二个alpha值
                j = select_j_rand(i, m)
                # 计算第二个alpha预测的类别
                forecast_x_j = float(np.multiply(alphas, label_mat).T *
                                     (data_matrix*data_matrix[j, :].T)) + b
                # 计算第二个alpha预测的类别和真实类别的误差
                error_j = forecast_x_j - float(label_mat[j])
                # 保存第一个和第二个alpha
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    # 计算l和h
                    # 用于将alpha[j]调整到0到c之间
                    l = max(0, alphas[j] - alphas[i])
                    h = max(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                if l == h:
                    # 如果两者相等则不做任何改变进入下一次循环
                    print('L==H')
                    continue
                # 计算alpha[j]的最优修改量
                eta = 2.0 * data_matrix[i, :]*data_matrix[j, :].T \
                    - data_matrix[i, :]*data_matrix[i, :].T \
                    - data_matrix[j, :]*data_matrix[j, :].T
                if eta >= 0:
                    print('eta>0')
                    continue
                # 计算新的alphas[j]
                alphas[j] -= label_mat[j]*(error_i - error_j)/eta
                alphas[j] = clip_alpha(alphas[j], h, l)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    # 检查alphas[j]是否有轻微改变，有就进入下一次循环
                    print('j未移动足够的量')
                    continue
                # 对i进行修改，修改量与j相同，单方向相反
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old-alphas[j])
                # 设置常数项
                b1 = b-error_i-label_mat[i]*(alphas[i]-alpha_i_old) * \
                    data_matrix[i, :]*data_matrix[i, :].T - \
                    label_mat[j]*(alphas[j]-alpha_j_old) * \
                    data_matrix[i, :]*data_matrix[j, :].T

                b2 = b-error_j-label_mat[i]*(alphas[i]-alpha_i_old) * \
                    data_matrix[i, :]*data_matrix[j, :].T - \
                    label_mat[j]*(alphas[j]-alpha_j_old) * \
                    data_matrix[j, :]*data_matrix[j, :].T
                if (alphas[i] > 0) and (alphas[i] < c):
                    b = b1
                elif (alphas[j] > 0) and (alphas[j] < c):
                    b = b2
                else:
                    b = (b1+b2)/2.0
                # 记录改变的对数
                alpha_pairs_changed += 1
                print("iter: %d i:%d,pair changed:%d" % (ite, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            # 如果未更新则迭代次数加1
            ite += 1
        else:
            # 否则迭代次数置为0
            ite = 0
        print("迭代次数:%d" % ite)
    return b, alphas
