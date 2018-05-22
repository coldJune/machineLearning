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


class OptStruct:
    """
    建立一个数据结构保存所有重要的值
    """
    def __init__(self, data_mat_in, class_labels, c, toler, k_tup):
        self.X = data_mat_in
        self.label_mat = class_labels
        self.C = c
        self.tol = toler
        self.m = np.shape(data_mat_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], k_tup)


def calc_ek(opt_s, k):
    """计算误差值并返回
    :param opt_s: 数据对象
    :param k: alpha下标
    :return:
    """
    # forecast_x_k = float(np.multiply(opt_s.alphas, opt_s.label_mat).T *
    #                      (opt_s.X*opt_s.X[k, :].T) + opt_s.b)
    # error_k = forecast_x_k - float(opt_s.label_mat[k])
    forecast_x_k = float(np.multiply(opt_s.alphas, opt_s.label_mat).T*opt_s.K[:, k] + opt_s.b)
    error_k = forecast_x_k - float(opt_s.label_mat[k])
    return error_k


def select_j(i, opt_s, error_i):
    """选择第二个alpha值
    :param i: 第一个alpha值的下表
    :param opt_s: 数据对象
    :param error_i: 第一个alpha值的误差
    :return:
    """
    max_k = -1
    max_delta_e = 0
    error_j = 0
    # 输入值error_i在缓存中设置成为有效的
    opt_s.eCache[i] = [1, error_i]
    # 构建一个非零表，非零error值对应的alpha值
    valid_e_chache_list = np.nonzero(opt_s.eCache[:, 0].A)[0]
    if len(valid_e_chache_list) > 1:
        for k in valid_e_chache_list:
            # 在所有值上进行循环并选择其中使得改变最大的那个
            if k == i:
                continue
            error_k = calc_ek(opt_s, k)
            delta_e = abs(error_i-error_k)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                error_j = error_k
        return max_k, error_j
    else:
        j = select_j_rand(i, opt_s.m)
        error_j = calc_ek(opt_s, j)
        return j, error_j


def update_error_k(opt_s, k):
    """计算误差值并存入缓存
    :param opt_s: 数据对象
    :param k: 第k个alpha
    :return:
    """
    error_k = calc_ek(opt_s, k)
    opt_s.eCache[k] = [1, error_k]


def inner_l(i, opt_s):
    """内循环
    :param i: 第一个alpha下标
    :param opt_s: 数据对象
    :return:
    """
    error_i = calc_ek(opt_s, i)
    if (((opt_s.label_mat[i]*error_i < -opt_s.tol) and (opt_s.alphas[i] < opt_s.C)) or
            ((opt_s.label_mat[i]*error_i > opt_s.tol) and (opt_s.alphas[i] > 0))):
        j, error_j = select_j(i, opt_s, error_i)
        alpha_i_old = opt_s.alphas[i].copy()
        alpha_j_old = opt_s.alphas[j].copy()
        if opt_s.label_mat[i] != opt_s.label_mat[j]:
            l = max(0, opt_s.alphas[j] - opt_s.alphas[i])
            h = min(opt_s.C, opt_s.C + opt_s.alphas[j] - opt_s.alphas[i])
        else:
            l = max(0, opt_s.alphas[j]+opt_s.alphas[i]-opt_s.C)
            h = min(opt_s.C, opt_s.alphas[j] + opt_s.alphas[i])
        if l == h:
            print('L==H')
            return 0
        # eta = 2.0*opt_s.X[i, :]*opt_s.X[j, :].T -\
        #     opt_s.X[i, :]*opt_s.X[i, :].T -\
        #     opt_s.X[j, :]*opt_s.X[j, :].T
        eta = 2.0* opt_s.K[i, j] - opt_s.K[i, i] - opt_s.K[j, j] #核函数
        if eta >= 0:
            print('eta >=0')
            return 0
        opt_s.alphas[j] -= opt_s.label_mat[j] * (error_i-error_j)/eta
        opt_s.alphas[j] = clip_alpha(opt_s.alphas[j], h, l)
        update_error_k(opt_s, j)
        if abs(opt_s.alphas[j] - alpha_j_old) < 0.00001:
            print('j未移动足够的量')
            return 0
        opt_s.alphas[i] += opt_s.label_mat[j]*opt_s.label_mat[i]*(alpha_j_old-opt_s.alphas[j])
        update_error_k(opt_s, i)
        # b1 = opt_s.b-error_i-opt_s.label_mat[i] * \
        #     (opt_s.alphas[i]-alpha_i_old)*opt_s.X[i, :]*opt_s.X[i, :].T - \
        #     opt_s.label_mat[j]*(opt_s.alphas[j]-alpha_j_old)*opt_s.X[i, :]*opt_s.X[j, :].T
        # b2 = opt_s.b - error_j-opt_s.label_mat[i] *\
        #     (opt_s.alphas[i]-alpha_i_old)*opt_s.X[i, :]*opt_s.X[i, :].T - \
        #     opt_s.label_mat[j]*(opt_s.alphas[j]-alpha_j_old)*opt_s.X[j, :]*opt_s.X[j, :].T
        b1 = opt_s.b - error_i - opt_s.label_mat[i] * (opt_s.alphas[i] - alpha_i_old) * opt_s.K[i, i] - \
            opt_s.label_mat[j] * (opt_s.alphas[j] - alpha_j_old) * opt_s.K[i, j]
        b2 = opt_s.b - error_j - opt_s.label_mat[i] * (opt_s.alphas[i] - alpha_i_old) * opt_s.K[i, j] - \
            opt_s.label_mat[j] * (opt_s.alphas[j] - alpha_j_old) * opt_s.K[j, j]
        if (0 < opt_s.alphas[i]) and (opt_s.C > opt_s.alphas[i]):
            opt_s.b = b1
        elif (0 < opt_s.alphas[j]) and (opt_s.C > opt_s.alphas[j]):
            opt_s.b = b2
        else:
            opt_s.b = (b1+b2)/2.0
        return 1
    else:
        return 0


def smo_p(data_mat_in, class_labels, c, toler, max_iter, k_tup=('lin', 0)):
    """外循环
    :param data_mat_in: 数据集
    :param class_labels: 类别标签
    :param c: 常数C
    :param toler: 容错率
    :param max_iter: 取消前最大的循环次数
    :param k_tup:
    :return:
    """
    # 构建数据结构容纳所有数据
    opt_s = OptStruct(np.mat(data_mat_in), np.mat(class_labels).transpose(), c, toler, k_tup)
    ite = 0
    entire_set = True
    alpha_pairs_changed = 0
    while ite < max_iter and (alpha_pairs_changed > 0 or entire_set):
        # 当迭代次数超过指定的最大值或遍历整个集合都为对任意alpha对进行修改时退出
        alpha_pairs_changed = 0
        if entire_set:
            # 在数据集上遍历任意可能的alpha
            for i in range(opt_s.m):
                alpha_pairs_changed += inner_l(i, opt_s)
            print("所有值，iter:%d i:%d ,pairs changed %d" % (ite, i, alpha_pairs_changed))
            ite += 1
        else:
            # 遍历所有非边界alpha值
            non_bound_is = np.nonzero((opt_s.alphas.A > 0)*(opt_s.alphas.A < c))[0]
            for i in non_bound_is:
                alpha_pairs_changed += inner_l(i, opt_s)
                print("非边界，iter:%d i:%d,pairs changed:%d" % (ite, i, alpha_pairs_changed))
            ite += 1
        if entire_set:
            entire_set = False
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("迭代次数：%d" % ite)
    return opt_s.b, opt_s.alphas


def calc_ws(alphas, data_arr, class_labels):
    """计算w
    :param alphas: alpha集
    :param data_arr: 数据集
    :param class_labels: 标签集
    :return:
    """
    X = np.mat(data_arr)
    label_mat = np.mat(class_labels).transpose()
    m, n = np.shape(X)
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*label_mat[i], X[i, :].T)
    return w

"""核函数"""


def kernel_trans(x, a, k_tup):
    """
    :param x: 数据集
    :param a: 数据集的一行
    :param k_tup: 核函数信息
    :return:
    """
    m, n = np.shape(x)
    k = np.mat(np.zeros((m, 1)))
    if k_tup[0] == 'lin':
        # 线性核函数
        # 就算所有数据集和数据集中的一行的内积
        k = x * a.T
    elif k_tup[0] == 'rbf':
        # 径向基核函数
        # 对于矩阵中每个元素计算高斯函数的值
        for j in range(m):
            delta_row = x[j, :] - a
            k[j] = delta_row*delta_row.T
        # 将值应用到整个向量
        k = np.exp(k/(-1*k_tup[1]**2))
    else:
        raise NameError('不支持该核函数')
    return k


def test_rbf(k1=1.3):
    """测试核函数
    :param k1:
    :return:
    """
    data_arr, label_arr = load_data_set('data/testSetRBF.txt')
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    # 得到支持向量和alpha的类别标签值
    sv_ind = np.nonzero(alphas.A > 0)[0]
    s_vs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print("有%d个支持向量" % np.shape(s_vs)[0])
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        # 得到转换后的数据
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], ('rbf', k1))
        # 将数据与前面的alpha及类别标签值求积
        predict = kernel_eval.T*np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    print("训练错误率为：%f" % (float(error_count)/m))

    # 测试数据集
    data_arr, label_arr = load_data_set('data/testSetRBF2.txt')
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    print("测试错误率为：%f" % (float(error_count)/m))


"""基于SVM的手写数字识别"""


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


def load_images(dir_name):
    """加载图片为矩阵
    :param dir_name: 文件目录
    :return:
    """
    import os
    hw_labels = []
    training_file_list = os.listdir(dir_name)
    m = len(training_file_list)
    training_mat = np.zeros((m, 1024))
    for i in range(m):
        file_name_set = training_file_list[i]
        file_str = file_name_set.split('.')[0]
        class_num_str = int(file_str.split('_')[0])
        if class_num_str == 9:
            # 为9时标签置为-1
            hw_labels.append(-1)
        else:
            # 否则置为1
            hw_labels.append(1)
        training_mat[i, :] = img2vector('%s/%s' % (dir_name, file_name_set))
    return training_mat, hw_labels


def test_digits(k_tup=('rbf', 10)):
    """测试核函数
    :param k_tup:
    :return:
    """
    data_arr, label_arr = load_images('data/digits/trainingDigits')
    b, alphas = smo_p(data_arr, label_arr, 200, 0.0001, 10000, k_tup)
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    # 得到支持向量和alpha的类别标签值
    sv_ind = np.nonzero(alphas.A > 0)[0]
    s_vs = data_mat[sv_ind]
    label_sv = label_mat[sv_ind]
    print("有%d个支持向量" % np.shape(s_vs)[0])
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        # 得到转换后的数据
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        # 将数据与前面的alpha及类别标签值求积
        predict = kernel_eval.T*np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    print("训练错误率为：%f" % (float(error_count)/m))

    # 测试数据集
    data_arr, label_arr = load_images('data/digits/testDigits')
    data_mat = np.mat(data_arr)
    label_mat = np.mat(label_arr).transpose()
    m, n = np.shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(s_vs, data_mat[i, :], k_tup)
        predict = kernel_eval.T * np.multiply(label_sv, alphas[sv_ind]) + b
        if np.sign(predict) != np.sign(label_mat[i]):
            error_count += 1
    print("测试错误率为：%f" % (float(error_count)/m))
