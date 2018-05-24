#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np


def load_data_set(file_name):
    """加载数据
    :param file_name: 文件名
    :return:
    """
    num_feat = len(open(file_name).readline().split('\t')) -1
    data_mat = []
    label_mat = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_arr = []
            cur_line = line.strip().split('\t')
            for i in range(num_feat):
                line_arr.append(float(cur_line[i]))
            data_mat.append(line_arr)
            label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


def stand_regress(x_arr, y_arr):
    """计算最佳拟合直线
    :param x_arr: 数据集
    :param y_arr: 结果集
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    x_t_x = x_mat.T*x_mat
    if np.linalg.det(x_t_x) == 0.0:
        # 判断x_t_x行列式是否为0(是否可逆)
        print('奇异矩阵不能求逆')
        return
    ws = x_t_x.I*(x_mat.T*y_mat)
    # 返回参数向量
    return ws


def lwlr(test_point, x_arr, y_arr, k=1.0):
    """局部加权线性回归
    给出x空间的任意一点，计算出对应的预测值y_hat
    :param test_point: 测试数据点
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param k: 权重参数，决定对附近的点赋予多大权重，控制衰减速度
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    m = np.shape(x_mat)[0]
    # 创建对角权重矩阵
    weights = np.mat(np.eye(m))
    for j in range(m):
        # 计算高斯核对应的权重
        # 随着样本点与待预测点距离的递增，权重将以指数基衰减
        diff_mat = test_point - x_mat[j, :]
        weights[j, j] = np.exp(diff_mat*diff_mat.T/(-2.0*k**2))
    x_t_x = x_mat.T*(weights*x_mat)
    if np.linalg.det(x_t_x) == 0.0:
        print('奇异矩阵不能求逆')
        return
    ws = x_t_x.I*(x_mat.T*(weights*y_mat))
    return test_point*ws


def lwlr_test(test_arr, x_arr, y_arr, k=1.0):
    """测试局部加权线性回归
    :param test_arr: 测试数据集
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param k: 权重参数，决定对附近的点赋予多大权重，控制衰减速度
    :return:
    """
    m = np.shape(test_arr)[0]
    y_hat = np.zeros((m,1))
    for i in range(m):
        y_hat[i] = lwlr(test_arr[i], x_arr, y_arr, k)
    return y_hat


def rss_error(y_arr, y_hat):
    """计算真实值与预测值的误差
    :param y_arr: 真实值
    :param y_hat: 预测值
    :return:
    """
    return ((y_arr-y_hat)**2).sum()


def ridge_regress(x_mat, y_mat, lam=0.2):
    """岭回归
    :param x_mat: 数据集
    :param y_mat: 结果集
    :param lam: 缩减系数
    :return:
    """
    x_t_x = x_mat.T*x_mat
    denom = x_t_x+np.eye(np.shape(x_mat)[1])*lam
    if np.linalg.det(denom) == 0.0:
        print('奇异矩阵不能求逆')
        return
    ws = denom.I*(x_mat.T*y_mat)
    return ws


def ridge_test(x_arr, y_arr):
    """测试岭回归
    :param x_arr: 数据集
    :param y_arr: 结果集
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T

    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    # 对特征进行标准化处理
    # 所有特征减去各自的均值并除以方差
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat-x_mean)/x_var
    num_test_pts = 30
    w_mat = np.zeros((num_test_pts, np.shape(x_mat)[1]))
    for i in range(num_test_pts):
        # 在30个不同的lambda下求回归系数
        # lambda以指数级变化
        ws = ridge_regress(x_mat, y_mat, np.exp(i-10))
        w_mat[i, :] = ws.T
    return w_mat


def stage_wise(x_arr, y_arr, eps=0.01, num_it=100):
    """逐步线性回归
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param eps: 每次迭代需要调整的步长
    :param num_it: 迭代次数
    :return:
    """
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    # 对数据进行标准化处理
    y_mean = np.mean(y_mat, 0)
    y_mat = y_mat - y_mean
    x_mean = np.mean(x_mat, 0)
    x_var = np.var(x_mat, 0)
    x_mat = (x_mat - x_mean) / x_var
    m, n = np.shape(x_mat)
    return_mat = np.zeros((num_it, n))
    ws = np.zeros((n, 1))
    ws_max = ws.copy()
    for i in range(num_it):
        # 开始优化
        print(ws.T)
        # 将当前最小误差设置为正无穷
        lowest_error = np.inf
        for j in range(n):
            # 对每个特征
            for sign in [-1, 1]:
                # 增大或缩小
                ws_test = ws.copy()
                # 改变一个系数得到新的ws
                # 计算预测值
                # 计算误差
                ws_test[j] += eps*sign
                y_test = x_mat*ws_test
                rss_err = rss_error(y_mat.A, y_test.A)
                if rss_err < lowest_error:
                    # 更新最小误差
                    lowest_error = rss_err
                    ws_max = ws_test
        # 更新回归系数
        ws = ws_max.copy()
        return_mat[i, :] = ws.T
    return return_mat


def scrape_page(in_file, out_file, yr, num_pce, orig_prc):
    """解析下载的网页
    :param in_file: 读取文件
    :param out_file: 写入文件
    :param yr: 年份
    :param num_pce: 数量
    :param orig_prc: 价格
    :return:
    """
    from bs4 import BeautifulSoup
    with open(in_file, 'r', encoding='utf-8') as in_f:
        with open(out_file, 'a', encoding='utf-8') as out_f:
            soup = BeautifulSoup(in_f.read(), 'lxml')
            i = 1
            current_row = soup.findAll('table', r='%d' % i)
            while len(current_row) != 0:
                title = current_row[0].findAll('a')[1].text
                lwr_title = title.lower()
                if lwr_title.find('new') > -1 or lwr_title.find('nisb') > -1:
                    new_flag = 1.0
                else:
                    new_flag = 0.0
                sold_uniccde = current_row[0].findAll('td')[3].findAll('span')
                if len(sold_uniccde) == 0:
                    print('item %d did not sell' % i)
                else:
                    sold_price = current_row[0].findAll('td')[4]
                    price_str = sold_price.text
                    price_str = price_str.replace('$', '')
                    price_str = price_str.replace(',', '')
                    if len(sold_price) > 1:
                        price_str = price_str.replace('Free shipping','')
                    print('%s\t%d\t%s' % (price_str, new_flag, title))
                    out_f.write('%d\t%d\t%d\t%f\t%s\n' % (yr, num_pce, new_flag, orig_prc, price_str))
                i += 1
                current_row = soup.findAll('table', r='%d' % i)


def set_data_collect():
    """设置数据集
    :return:
    """
    scrape_page('data/setHtml/lego8288.html', 'data/lego.txt', 2006, 800, 49.99)
    scrape_page('data/setHtml/lego10030.html', 'data/lego.txt', 2002, 3096, 269.99)
    scrape_page('data/setHtml/lego10179.html', 'data/lego.txt', 2007, 5195, 499.99)
    scrape_page('data/setHtml/lego10181.html', 'data/lego.txt', 2007, 3428, 199.99)
    scrape_page('data/setHtml/lego10189.html', 'data/lego.txt', 2008, 5922, 299.99)
    scrape_page('data/setHtml/lego10196.html', 'data/lego.txt', 2009, 3263, 249.99)


def cross_validation(x_arr, y_arr, num_val=10):
    """交叉验证集测试岭回归
    :param x_arr: 数据集
    :param y_arr: 结果集
    :param num_val: 交叉验证次数
    :return:
    """
    m = len(x_arr)
    index_list = list(range(m))
    error_mat = np.zeros((num_val, 30))
    for i in range(num_val):
        train_x, train_y = [], []
        test_x, test_y = [], []
        # 对元素进行混洗，实现对训练集和测试集数据点的随机选取
        np.random.shuffle(index_list)
        for j in range(m):
            if j < m*0.9:
                # 90%分隔成训练集，其余10%为测试集
                train_x.append(x_arr[index_list[j]])
                train_y.append(y_arr[index_list[j]])
            else:
                test_x.append(x_arr[index_list[j]])
                test_y.append(y_arr[index_list[j]])
        # 保存所有回归系数
        w_mat = ridge_test(train_x, train_y)
        for k in range(30):
            # 使用30个不同的lambda值创建30组不同的回归系数
            mat_test_x = np.mat(test_x)
            mat_train_x = np.mat(train_x)
            # 岭回归需要使用标准化数据
            # 对数据进行标准化
            mean_train = np.mean(mat_train_x, 0)
            var_train = np.var(mat_train_x, 0)
            mat_test_x = (mat_test_x-mean_train)/var_train
            y_est = mat_test_x*np.mat(w_mat[k, :]).T + np.mean(train_y)
            # 计算误差
            # 保存每个lambda对应的多个误差值
            error_mat[i, k] = rss_error(y_est.T.A, np.array(test_y))
    # 计算误差的均值
    mean_errors = np.mean(error_mat, 0)
    min_mean = float(min(mean_errors))
    best_weights = w_mat[np.nonzero(mean_errors == min_mean)]
    x_mat = np.mat(x_arr)
    y_mat = np.mat(y_arr).T
    mean_x = np.mean(x_mat, 0)
    var_x = np.var(x_mat, 0)
    # 数据还原
    un_reg = best_weights/var_x
    print('岭回归最好的模型是：\n', un_reg)
    print('常数项是：', -1*np.sum(np.multiply(mean_x, un_reg)) + np.mean(y_mat))
