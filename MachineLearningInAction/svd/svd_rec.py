#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np
import numpy.linalg as la


def load_ex_data():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def load_ex_data_2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def eclud_sim(in_a, in_b):
    """欧式距离计算相似度
    相似度=1/(1+距离)
    :param in_a: 矩阵A
    :param in_b: 矩阵B
    :return:
    """
    return 1.0/(1.0+la.norm(in_a-in_b))


def pears_sim(in_a, in_b):
    """皮尔逊相关系数计算相似度
    相似度=0.5+0.5*相关系数
    :param in_a: 矩阵A
    :param in_b: 矩阵B
    :return:
    """
    if len(in_a) < 3:
        return 1.0
    return 0.5+0.5*np.corrcoef(in_a, in_b, rowvar=0)[0][1]


def cos_sim(in_a, in_b):
    """余弦相似度
    相似度=向量乘积/范数乘积
    :param in_a: 矩阵A
    :param in_b: 矩阵B
    :return:
    """
    num = np.float(in_a.T*in_b)
    denom = la.norm(in_a)*la.norm(in_b)
    return 0.5+0.5*(num/denom)


def stand_est(data_mat, user, sim_meas, item):
    """计算用户对物品的估计评分值
    :param data_mat: 数据矩阵
    :param user: 用户编号
    :param sim_meas: 相似度计算方法
    :param item: 物品编号
    :return:
    """
    # 获取物品总数
    n = np.shape(data_mat)[1]
    # 总的相似度
    sim_total = 0.0
    # 总的相似度评分
    rat_sim_total = 0.0
    for j in range(n):
        # 对每一个物品
        # 计算用户的评分
        user_rating = data_mat[user, j]
        if user_rating == 0:
            # 如果用户没有对该物品评分
            # 跳过该物品
            continue
        # 找到重合的元素
        # 即寻找用户都评级的两个物品
        over_lap = np.nonzero(np.logical_and(data_mat[:, item].A > 0,
                                             data_mat[:, j].A > 0))[0]
        if len(over_lap) == 0:
            # 如果没有重合的元素
            similarity = 0
        else:
            # 计算重合物品的相似度
            similarity = sim_meas(data_mat[over_lap, item], data_mat[over_lap, j])
        print('%d 和 %d 相似度为 %f' % (item, j, similarity))
        # 累加相似度
        sim_total += similarity
        # 计算相似度和当前用户评分的乘积
        rat_sim_total += similarity*user_rating
    if sim_total == 0:
        # 直接退出
        return 0
    else:
        # 对评分进行归一化
        return rat_sim_total/sim_total


def recommend(data_mat, user, n=3, sim_meas=cos_sim, est_method=stand_est):
    """推荐
    :param data_mat: 数据矩阵
    :param user: 用户编号
    :param n: 推荐结果数量
    :param sim_meas: 相似度计算函数
    :param est_method: 估计方法
    :return:
    """
    # 找到用户未评级的武平
    unrated_items = np.nonzero(data_mat[user, :].A == 0)[1]
    if len(unrated_items) == 0:
        return '你对所有物品都进行了评级'
    item_scores = []
    for item in unrated_items:
        # 对每一个未评级的物品
        # 计算估计评分
        estimated_score = est_method(data_mat, user, sim_meas, item)
        # 将物品编号和对应评分存入列表
        item_scores.append((item, estimated_score))
    # 排序后返回前n个值
    return sorted(item_scores, key=lambda j: j[1], reverse=True)[:n]


def svd_est(data_mat, user, sim_meas, item):
    """基于SVD的评分估计
    :param data_mat:
    :param user:
    :param sim_meas:
    :param item:
    :return:
    """
    n = np.shape(data_mat)[1]
    sim_total = 0.0
    rat_sim_total = 0.0
    # 使用SVD分解
    u, sigma, vt = la.svd(data_mat)
    # 使用包含90%能量值的奇异值
    # 将其转换为一个对角矩阵
    sig4 = np.mat(np.eye(4)*sigma[:4])
    # 使用U矩阵将物品转换到低维空间
    x_formed_items = data_mat.T*u[:, :4]*sig4.I
    for j in range(n):
        # 对所有物品
        # 获取用户的评分值
        user_rating = data_mat[user, j]
        if user_rating == 0 or j == item:
            continue
        # 计算低维空间下的相似度
        similarity = sim_meas(x_formed_items[item, :].T, x_formed_items[j, :].T)
        print('%d 和 %d 相似度为 %f' % (item, j, similarity))
        # 累加相似度
        sim_total += similarity
        # 计算相似度和当前用户评分的乘积
        rat_sim_total += similarity*user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total/sim_total


def print_mat(in_mat, thresh=0.8):
    """打印矩阵
    :param in_mat: 输入矩阵
    :param thresh: 阈值
    :return:
    """
    for i in range(32):
        for k in range(32):
            # 遍历所有元素
            if np.float(in_mat[i, k]) > thresh:
                # 大于阈值打印1
                print(1, end='')
            else:
                # 否则打印0
                print(0, end='')
        print('\n')


def img_compress(num_sv=3, thresh=0.8):
    """图像压缩
    :param num_sv: 奇异值数目
    :param thresh: 阈值
    :return:
    """
    myl = []
    with open('data/0_5.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            new_row = []
            for i in range(32):
                new_row.append(np.int(line[i]))
            myl.append(new_row)
    my_mat = np.mat(myl)
    print('原始矩阵')
    print_mat(my_mat, thresh)
    u, sigma, v_t = la.svd(my_mat)
    sig_recon = np.mat(np.zeros((num_sv, num_sv)))
    for k in range(num_sv):
        sig_recon[k, k] = sigma[k]
    recon_mat = u[:, :num_sv]*sig_recon*v_t[:num_sv, :]
    print('使用%d个奇异值重构矩阵' % num_sv)
    print_mat(recon_mat, thresh)

