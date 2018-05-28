#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np


def load_data_set(filename):
    """加载数据集
    :param filename: 文件名
    :return:
    """
    data_mat = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cur_line = line.strip().split('\t')
            flt_line = list(map(float, cur_line))
            data_mat.append(flt_line)
    return data_mat


def dist_eclud(vec_a, vec_b):
    """计算向量欧式距离
    :param vec_a: 向量a
    :param vec_b: 向量b
    :return:
    """
    return np.sqrt(np.sum(np.power(vec_a-vec_b, 2)))


def rand_cent(data_set, k):
    """构造质心
    :param data_set: 数据集
    :param k: 质心个数
    :return:
    """
    n = np.shape(data_set)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        # 生成质心位于每个数据点的最大值和最小值之间
        min_j = np.min(data_set[:, j])
        max_j = np.max(data_set[:, j])
        range_j = np.float(max_j-min_j)
        centroids[:, j] = min_j + range_j * np.random.rand(k, 1)
    return centroids


def k_means(data_set, k, dist_meas=dist_eclud, create_cent=rand_cent):
    """k-means算法
    :param data_set: 数据集
    :param k: 质心个数
    :param dist_meas: 距离计算方法，默认欧式距离
    :param create_cent: 创建质心方法
    :return:
    """
    # 数据点的个数
    m = np.shape(data_set)[0]
    # 每个点的簇分配结果
    # 第一列记录簇的索引值
    # 第二列记录存储误差
    cluster_assment = np.mat(np.zeros((m, 2)))
    # 生成k个质心
    centroids = create_cent(data_set, k)
    # 簇是否改变
    cluster_changed = True
    while cluster_changed:
        # 当簇发生改变
        cluster_changed = False
        for i in range(m):
            # 对每个数据点
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                # 对每个质心
                # 计算距离
                dist_j_i = dist_meas(centroids[j, :], data_set[i, :])
                if dist_j_i < min_dist:
                    # 如果距离小于最小距离
                    # 更新最小距离
                    # 更新该数据点对应的质心下标
                    min_dist = dist_j_i
                    min_index = j
            if cluster_assment[i, 0] != min_index:
                # 如果对应数据点质心改变
                # 更新标识
                cluster_changed = True
            # 更新该数据点的对应质心下标和对应的误差值
            cluster_assment[i, :] = min_index, min_dist**2
        # print(centroids)
        for cent in range(k):
            # 遍历所有质心
            pts_in_clust = data_set[np.nonzero(cluster_assment[:, 0].A == cent)[0]]
            # 更新取值,沿数据点列的方向进行均值计算
            centroids[cent, :] = np.mean(pts_in_clust, axis=0)
    return centroids, cluster_assment


def bi_k_means(data_set, k, dis_meas=dist_eclud):
    """二分k-means
    :param data_set: 数据集
    :param k: 质心个数
    :param dis_meas: 距离计算方法
    :return:
    """
    # 回去数据集大小
    m = np.shape(data_set)[0]
    # 每个点的簇分配结果
    # 第一列记录簇的索引值
    # 第二列记录存储误差
    cluster_assment = np.mat(np.zeros((m, 2)))
    # 计算整个数据集的质心
    centroid_0 = np.mean(data_set, axis=0).tolist()[0]
    cent_list = [centroid_0]
    for j in range(m):
        # 计算每个点到质心的误差值
        cluster_assment[j, 1] = dis_meas(np.mat(centroid_0), data_set[j, :])**2
    while len(cent_list) < k:
        # 对簇进行划分直到得到k个簇
        lowest_sse = np.inf
        for i in range(len(cent_list)):
            # 遍历列表中的每一个簇
            # 得到当前簇的数据
            pts_in_curr_cluster = data_set[np.nonzero(cluster_assment[:, 0].A == i)[0], :]
            # 对当前簇进行划分
            centroid_mat, split_cluster_ass = k_means(pts_in_curr_cluster, 2, dis_meas)
            # 计算划分数据的误差
            sse_split = np.sum(split_cluster_ass[:, 1])
            # 计算剩余数据集的误差
            sse_not_split = np.sum(cluster_assment[np.nonzero(cluster_assment[:, 0].A != i)[0], 1])
            print('sse_split,sse_not_split', sse_split, sse_not_split)
            if sse_split + sse_not_split < lowest_sse:
                # 如果误差小于最小误差
                # 更新质心的下标
                best_cent_to_split = i
                # 更新质心
                best_new_cents = centroid_mat
                # 更新簇数据
                best_clust_ass = split_cluster_ass.copy()
                # 更新最小误差
                lowest_sse = sse_split + sse_not_split
        # 将编号为0和1的结果簇修改为划分簇及新加簇的编号
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 1)[0], 0] = len(cent_list)
        best_clust_ass[np.nonzero(best_clust_ass[:, 0].A == 0)[0], 0] = best_cent_to_split
        print('the best_cent_to_split is:', best_cent_to_split)
        print('the len of best_clust_ass is', len(best_clust_ass))
        # 新的质心添加到cent_list
        cent_list[best_cent_to_split] = best_new_cents[0, :]
        cent_list.append(best_new_cents[1, :])
        cluster_assment[np.nonzero(cluster_assment[:, 0].A == best_cent_to_split)[0], :] = best_clust_ass
    print('cent_list:', cent_list)
    return np.mat(np.array(cent_list)), cluster_assment


def dist_slc(vec_a, vec_b):
    """计算球面距离
    :param vec_a: 向量a
    :param vec_b: 向量b
    :return:
    """
    # 给定两点的经纬度，使用球面余弦定理计算两点距离
    # sin和cos函数需先将角度转换为弧度
    a = np.sin(vec_a[0, 1]*np.pi/180)*np.sin(vec_b[0, 1]*np.pi/180)
    b = np.cos(vec_a[0, 1]*np.pi/180)*np.cos(vec_b[0, 1]*np.pi/180)\
        * np.cos(np.pi*(vec_b[0, 0]-vec_a[0, 0])/180)
    return np.arccos(a+b)*6371.0


def cluster_clubs(num_clust=5):
    """画出聚类结果
    :param num_clust:
    :return:
    """
    import matplotlib.pyplot as plt
    data_list = []
    with open('data/places.txt', 'r', encoding='utf-8') as f:
        # 提取文件中的经纬度
        for line in f.readlines():
            line_arr = line.split('\t')
            data_list.append([np.float(line_arr[4]), np.float(line_arr[3])])
    data_mat = np.mat(data_list)
    # 运行bi_k_means函数得到聚类中心以及聚类对应的数据点
    my_centroids, clust_assing = bi_k_means(data_mat, num_clust, dis_meas=dist_slc)
    # 创建一幅图和一个矩形
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    # 标记形状的列表用于绘制散点图
    scatter_markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    # 基于一副图像来创建矩阵
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    img_p = plt.imread('data/Portland.png')
    ax0.imshow(img_p)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(num_clust):
        # 用索引创建标记形状
        pts_in_curr_cluster = data_mat[np.nonzero(clust_assing[:, 0].A == i), :]
        marker_style = scatter_markers[i % len(scatter_markers)]
        ax1.scatter(pts_in_curr_cluster[:, 0].flatten().A[0],
                    pts_in_curr_cluster[:, 1].flatten().A[0], marker=marker_style, s=90)
    # 创建质心的形状
    ax1.scatter(my_centroids[:, 0].flatten().A[0], my_centroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()

