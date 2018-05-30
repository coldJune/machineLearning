#! /usr/sbin/python3
# -*- coding:utf-8 -*-


import numpy as np


def load_data_set(filename, delim='\t'):
    """加载数据
    :param filename: 文件名
    :param delim: 文件分隔符
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        string_arr = [line.strip().split(delim) for line in f.readlines()]
        data_arr = [list(map(float, line)) for line in string_arr]
        return np.mat(data_arr)


def pca(data_mat, top_n_feat=9999999):
    """主成分分析法
    :param data_mat: 数据集
    :param top_n_feat: 最大的特征值的个数
    :return:
    """
    # 计算并减去原始数据的平均值
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = data_mat-mean_vals
    # 计算协方差矩阵及其特征值
    cov_mat = np.cov(mean_removed, rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))
    # 根据特征值排序
    eig_vals_ind = np.argsort(eig_vals)
    # 获取前top_n_feat个特征值的下标
    eig_vals_ind = eig_vals_ind[:-(top_n_feat+1):-1]
    # 获取特征向量
    red_eig_vects = eig_vects[:, eig_vals_ind]
    # 将数据转换到新空间中
    low_dem_data_mat = mean_removed*red_eig_vects
    recon_mat = (low_dem_data_mat*red_eig_vects.T)+mean_vals
    # 返回降维后的数据和重构后的数据
    return low_dem_data_mat, recon_mat


def replace_nan_with_mean():
    """将NAN替换为该特征的平均值
    :return:
    """
    data_mat = load_data_set('data/secom.data', ' ')
    num_feat = np.shape(data_mat)[1]
    for i in range(num_feat):
        mean_val = np.mean(data_mat[np.nonzero(~np.isnan(data_mat[:, i].A)), i])
        data_mat[np.nonzero(np.isnan(data_mat[:, i].A))[0], i] = mean_val
    return data_mat