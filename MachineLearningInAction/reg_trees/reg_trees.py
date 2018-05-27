#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np


class TreeNode:
    """
    建立树节点
    """
    def __init__(self, feat, val, right, left):
        feature_to_splition = feat
        value_of_split = val
        right_branch = right
        left_branch = left


def load_data_set(file_name):
    """加载数据
    :param file_name: 文件名
    :return:
    """
    data_mat = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cur_line = line.strip().split('\t')
            # 将每行映射成为浮点数
            flt_line = list(map(np.float, cur_line))
            data_mat.append(flt_line)
    return data_mat


def bin_split_data_set(data_set, feature, value):
    """将数据切分成两个子集并返回
    :param data_set: 数据集
    :param feature: 待切分的特征
    :param value: 该特征的某个值
    :return:
    """
    # nonzero返回的是index
    mat_0 = data_set[np.nonzero(data_set[:, feature] > value)[0], :]
    mat_1 = data_set[np.nonzero(data_set[:, feature] <= value)[0], :]
    return mat_0, mat_1


def reg_leaf(data_set):
    """生成叶节点
    :param data_set: 数据集
    :return:
    """
    # 回归树中是目标变量的均值
    return np.mean(data_set[:, -1])


def reg_err(data_set):
    """误差估计函数
    :param data_set:
    :return:
    """
    # 总方差等于均方差乘以样本个数
    return np.var(data_set[:, -1]) * np.shape(data_set)[0]


def create_tree(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """创建树
    :param data_set: 数据集
    :param leaf_type: 建立叶节点的函数
    :param err_type: 误差计算函数
    :param ops: 树构建所需要的其他元组
    :return:
    """
    # 切分数据集
    feat, val = choose_best_split(data_set, leaf_type, err_type, ops)
    if feat is None:
        # 如果满足条件
        # 则返回叶节点
        return val
    # 不满足则创建新的字典
    ret_tree = dict({})
    ret_tree['sp_ind'] = feat
    ret_tree['sp_val'] = val
    # 切分数据集
    l_set, r_set = bin_split_data_set(data_set, feat, val)
    # 递归调用创建树
    ret_tree['left'] = create_tree(l_set, leaf_type, err_type, ops)
    ret_tree['right'] = create_tree(r_set, leaf_type, err_type, ops)
    return ret_tree


def choose_best_split(data_set, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """寻找数据的最佳二元切分方式
    :param data_set: 数据集
    :param leaf_type: 生成叶节点函数
    :param err_type: 误差函数
    :param ops: 用于控制函数的停止时机
    :return:
    """
    # 容许的误差下降值
    tol_s = ops[0]
    # 切分的最少样本树
    tol_n = ops[1]
    if len(set(data_set[:, -1].T.tolist()[0])) == 1:
        # 统计不同特征的数目
        # 如果为1就不再切分直接返回
        return None, leaf_type(data_set)
    # 计算当前数据集的大小和误差
    m, n = np.shape(data_set)
    # 误差用于与新切分的误差进行对比
    # 来检查新切分能否降低误差
    s = err_type(data_set)
    best_s = np.inf
    best_index = 0
    best_value = 0
    for feat_index in range(n-1):
        # 遍历所有特征
        for split_val in set(data_set[:, feat_index].T.tolist()[0]):
            # 遍历该特征上的所有取值
            mat_0, mat_1 = bin_split_data_set(data_set, feat_index, split_val)
            if np.shape(mat_0)[0] < tol_n or np.shape(mat_1)[0] < tol_n:
                # 如果切分后子集不够大
                # 进入下一次循环
                continue
            new_s = err_type(mat_0) + err_type(mat_1)
            if new_s < best_s:
                # 如果新子集的误差小于最好误差
                # 更新特征下标/切分值/最小误差
                best_index = feat_index
                best_value = split_val
                best_s = new_s
    if s-best_s < tol_s:
        # 如果切分数据集后提升不大
        #  则不在进行切分而直接创建叶节点
        return None, leaf_type(data_set)
    mat_0, mat_1 = bin_split_data_set(data_set, best_index, best_value)
    if np.shape(mat_0)[0] < tol_n or np.shape(mat_1)[0] < tol_n:
        # 如果切分出的数据集很小则退出
        return None, leaf_type(data_set)
    return best_index, best_value


def is_tree(obj):
    """判断是否是树
    :param obj: 需要判断的对象
    :return:
    """
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """计算叶节点的平均值
    :param tree: 树
    :return:
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, test_data):
    """ 树剪枝
    :param tree: 待剪枝的树
    :param test_data: 剪枝所需的测试数据
    :return:
    """
    if np.shape(test_data)[0] == 0:
        # 确认数据集是否是空
        return get_mean(tree)
    # 如果是子树就对该子树进行剪枝
    if is_tree(tree['right']) or is_tree(tree['left']):
        l_set, r_set = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], l_set)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], r_set)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        # 如果不是子树就进行合并
        l_set, r_set = bin_split_data_set(test_data, tree['sp_ind'], tree['sp_val'])
        error_no_merge = np.sum(np.power(l_set[:, -1]-tree['left'], 2))+np.sum(np.power(r_set[:, -1]-tree['right'], 2))
        tree_mean = (tree['left']+tree['right'])/2.0
        error_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if error_merge < error_no_merge:
            # 比较合并前后的误差
            # 合并后的误差比合并前的误差小
            # 就返回合并的树
            print("merging")
            return tree_mean
        else:
            # 否则直接返回
            return tree
    else:
        return tree


# 模型树
def linear_solve(data_set):
    """目标变量转换
    将数据集格式化成目标变量Y和自变量X
    :param data_set: 数据集
    :return:
    """
    m, n = np.shape(data_set)
    x = np.mat(np.ones((m, n)))
    x[:, 1:n] = data_set[:, 0:n-1]
    y = data_set[:, -1]
    x_t_x = x.T*x
    if np.linalg.det(x_t_x) == 0.0:
        raise NameError('该矩阵不能求逆，尝试提高ops参数的第二个值')
    ws = x_t_x.I*(x.T*y)
    return ws, x, y


def model_leaf(data_set):
    """生成叶子节点的回归系数
    :param data_set: 数据集
    :return:
    """
    ws, x, y = linear_solve(data_set)
    return ws


def model_err(data_set):
    """计算误差
    :param data_set: 数据集
    :return:
    """
    ws, x, y = linear_solve(data_set)
    y_hat = x*ws
    return np.sum(np.power(y-y_hat, 2))


def reg_tree_eval(model, in_dat):
    """数据转换
    :param model: 树结构
    :param in_dat: 输入数据
    :return:
    """
    return np.float(model)


def model_tree_eval(model, in_data):
    """数据转换
    :param model: 树结构
    :param in_data: 输入数据
    :return:
    """
    n = np.shape(in_data)[1]
    x = np.mat(np.ones((1, n+1)))
    x[:, 1:n+1] = in_data
    return np.float(x*model)


def tree_fore_cast(tree, in_data, model_eval=reg_tree_eval):
    """预测函数
    :param tree: 树结构
    :param in_data: 输入数据
    :param model_eval: 树的类型
    :return:
    """
    if not is_tree(tree):
        return model_err(tree, in_data)
    if in_data[tree['sp_ind']] > tree['sp_val']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], in_data, model_eval)
        else:
            return model_eval(tree['left'], in_data)
    else:
        if is_tree(tree['right']):
            return tree_fore_cast(tree['right'], in_data, model_eval)
        else:
            return model_eval(tree['right'], in_data)


def create_for_cast(tree, test_data, model_eval=reg_tree_eval):
    """ 测试函数
    :param tree: 树结构
    :param test_data: 测试数据
    :param model_eval: 树类型
    :return:
    """
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[1, 0] = tree_fore_cast(tree, np.mat(test_data[i]), model_eval)
    return y_hat

