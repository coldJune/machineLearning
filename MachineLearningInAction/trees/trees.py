#! usr/sbin/python3
# -*- coding:utf-8 -*-

from math import log
import operator
import pickle


def calc_shannon_ent(data_set):
    """计算给定数据集的香农熵
    :param data_set:
    :return:
    """
    # 计算数据集大小
    num_entries = len(data_set)
    # 创建数据字典，键值为数据集最后一列的值
    # 计算每一个键值出现的次数
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        # 使用类标签的发生频率计算类别出现的概率
        # 计算香农熵
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def split_data_set(data_set, axis, value):
    """按照给定特征划分数据集
    :param data_set: 待划分的数据集
    :param axis: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return:
    """
    # 创建新的list对象
    ret_data_set = []
    for feat_vec in data_set:
        # 遍历数据集中每个元素，发现符合要求的值将其添加到新的列表中
        # 即按照某个特征划分数据集时将所有符合要求的元素抽取出来
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis+1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """选取最好的数据集划分方式
    循环计算香农熵和split_data_set()函数，熵计算会说明划分数据集的最好数据组织方式
    1.数据必须是一种由列表元素组成的列表，所有列表元素长度相同
    2.数据的最后一列或每个实例的最后一个元素是当前实例的类别标签
    :param data_set: 待划分的数据集
    :return:
    """
    num_features = len(data_set[0]) - 1
    # 计算整个数据集的原始香农熵，用于与划分完之后的数据集计算的熵值进行比较
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain, best_feature = 0.0, -1
    for i in range(num_features):
        # 遍历数据集中的所有特征
        # 将数据集中所有第i个特征值或者所有可能存在的值写入新的list中
        feat_list = [example[i] for example in data_set]
        # 去重
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            # 遍历所有当前特征中的所有唯一属性值
            # 对每个唯一属性划分一次数据集
            # 然后计算数据集的新熵值
            # 对所有数据集求得的熵值求和
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            # 比较信息增益，返回最好特征的索引值
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    """多数表决
    计算每个类标签出现的频率，返回出现次数最多的分类名称
    :param class_list: 分类名称的列表
    :return:
    """
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """创建树
    :param data_set: 数据集
    :param labels: 标签列表
    :return:
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        # 所有类标签完全相同则停止划分
        return class_list
    if len(data_set[0]) == 1:
        # 使用完所有特征停止划分
        return majority_cnt(class_list)
    # 存储最好特征
    # 得到列表包含的所有属性值
    best_feature = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feature]
    my_tree = {best_feat_label: {}}
    del(labels[best_feature])
    feat_values = [example[best_feature] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        # 复制类标签
        sub_labels = labels[:]
        # 在每个数据集划分上调用create_tree()
        # 得到的返回值被插入到my_tree中
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feature, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """使用决策树分类
    :param input_tree: 数据集
    :param feat_labels: 特征标签
    :param test_vec: 测试向量
    :return:
    """
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    # 找到当前列表第一个匹配first_str变量的标签
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        # 遍历整棵树，比较test_vec变量中的值与树节点的值
        # 如果到达叶子节点就返回节点的分类标签
        if test_vec[feat_index] == key:

            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, filename):
    """存储决策树
    :param input_tree: 树结构
    :param filename: 文件名
    :return:
    """
    with open(filename, 'w', encoding='utf-8') as fw:
        pickle.dump(input_tree, fw)


def grab_tree(filename):
    """读取决策树
    :param filename: 文件名
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as fr:
        return pickle.load(fr)
