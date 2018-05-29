#! /usr/sbin/python3
# -*- coding:utf-8 -*-


import numpy as np


def load_data_set():
    """生成数据集
    :return:
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_c1(data_set):
    """构建大小为1的所有候选项集的集合
    :param data_set:
    :return:
    """
    # 构建集合C1
    candidate_1 = []
    for transaction in data_set:
        for item in transaction:
            # 提取候选集
            if not [item] in candidate_1:
                candidate_1.append([item])
    candidate_1.sort()
    # 创建一个不可改变的集合
    return list(map(frozenset, candidate_1))


def scan_d(d, candidate_k, min_support):
    """创建满足最小支持度的列表
    :param d: 数据集
    :param candidate_k: 候选集合列表
    :param min_support: 最小支持度
    :return:
    """

    ss_cnt = {}
    for tid in d:
        # 遍历所有交易记录
        for can in candidate_k:
            # 遍历所有候选集
            if can.issubset(tid):
                # 如果候选集是记录的一部分
                if can not in ss_cnt.keys():
                    # 为记载则记为1
                    ss_cnt[can] = 1
                else:
                    # 否则记录+1
                    ss_cnt[can] += 1
    # 记录的大小
    num_items = np.float(len(d))
    ret_list = []
    support_data = {}
    for key in ss_cnt:
        # 扫描所有候选集
        # 计算候选集的支持度
        support = ss_cnt[key]/num_items
        if support >= min_support:
            # 候选集的支持度大于最小支持度
            # 将候选集加入列表
            ret_list.insert(0, key)
        # 记录候选集的支持度
        support_data[key] = support
    return ret_list, support_data


def apriori_gen(list_k, k):
    """构建大小为k的所有候选项集的集合
    :param list_k: 频繁项列表
    :param k: 项集元素个数
    :return:
    """
    ret_list = []
    # 计算l_k的元素个数
    len_list_k = len(list_k)
    for i in range(len_list_k):
        for j in range(i+1, len_list_k):
            # 比较l_k中的每个元素和其他元素
            # 前k-2个元素相同
            l_1 = list(list_k[i])[:k-2]
            l_2 = list(list_k[j])[:k-2]
            l_1.sort()
            l_2.sort()
            if l_1 == l_2:
                # 求并并添加到列表中
                ret_list.append(list_k[i] | list_k[j])
    return ret_list


def apriori(data_set, min_support=0.5):
    """主函数
    :param data_set:
    :param min_support:
    :return:
    """
    # 创建大小为1的所有候选项集的集合
    candidate_1 = create_c1(data_set)
    # 数据去重
    d = list(map(set, data_set))
    # 计算候选项集为1的满足最小支持度的列表
    list_1, support_data = scan_d(d, candidate_1, min_support)
    lis = [list_1]
    k = 2
    while len(lis[k-2]) > 0:
        # 创建大小为k的所有候选项集的集合
        candidate_k = apriori_gen(lis[k-2], k)
        # 计算候选项集为k的满足最小支持度的列表
        list_k, support_k = scan_d(d, candidate_k, min_support)
        # 更新支持度字典
        support_data.update(support_k)
        # 添加候选集为k的满足最小支持度的列表
        lis.append(list_k)
        k += 1
    return lis, support_data


def rule_from_conseq(freq_set, h, support_data, big_rule_list, min_conf=0.7):
    """关联规则生成
    :param freq_set: 频繁项集
    :param h: 每个频繁项集只包含单个元素集合的列表
    :param support_data: 频繁项集支持数据的字典
    :param big_rule_list: 包含可信度的规则列表
    :param min_conf: 最小可信度
    :return:
    """
    # 计算h中频繁集大小
    m = len(h[0])
    if len(freq_set) > m+1:
        # 频繁项集大到可以移除大小为m的子集
        # 生成h中元素的无重复组合
        hmp1 = apriori_gen(h, m+1)
        # 计算可信度
        hmp1 = calc_conf(freq_set, hmp1, support_data, big_rule_list, min_conf)
        if len(hmp1) > 1:
            # 如果不止一条规则满足要求
            # 进行迭代
            rule_from_conseq(freq_set, hmp1, support_data, big_rule_list, min_conf)


def calc_conf(freq_set, h, support_data, big_rule_list, min_conf=0.7):
    """计算可信度
    :param freq_set: 频繁项集
    :param h:每个频繁项集只包含单个元素集合的列表
    :param support_data: 频繁项集支持数据的字典
    :param big_rule_list: 包含可信度的规则列表
    :param min_conf: 最小可信度
    :return:
    """
    pruned_h = []
    for conseq in h:
        # 计算所有项集的可信度
        conf = support_data[freq_set]/support_data[freq_set-conseq]
        if conf >= min_conf:
            # 可信度大于最小可信度
            print(freq_set-conseq, '-->',  conseq, 'conf:', conf)
            # 加入列表
            big_rule_list.append((freq_set-conseq, conseq, conf))
            pruned_h.append(conseq)
    return pruned_h


def generate_rules(lis, support_data, min_conf=0.7):
    """关联规则生成主函数
    :param lis: 频繁项集列表
    :param support_data: 频繁项集支持数据的字典
    :param min_conf: 最小可信度阈值
    :return:
    """
    # 包含可信度的规则列表
    big_rule_list = []
    for i in range(1, len(lis)):
        # 获取两个或更多元素的列表
        for freq_set in lis[i]:
            # 对每个频繁项集创建只包含单个元素集合的列表
            h_1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                # 如果频繁项集的元素数目超过2就对它进行进一步合并
                rule_from_conseq(freq_set, h_1, support_data, big_rule_list, min_conf)
            else:
                # 只有两个元素则计算可信度
                calc_conf(freq_set, h_1, support_data, big_rule_list, min_conf)
    return big_rule_list
