#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np


# 树结构
class TreeNode:
    def __init__(self, name_value, num_occur, parent_node):
        """初始化方法
        :param name_value: 节点名称
        :param num_occur: 对应节点元素的个数
        :param parent_node: 父节点
        """
        self.name = name_value
        self.count = num_occur
        # 链接相似的元素项
        self.node_link = None
        self.parent = parent_node
        self.children = {}

    def inc(self, num_occur):
        """计数值
        :param num_occur: 数量
        :return:
        """
        self.count += num_occur

    def display(self, ind=1):
        """显示树结构
        :param ind: 子树锁进个数
        :return:
        """
        print(' '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.display(ind+1)


def create_tree(data_set, min_sup=1):
    """构建FP树
    :param data_set: 数据集
    :param min_sup: 最小支持度
    :return:
    """
    # 用字典创建头指针表
    header_table = {}
    for trans in data_set:
        # 扫描数据集
        for item in trans:
            # 统计每个元素项出现的频度
            # 将信息存储在头指针标中
            header_table[item] = header_table.get(item, 0) + data_set[trans]
    keys = header_table.keys()
    for key in list(keys):
        if header_table[key] < min_sup:
            # 删除头指针表中出现次数少于最小支持度的项
            del(header_table[key])
    # 对头指针表中的数据去重
    freq_item_set = set(header_table.keys())
    if len(freq_item_set) == 0:
        # 如果去重后的数据为空则返回None
        return None, None
    for key in header_table:
        # 扩展头指针表
        # 用于保存元素项的计数值及指向第一个元素项的指针
        header_table[key] = [header_table[key], None]
    # 创建只包含空集合的根节点
    ret_tree = TreeNode('Null Set', 1, None)
    for tran_set, count in data_set.items():
        # 遍历数据集
        local_d = {}
        for item in tran_set:
            if item in freq_item_set:
                # 统计元素全局频率
                local_d[item] = header_table[item][0]
        if len(local_d) > 0:
            # 如果元素大于0
            # 根据全局频率对每个事务中的元素进行排序
            ordered_items = [v[0] for v in sorted(local_d.items(),
                                                  key=lambda p: p[1], reverse=True)]
            # 更新树
            update_tree(ordered_items, ret_tree, header_table, count)
    return ret_tree, header_table


def update_tree(items, in_tree, header_table, count):
    """更新树
    :param items: 一个项集
    :param in_tree: 树
    :param header_table: 头指针表
    :param count: 该项集对应的计数
    :return:
    """
    if items[0] in in_tree.children:
        # 测试事务的第一恶搞元素是否作为子节点
        # 如果是，更新该元素项的计数
        in_tree.children[items[0]].inc(count)
    else:
        # 如果不存在
        # 创一个新的子节点添加到树中
        in_tree.children[items[0]] = TreeNode(items[0], count, in_tree)
        if header_table[items[0]][1] is None:
            # 如果头指针没有指向
            # 将其添加到头指针指向
            header_table[items[0]][1] = in_tree.children[items[0]]
        else:
            # 否则在该元素项头指针列表后面进行更新
            update_header(header_table[items[0]][1], in_tree.children[items[0]])
    if len(items) > 1:
        # 迭代调用
        # 每次去掉列表中的第一个元素
        update_tree(items[1::], in_tree.children[items[0]], header_table, count)


def update_header(node_to_test, target_node):
    """更新头指针列表
    :param node_to_test: 指针为止
    :param target_node: 当前节点
    :return:
    """
    while node_to_test.node_link is not None:
        # 更新指针为止直到成为尾指针
        node_to_test = node_to_test.node_link
    # 将该节点添加到尾指针后面
    node_to_test.node_link = target_node


def load_simple_dat():
    """创建简单数据集
    :return:
    """
    simple_dat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simple_dat


def create_init_set(data_set):
    """包装数据为字典
    :param data_set: 数据集
    :return:
    """
    ret_dict = {}
    for trans in data_set:
        ret_dict[frozenset(trans)] = 1
    return ret_dict


def ascend_tree(leaf_node, prefix_path):
    """上溯FP树
    :param leaf_node: 树节点
    :param prefix_path: 前缀路径
    :return:
    """
    if leaf_node.parent is not None:
        # 如果树节点的父节点不为空
        # 添加前缀路径
        prefix_path.append(leaf_node.name)
        # 上溯父节点
        ascend_tree(leaf_node.parent, prefix_path)


def find_prefix_path(base_path, tree_node):
    """寻找前缀路径
    :param base_path: 基本路径
    :param tree_node: 树节点
    :return:
    """
    # 条件模式基字典
    cond_paths = {}
    while tree_node is not None:
        # 如果树节点不为空
        prefix_path = []
        # 从该节点上溯
        ascend_tree(tree_node, prefix_path)
        if len(prefix_path) > 1:
            # 更新该条件模式基的数目
            cond_paths[frozenset(prefix_path[1:])] = tree_node.count
        # 指向链表的下一个元素
        tree_node = tree_node.node_link
    return cond_paths


def mine_tree(in_tree, header_table, min_sup, pre_fix, freq_item_list):
    """查找频繁项集
    :param in_tree: 树
    :param header_table: 头指针列表
    :param min_sup: 最小支持度
    :param pre_fix: 前缀路径
    :param freq_item_list: 频繁项列表
    :return:
    """
    # 对头指针表中的元素按照其出现频率进行排序
    big_list = [v[0] for v in sorted(header_table.items(), key=lambda p: p[0])]
    for base_pat in big_list:
        # 将频繁项添加到频繁项列表
        new_freq_set = pre_fix.copy()
        new_freq_set.add(base_pat)
        freq_item_list.append(new_freq_set)
        # 查找条件模式基
        cond_patt_bases = find_prefix_path(base_pat, header_table[base_pat][1])
        # 创建条件基对应的FP树
        my_cond_tree, my_head = create_tree(cond_patt_bases, min_sup)
        if my_head is not None:
            # 如果树中有元素项递归调用
            print('conditional tree for: ', new_freq_set)
            my_cond_tree.display()
            mine_tree(my_cond_tree, my_head, min_sup, new_freq_set, freq_item_list)