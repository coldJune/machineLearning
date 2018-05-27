#! /usr/sbin/python3
# -*- coding:utf-8 -*-

import numpy as np
import reg_trees
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def re_draw(tol_s, tol_n):
    """绘图
    :param tol_s:
    :param tol_n:
    :return:
    """
    # 清空之前的图像
    re_draw.f.clf()
    # 添加一个新图
    re_draw.a = re_draw.f.add_subplot(111)
    if chk_btn_var.get():
        # 判断复选框是否被选中
        # 选中创建模型树
        if tol_n < 2:
            tol_n = 2
        my_tree = reg_trees.create_tree(re_draw.raw_dat,
                                        reg_trees.model_leaf, reg_trees.model_err, (tol_s, tol_n))
        y_hat = reg_trees.create_for_cast(my_tree, re_draw.test_dat, reg_trees.model_tree_eval)
    else:
        # 否则创建回归树
        my_tree = reg_trees.create_tree(re_draw.raw_dat, ops=(tol_s, tol_n))
        y_hat = reg_trees.create_for_cast(my_tree, re_draw.test_dat)
    # 绘制真实值
    re_draw.a.scatter(np.array(re_draw.raw_dat[:, 0]), np.array(re_draw.raw_dat[:, 1]), s=5)
    # 绘制预测值
    re_draw.a.plot(re_draw.test_dat, y_hat, linewidth=2.0)
    re_draw.canvas.show()


def get_inputs():
    """获取输入值
    :return:
    """
    try:
        tol_n = int(tol_n_entry.get())
    except:
        tol_n = 10
        print('输入tol_n的数值')
        tol_n_entry.delete(0, tk.END)
        tol_n_entry.insert(0, '10')
    try:
        tol_s = int(tol_s_entry.get())
    except:
        tol_s = 1.0
        print('输入tol_s的数值')
        tol_s_entry.delete(0, tk.END)
        tol_s_entry.insert(0, '1.0')
    return tol_n, tol_s


def draw_new_tree():
    """画图入口
    :return:
    """
    tol_n, tol_s = get_inputs()
    re_draw(tol_s, tol_n)


root = tk.Tk()

# 创建画布
re_draw.f = Figure(figsize=(5, 4), dpi=100)
re_draw.canvas = FigureCanvasTkAgg(re_draw.f, master=root)
re_draw.canvas.show()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)
# 生成tol_n的输入框
tk.Label(root, text='tol_n').grid(row=1, column=0)
tol_n_entry = tk.Entry(root)
tol_n_entry.grid(row=1, column=1)
tol_n_entry.insert(0, '10')
# 生成tol_s的输入框
tk.Label(root, text='tol_s').grid(row=2, column=0)
tol_s_entry = tk.Entry(root)
tol_s_entry.grid(row=2, column=1)
tol_s_entry.insert(0, '1.0')
# 生成重绘按钮
tk.Button(root, text='重绘', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

# 切换树结构
chk_btn_var = tk.IntVar()
chk_btn = tk.Checkbutton(root, text="模型树", variable=chk_btn_var)
chk_btn.grid(row=3, column=0, columnspan=2)
# 加载数据
re_draw.raw_dat = np.mat(reg_trees.load_data_set('data/sine.txt'))
re_draw.test_dat = np.arange(min(re_draw.raw_dat[:, 0]), max(re_draw.raw_dat[:, 0]), 0.01)
re_draw(1.0, 10)
root.mainloop()
