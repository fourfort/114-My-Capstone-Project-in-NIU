# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：plot_results.py
@IDE ：PyCharm
@Motto:学习新思想，争做新青年
"""

import pandas as pd
import matplotlib.pyplot as plt

# 训练结果文件列表与模型标签
results_files = [
    'D:/123main/runs/exp1/results.csv',
    'D:/123main/runs/exp/results.csv',
]

# 与results_files顺序对应
custom_labels = [
    'yolov8s-EMAttention',
    'yolov8s',
]

# PR 和 F1 曲线文件路径
pr_csv_dict = {
    'YOLOv5m': r'F:\ChromeDown\yolov5-6.1-pruning-autodl\yolov5-6.1-pruning-autodl\runs\val\exp\PR_curve.csv',
    'YOLOv7': r'G:\pycharmprojects\yolov7-distillation\runs\test\exp\PR_curve.csv',
    'YOLOv7-tiny': r'G:\pycharmprojects\yolov7-distillation\runs\test\exp2\PR_curve.csv',
    'YOLOv8s': r'G:\pycharmprojects\ultralytics-main\runs\detect\yolov8s-from-ultralytics-main-bs111\PR_curve.csv',
}

f1_csv_dict = {
    'YOLOv5m': r'F:\ChromeDown\yolov5-6.1-pruning-autodl\yolov5-6.1-pruning-autodl\runs\val\exp\F1_curve.csv',
    'YOLOv7': r'G:\pycharmprojects\yolov7-distillation\runs\test\exp5\F1_curve.csv',
    'YOLOv7-tiny': r'G:\pycharmprojects\yolov7-distillation\runs\test\exp4\F1_curve.csv',
    'YOLOv8s': r'G:\pycharmprojects\ultralytics-main\runs\detect\yolov8s-from-ultralytics-main-bs111\F1_curve.csv'
}


# 通用绘图函数
def plot_comparison(metrics, labels, custom_labels, layout=(2, 2)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10))  # 创建网格布局
    axes = axes.flatten()  # 将子图对象展平，方便迭代

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        for file_path, custom_label in zip(results_files, custom_labels):
            df = pd.read_csv(file_path)

            # 清理列名中的多余空格
            df.columns = df.columns.str.strip()

            # 检查 'epoch' 列是否存在
            if 'epoch' not in df.columns:
                print(f"'epoch' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 检查目标指标列是否存在
            if metric_key not in df.columns:
                print(f"'{metric_key}' column not found in {file_path}. Available columns: {df.columns}")
                continue

            # 在对应的子图上绘制线条
            axes[i].plot(df['epoch'], df[metric_key], label=f'{custom_label}')

        axes[i].set_title(f'{metric_label}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(metric_label)
        axes[i].legend()

    plt.tight_layout()  # 自动调整子图布局，防止重叠
    plt.show()


# 绘制 PR 曲线
def plot_PR():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname, res_path in pr_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    fig.savefig("pr.png", dpi=250)
    plt.show()


# 绘制 F1 曲线
def plot_F1():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname, res_path in f1_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    fig.savefig("F1.png", dpi=250)
    plt.show()


if __name__ == '__main__':
    # 绘制精度对比图
    metrics = [
        'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    ]
    labels = [
        'Precision', 'Recall', 'mAP@50', 'mAP@50-95'
    ]
    plot_comparison(metrics, labels, custom_labels, layout=(2, 2))

    # 绘制损失对比图
    loss_metrics = [
        'train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
    ]
    loss_labels = [
        'Train Box Loss', 'Train Class Loss', 'Train DFL Loss', 'Val Box Loss', 'Val Class Loss', 'Val DFL Loss'
    ]
    plot_comparison(loss_metrics, loss_labels, custom_labels, layout=(2, 3))

    # 绘制 PR 曲线
    plot_PR()

    # 绘制 F1 曲线
    plot_F1()
