import pandas as pd
import matplotlib.pyplot as plt

results_files = [
    r'D:\python main\123\123main\pt（all）\exp1(other)\results(swin).csv',
    r'D:\python main\123\123main\pt（all）\exp1(other)\results(ghost-p6).csv',
    r'D:\python main\123\123main\pt（all）\exp(_init)\results(yolov8s).csv',
    r'D:\python main\123\123main\pt（all）\exp1(other)\results(CSPliteshufflenet).csv'
]

custom_labels = [
    'YOLOv8s+swin',
    'YOLOv8s+GhostNet',
    'YOLOv8s',
    'Ours'
]

pr_csv_dict = {
    'YOLOv8s+swin': r'D:\python main\123\123main\pt（all）\exp1(PR other)\PR_curve(swin).csv',
    'YOLOv8s+GhostNet': r'D:\python main\123\123main\pt（all）\exp1(PR other)\PR_curve(ghost-p6).csv',
    'YOLOv8s': r'D:\python main\123\123main\pt（all）\exp(PR_init)\PR_curve(yolov8s).csv',
    'Ours': r'D:\python main\123\123main\pt（all）\exp1(PR other)\PR_curve(CSPliteshufflenet).csv'
}

f1_csv_dict = {
    'YOLOv8s+swin': r'D:\python main\123\123main\pt（all）\exp1(F1 other)\F1_curve(swin).csv',
    'YOLOv8s+GhostNet': r'D:\python main\123\123main\pt（all）\exp1(F1 other)\F1_curve(ghost-p6).csv',
    'YOLOv8s': r'D:\python main\123\123main\pt（all）\exp(F1_init)\F1_curve(yolov8s).csv',
    'Ours': r'D:\python main\123\123main\pt（all）\exp1(F1 other)\F1_curve(CSPliteshufflenet).csv'
}

def plot_comparison(metrics, labels, custom_labels, layout=(2, 2)):
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(15, 10), dpi=120)
    axes = axes.flatten()

    for i, (metric_key, metric_label) in enumerate(zip(metrics, labels)):
        for file_path, custom_label in zip(results_files, custom_labels):
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            if 'epoch' not in df.columns or metric_key not in df.columns:
                continue

            axes[i].plot(df['epoch'], df[metric_key], label=custom_label)

        axes[i].set_title(metric_label, fontsize=14)
        axes[i].set_xlabel('Epochs', fontsize=12)
        axes[i].set_ylabel(metric_label, fontsize=12)
        axes[i].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_PR():
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=100, tight_layout=True)

    for modelname, res_path in pr_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2)

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=12)
    plt.grid()
    plt.show()

def plot_F1():
    fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=90, tight_layout=True)

    for modelname, res_path in f1_csv_dict.items():
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2)

    # 設定標籤與範圍
    ax.set_xlabel('Confidence', fontsize=20)
    ax.set_ylabel('F1', fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ✅ 調整 x 與 y 軸的數值字體大小
    ax.tick_params(axis='both', labelsize=16)

    # 圖例設定
    ax.legend(
        fontsize=20,
        loc='lower left',
        frameon=True,
        facecolor='white',
        edgecolor='black'
    )

    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_comparison(
        ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'],
        ['Precision', 'Recall', 'mAP@50', 'mAP@50-95'],
        custom_labels,
        layout=(2, 2)
    )

    plot_comparison(
        ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'],
        ['Train Box Loss', 'Train Class Loss', 'Train DFL Loss', 'Val Box Loss', 'Val Class Loss', 'Val DFL Loss'],
        custom_labels,
        layout=(2, 3)
    )

    plot_PR()
    plot_F1()
