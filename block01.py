import pandas as pd
import matplotlib.pyplot as plt

# 使用預設英文字體就好，不需要特別設置中文字體

# 模擬數據（英文版本）
fps_data = {
    "Model Name": ["Ours", "YOLOv8s", "YOLOv8s+Swin", "YOLOv8s+GEShuffleNetV2", "YOLOv8s+ShuffleNetV2", "YOLOv8s+Ghostnet"],
    "Jetson(Orin NX)": [31.14, 19.24, 14.34, 27.57, 28.78, 18.27],
    "Laptop(NVIDIA GeForce GTX 1650 Ti)": [95.86, 60.99, 53.17, 90.28, 92.67, 68.59],
    "Desktop(NVIDIA GeForce RTX 2080 Ti)": [206.61, 153.42, 139.74, 186.45, 204.04, 150.87]
}

df = pd.DataFrame(fps_data)

def plot_fps_table(dataframe):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    table = ax.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.4)
    # 將 "Ours" 那一行的數值變紅色（但不包括模型名稱那一欄）
    for i in range(len(dataframe)):
        if dataframe.iloc[i, 0] == "Ours":
            for j in range(1, len(dataframe.columns)):  # 從第1欄開始跳過"Model Name"
                table[(i + 1, j)].get_text().set_color('red')
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    plt.title("FPS Comparison Across Devices(fps)", fontsize=30,pad=10)
    plt.tight_layout()
    plt.show()

plot_fps_table(df)
