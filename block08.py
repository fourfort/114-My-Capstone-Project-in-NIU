import pandas as pd
import matplotlib.pyplot as plt

# 設定全域字體為微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 模擬數據
fps_data = {
    "Name": ["Ours", "YOLOv8s", "YOLOv9s+CSPliteshufflenetV2", "YOLOv9s", "YOLOv10s+CSPliteshufflenetV2", "YOLOv10s", "YOLOv11s+CSPliteshufflenetV2", "YOLOv11s"],
    "layers": [284, 225, 972, 917, 474, 402, 406, 295],
    "parameters": [6423609, 11136761, 7943777, 7288569, 7048314, 8068674, 6701385,10725593],
    "GFLOPS": [16.417, 28.651, 6.164, 27.39, 4.24,24.78,15.5,26.63],
    "epoch": [198, 186, 358, 201, 492, 469,221,314]
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

    for (row, col), cell in table.get_celld().items():
        if row == 0:  # 欄位標題
            cell.set_text_props(weight='bold', fontsize=20)
        elif col == 0:  # 模型名稱欄
            cell.set_text_props(weight='bold', fontsize=14)
        else:  # 其他數值
            cell.set_text_props(weight='bold', fontsize=20)

    table.scale(1.2, 1.4)

    # 將 "Ours" 那一行的數值變紅色（不包括模型名稱欄）
    for i in range(len(dataframe)):
        if dataframe.iloc[i, 0] == "Ours":
            for j in range(1, len(dataframe.columns)):
                table[(i + 1, j)].get_text().set_color('red')

    plt.tight_layout()
    plt.show()

plot_fps_table(df)
