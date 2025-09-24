import pandas as pd
import matplotlib.pyplot as plt

# 設定全域字體為微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 模擬數據
fps_data = {
    "Name": ["Ours", "YOLOv8s", "YOLOv8s+Swin",  "YOLOv8s+Ghostnet"],
    "layers": [284, 225, 258,  529],
    "parameters": [6423609, 11136761, 11106557,  9480276],
    "GFLOPS": [16.417, 28.651, 32.098, 16.498],
    "epoch": [198, 186, 162, 588]
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
            cell.set_text_props(weight='bold', fontsize=25)
        elif col == 0:  # 模型名稱欄
            cell.set_text_props(weight='bold', fontsize=20)
        else:  # 其他數值
            cell.set_text_props(weight='bold', fontsize=25)

    table.scale(1.4, 1.6)

    # 將 "Ours" 那一行的數值變紅色（不包括模型名稱欄）
    for i in range(len(dataframe)):
        if dataframe.iloc[i, 0] == "Ours":
            for j in range(1, len(dataframe.columns)):
                table[(i + 1, j)].get_text().set_color('red')

    plt.tight_layout()
    plt.show()

plot_fps_table(df)
