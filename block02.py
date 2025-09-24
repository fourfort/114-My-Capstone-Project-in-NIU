import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 模擬資料
fps_data = {
    "Model Name": ["Ours", "YOLOv8s", "YOLOv8s+Swin",  "YOLOv8s+Ghostnet"],
    "Jetson(Orin NX, 15W)": [31.14, 19.24, 14.34, 18.27],
    "Laptop(NVIDIA GeForce GTX 1650 Ti)": [95.86, 60.99, 53.17, 68.59],
    "Desktop(NVIDIA GeForce RTX 2080 Ti)": [206.61, 153.42, 139.74,150.87]
}

df = pd.DataFrame(fps_data)
devices = ["Jetson(Orin NX, 15W)", "Laptop(NVIDIA GeForce GTX 1650 Ti)", "Desktop(NVIDIA GeForce RTX 2080 Ti)"]
model_names = df["Model Name"].tolist()
num_models = len(model_names)

# 顏色設定
colors = plt.colormaps.get_cmap('tab10').resampled(num_models)
color_map = {name: colors(i) for i, name in enumerate(model_names)}

# 橫條圖參數
bar_height = 0.12
fig, ax = plt.subplots(figsize=(12, 6))

# 繪製每個裝置的 group
for device_idx, device in enumerate(devices):
    y_base = device_idx  # Y 軸位置（每一個裝置一組）
    sorted_df = df[["Model Name", device]].sort_values(by=device, ascending=True).reset_index(drop=False)  # 排序
    for model_idx, row in sorted_df.iterrows():  # 使用排序後的 df
        model_name = row["Model Name"]
        fps_val = row[device]
        y_pos = y_base + (model_idx - num_models / 2) * bar_height + bar_height / 2
        bar = ax.barh(y_pos, fps_val, height=bar_height, color=colors(model_idx), label=model_name if device_idx == 0 else "")
        ax.text(fps_val + 1, y_pos, f"{fps_val:.1f}", va='center', fontsize=8)

# 設定 Y 軸為設備名稱，並讓設備名稱顯示為多行
device_labels = [device.replace("(", "\n(").replace(")", ")\n") for device in devices]
ax.set_yticks(range(len(devices)))
ax.set_yticklabels(device_labels,fontsize=14)

# 設置標籤、標題等
ax.set_xlabel("FPS")
ax.set_title("FPS Comparison per Model Across Devices", fontsize=24, pad=8)

# 調整 legend，放大字體
ax.legend(title="Model", bbox_to_anchor=(1, 0), loc="lower right", fontsize=12, title_fontsize=14)

plt.tight_layout()
plt.show()
