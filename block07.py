import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 模擬資料
fps_data = {
    "Model Name": ["YOLOv8s+CSPliteshufflenetV2", "YOLOv11s+CSPliteshufflenetV2", "YOLOv8s", "YOLOv11s", "YOLOv10s","Ours","YOLOv9s","YOLOv9s+CSPliteshufflenetV2"],
    "Jetson(Orin NX)": [30.21, 26.15, 18.14, 16.77, 13.09,42.07,12.54,25.6],
    "Laptop(NVIDIA GeForce GTX 1650 Ti)": [118.59, 104.57, 69.59, 63.84,60.47,212.32,60.98,137.73],
    "Desktop(NVIDIA GeForce RTX 2080 Ti)": [368.78, 300.30, 228.68, 194.52,169.77,499.66,163.73,378.26],
    "Desktop(NVIDIA GeForce RTX 4090 )": [702.37, 574.38, 427.94, 389.91, 380.21,770.18,362.98,519.92]   
    }


df = pd.DataFrame(fps_data)
devices = list(fps_data.keys())[1:]
model_names = df["Model Name"].tolist()
num_models = len(model_names)

# 選擇要根據哪個裝置的 FPS 來排序 legend（例如 RTX 4090 ）
device_for_legend = "Desktop(NVIDIA GeForce RTX 4090 )"

# 根據指定裝置 FPS 排序 model_names（由高到底）
sorted_models = df.sort_values(by=device_for_legend, ascending=False)["Model Name"].tolist()

# 色彩映射
cmap = cm.get_cmap('tab10', num_models)
colors = [cmap(i) for i in range(num_models)]
color_map = {name: colors[i] for i, name in enumerate(model_names)}

# 橫條圖設定
bar_height = 0.1
fig, ax = plt.subplots(figsize=(12, 7))

for device_idx, device in enumerate(devices):
    y_base = device_idx
    sorted_df = df[["Model Name", device]].sort_values(by=device, ascending=True).reset_index(drop=True)
    for model_idx, row in sorted_df.iterrows():
        model_name = row["Model Name"]
        fps_val = row[device]
        y_pos = y_base + (model_idx - num_models / 2) * bar_height + bar_height / 2
        ax.barh(y_pos, fps_val, height=bar_height, color=color_map[model_name], label=model_name if device_idx == 0 else "")
        ax.text(fps_val + 1, y_pos, f"{fps_val:.1f}", va='center', fontsize=8)

ax.set_yticks(range(len(devices)))
ax.set_yticklabels([d.replace("(", "\n(").replace(")", ")\n") for d in devices], fontsize=14)
ax.set_xlabel("FPS")
ax.set_title("FPS Comparison per Model Across Devices", fontsize=20)

# 避免 legend 重複，並依 FPS 值排序
handles, labels = ax.get_legend_handles_labels()
label_handle_map = dict(zip(labels, handles))

# 按照 sorted_models 排序 legend
sorted_handles = [label_handle_map[label] for label in sorted_models if label in label_handle_map]
sorted_labels = [label for label in sorted_models if label in label_handle_map]

# 設定 legend，順序為從高到低
ax.legend(sorted_handles, sorted_labels, title="Model", bbox_to_anchor=(1.0, 0), loc="lower right", fontsize=12, title_fontsize=14)

plt.tight_layout()
plt.show()
