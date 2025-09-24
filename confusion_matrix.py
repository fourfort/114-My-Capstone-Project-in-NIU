import numpy as np
import matplotlib.pyplot as plt

# 類別名稱
classes = ['broken', 'dirty', 'normal']
classNamber = 3

# 混淆矩陣數值
confusion_matrix = np.array([
    (0.86, 0.01, 0.01),
    (0.11, 0.73, 0.16),
    (0.01, 0.15, 0.77),
], dtype=np.float64)

# 繪圖
plt.figure(figsize=(10, 8))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix', size=18)
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=0)  # ✅ 水平顯示 x 軸標籤
plt.yticks(tick_marks, classes, rotation=-270)

thresh = confusion_matrix.max() / 2.
iters = np.reshape([[[i, j] for j in range(classNamber)] for i in range(classNamber)], (confusion_matrix.size, 2))
for i, j in iters:
    value = confusion_matrix[i, j]
    color = "white" if value > 0.4 else "black"
    plt.text(j, i, f"{value:.2f}", va='center', ha='center', size=16, color=color)

plt.ylabel('Predicted', size=15)
plt.xlabel('True', size=15)
plt.tight_layout()

plt.savefig("confusion_matrix_horizontal_xticks.png", dpi=300)
plt.show()
