import numpy as np
import matplotlib.pyplot as plt

# 创建矩阵
matrix = np.array([[61, 54, 51, 57, 59, 56, 54, 56],
                   [55, 49, 47, 52, 54, 51, 50, 53],
                   [59, 55, 54, 57, 57, 55, 57, 61],
                   [53, 51, 50, 52, 51, 50, 53, 57],
                   [49, 48, 48, 49, 48, 48, 50, 53],
                   [55, 54, 54, 54, 55, 55, 56, 56],
                   [52, 51, 50, 51, 53, 54, 52, 49],
                   [52, 52, 51, 52, 54, 55, 52, 48]])

fig, ax = plt.subplots(figsize=(6, 6))  # 设置画布大小

# 关闭背景颜色（纯白）
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# 画网格线
ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)  # 细黑线

# 在每个格子中间添加文本
for i in range(matrix.shape[0]):  # 遍历行
    for j in range(matrix.shape[1]):  # 遍历列
        ax.text(j, i, str(matrix[i, j]), va='center', ha='center', color='black', fontsize=12)

# 隐藏坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 显示图片
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 定义原始矩阵（数据完全一致）
matrix = np.array([
    [423.87500000, -3.05714204,  2.89533080,  4.76753030,  7.12500000,  0.34428324, -0.52279016, -0.49346373],
    [  7.80026299, -0.20873931,  3.73357038,  0.07887329,  7.92355626,  0.01122378, -0.10594294, -0.35991115],
    [  4.01618462,  3.72883026, -5.18826208,  7.21472365,  0.06764951, -0.39557320,  0.01516504, -0.26924379],
    [ -0.39317473,  0.32887329, -0.13641188,  0.75735870,  0.10498963,  0.03437910,  0.20556591,  0.16555292],
    [ -5.37500000,  0.32226306,  0.06764951, -0.68397488, -0.12500000,  0.03046894, -0.16332037, -0.60406665],
    [  7.39558538,  0.26122378,  0.49812371,  0.38793250, -0.13177163,  0.19974808, -0.26945213, -0.09790341],
    [ 15.24882000,  0.03174579,  0.51516504, -0.74933411, -0.16332037,  0.15598414, -0.06173792,  0.32058811],
    [ -0.60755894, -0.00635776, -0.08994671, -0.08444708, -0.03700875,  0.15209659,  0.18585117, -0.24836747]
])

fig, ax = plt.subplots(figsize=(6, 6))  # 设置画布大小

# 关闭背景颜色（纯白）
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# 画网格线
ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)  # 黑色网格线

# 在每个格子中间添加文本（不使用科学计数法，保留 8 位小数）
for i in range(matrix.shape[0]):  # 遍历行
    for j in range(matrix.shape[1]):  # 遍历列
        ax.text(j, i, f"{matrix[i, j]:.8f}", va='center', ha='center', color='black', fontsize=10)

# 设置坐标轴范围
ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
ax.set_ylim(matrix.shape[0] - 0.5, -0.5)

# 隐藏坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 显示图片
plt.show()
