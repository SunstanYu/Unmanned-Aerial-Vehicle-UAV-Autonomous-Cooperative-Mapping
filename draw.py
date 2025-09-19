import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.array([[0.56, 0.56, 0.39],[0.53, 0.53, 0.47],[0.59, 0.53, 0.47],[0.56, 0.53, 0.51],[0.35,0.53,0.59],[0.53,0.56,0.53],[0.53,0.56,0.39]])  # 5组数据，每组3个数据值

# 设置图表
fig, ax = plt.subplots()
index = np.arange(7)  # 每组数据的索引，共有5组数据
bar_width = 0.1  # 柱子宽度

# 绘制柱状图
for i in range(3):  # 遍历每个数据值
    ax.bar(index + i * bar_width, data[:, i], bar_width, label='Data {}'.format(i+1))

# 添加标签和标题
ax.set_xlabel('Images')
ax.set_ylabel('Overlap')
ax.set_title('Overlap threshold of different image with different transmission and rotation')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(['Image {}'.format(i+1) for i in range(7)])

# 在每个柱子上添加数据值
for i in range(7):  # 遍历每组数据
    for j in range(3):  # 遍历每个数据值
        ax.text(index[i] + j * bar_width, data[i, j], '{:.2f}'.format(data[i, j]), ha='center', va='bottom')

# 显示图表
plt.tight_layout()
plt.show()