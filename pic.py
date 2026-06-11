import matplotlib.pyplot as plt

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，可根据系统情况更换
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
value = ['Con-YOLO', 'YOLOv11','Tesseract-OCR']
# 转换 score 为浮点数
score = [float('96.8'), float('92'), float('83.2')]

# 创建横向条形图
fig, ax = plt.subplots()
bars = ax.barh(value, score)

# 在条形图上显示具体数值，并设置字体大小
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2,
            f'{width}%', ha='left', va='center', fontsize=12)

# 添加标题和标签，并设置字体大小
plt.title('模型对比', fontsize=18)
plt.xlabel('平均精确率', fontsize=14)
plt.ylabel('模型', fontsize=14)

# 设置纵轴标签字体大小
ax.tick_params(axis='y', labelsize=14)

# 显示图形
plt.show()
    