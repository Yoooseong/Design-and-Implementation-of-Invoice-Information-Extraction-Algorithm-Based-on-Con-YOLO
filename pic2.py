import matplotlib.pyplot as plt

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，可根据系统情况更换
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
models = ['Con-YOLO', 'YOLOv11','Tesseract-OCR']
# 转换 score 为浮点数
scores = [float('96.8'), float('92'),float('83.2')]

# 创建柱状图
plt.bar(models, scores)

# 在柱状图上显示具体数值，并设置字体大小
for i, score in enumerate(scores):
    plt.text(models[i], score, f'{score}%', ha='center', va='bottom', fontsize=12)

# 添加标题和标签，并设置字体大小
plt.title('模型对比', fontsize=18)
plt.xlabel('模型', fontsize=14)
plt.ylabel('平均精确率', fontsize=14)

# 设置坐标轴标签字体大小
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# 显示图形
plt.show()
    