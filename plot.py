# -*- coding: utf-8 -*-
"""
@Author  : Morvan Li
@FileName: plot.py
@Software: PyCharm
@Time    : 7/10/23 1:58 PM
"""

# import matplotlib.pyplot as plt
# import numpy as np
#
# # 假设有十个算法的 CPU 和 GPU 运行时间数据以及对应的标准差
# algorithm_names = ['LPSR', 'PAPCNN', 'DTNP', 'JBF-LGE', 'Proposed',
#                    'EMFusion', 'PMGI', 'U2Fusion', 'DDcGAN', 'MATR']
# cpu_times = [0.0169, 4.1596, 26.2073, 0.4195, 1.2631, 0.2369, 0.2026, 0.6028, 1.8034, 0.8246]  # 每个算法的 CPU 运行时间（以秒为单位）
# cpu_std = [0.0013, 0.0112, 0.048, 0.0019, 0.0262, 0.0294, 0.0190, 0.0454, 0.0524, 0.0073]  # 每个算法的 CPU 运行时间标准差
# gpu_times = [0, 0, 0, 0, 0, 0.0249, 0.0182, 0.1026, 0.3421, 0.2025]  # 每个算法的 GPU 运行时间（以秒为单位）
# gpu_std = [0, 0, 0, 0, 0, 0.0136, 0.0021, 0.0157, 0.2348, 0.0275]  # 每个算法的 GPU 运行时间标准差
#
# # 设置柱状图的位置
# ind = np.arange(len(algorithm_names))
# width = 0.45
#
# # 创建图表和子图
# fig, ax = plt.subplots()
#
# # 绘制 CPU 时间的柱状图
# cpu_bars = ax.bar(ind, cpu_times, width, label='CPU', edgecolor='black')
# # 绘制 GPU 时间的柱状图
# gpu_bars = ax.bar(ind + width, gpu_times, width, label='GPU', edgecolor='black')
#
# # 绘制 CPU 时间的误差棒
# ax.errorbar(ind, cpu_times, yerr=cpu_std, fmt='none', color='black', capsize=4)
# # 绘制 GPU 时间的误差棒
# ax.errorbar(ind + width, gpu_times, yerr=gpu_std, fmt='none', color='black', capsize=4)
#
# # 设置 x 轴刻度标签
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(algorithm_names, rotation=30)
#
# # 设置 y 轴为对数刻度
# ax.set_yscale('log')
#
# # 添加图例
# ax.legend()
#
# # 添加图表标题和坐标轴标签
# ax.set_title('Algorithm Execution Time on CPU and GPU')
# ax.set_xlabel('Algorithm')
# ax.set_ylabel('Execution Time (s)')
#
# # 显示图表
# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# # 假设有十个算法的 CPU 和 GPU 运行时间数据以及对应的标准差
# algorithm_names = ['LPSR', 'PAPCNN', 'DTNP', 'JBF-LGE', 'EMFusion',
#                    'PMGI', 'U2Fusion', 'DDcGAN', 'MATR', 'Proposed']
# cpu_times = [0.0169, 4.1596, 26.2073, 0.4195, 0.2369, 0.2026, 0.6028, 1.8034, 0.8246, 1.2631]  # 每个算法的 CPU 运行时间（以秒为单位）
# cpu_std = [0.0013, 0.0112, 0.048, 0.0019, 0.0262, 0.0294, 0.0190, 0.0454, 0.0524, 0.0073]  # 每个算法的 CPU 运行时间标准差
# gpu_times = [0, 0, 0, 0, 0.0249, 0.0182, 0.1026, 0.3421, 0.2025, 0]  # 每个算法的 GPU 运行时间（以秒为单位）
# gpu_std = [0, 0, 0, 0, 0, 0.0136, 0.0021, 0.0157, 0.2348, 0.0275]  # 每个算法的 GPU 运行时间标准差
#
# # 创建数据框
# df = pd.DataFrame({
#     'Algorithm': algorithm_names,
#     'CPU Time': cpu_times,
#     'GPU Time': gpu_times,
# })
#
# # 融合数据框，以便绘制分组柱状图
# df_melt = pd.melt(df, id_vars='Algorithm', value_vars=['CPU Time', 'GPU Time'], var_name='Device',
#                   value_name='Execution Time')
#
# # 使用 Seaborn 绘制分组柱状图，带误差棒
# sns.barplot(data=df_melt, x='Algorithm', y='Execution Time', hue='Device', edgecolor='black')
# plt.yscale('log')  # 设置 y 轴为对数刻度
# # 添加图表标题和坐标轴标签
# plt.title('Algorithm Execution Time on CPU and GPU')
# plt.xlabel('Algorithm')
# plt.ylabel('Execution Time (s)')
#
# # 调整 x 轴刻度标签的角度以防止重叠
# plt.xticks(rotation=30)
#
# # 显示图表
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# 假设有多个算法的性能数据，其中自己的算法优于其他算法
algorithm_names = ['Algorithm A', 'Algorithm B', 'Algorithm C', 'Algorithm D']
performance_data = {
    'Metric1': [80, 60, 70, 90],  # 各个算法在Metric1上的性能数据
    'Metric2': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric3': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric4': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric5': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric6': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric7': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric8': [75, 70, 65, 85],  # 各个算法在Metric2上的性能数据
    'Metric9': [90, 80, 85, 95]  # 各个算法在Metric9上的性能数据
}

# 创建数据框
df = pd.DataFrame(performance_data, index=algorithm_names)

# 计算相对改进的百分比
df_relative_improvement = ((df - df.iloc[0]) / df.iloc[0]) * 100

# 计算雷达图中的角度
angles = [(i / 9) * 2 * 3.14159 for i in range(9)]
angles += angles[:1]  # 闭合雷达图

# 创建子图和坐标轴
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 绘制每个算法的雷达图
for i in range(len(algorithm_names)):
    values = df_relative_improvement.iloc[i].tolist()
    values += values[:1]  # 闭合多边形
    ax.plot(angles, values, label=algorithm_names[i])
    ax.fill(angles, values, alpha=0.25)  # 填充多边形

# 设置刻度标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(df.columns)

# 添加网格线
ax.yaxis.grid(True)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# 添加标题
plt.title('Performance Comparison of Algorithms')

# 显示图表
plt.show()
