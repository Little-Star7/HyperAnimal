# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
# # 创建画布
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
#
# # 关闭坐标面板和网格
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.grid(False)
#
# # 仅绘制经纬线球体框架
# u = np.linspace(0, 2 * np.pi, 100)
# v = np.linspace(0, np.pi, 100)
#
# # 绘制经线（纵向）
# for theta in np.linspace(0, 2 * np.pi, 24):  # 增加经线密度
#     x = np.cos(theta) * np.sin(v)
#     y = np.sin(theta) * np.sin(v)
#     ax.plot(x, y, np.cos(v), color='#2F4F4F', lw=1.2, alpha=0.7)  # 石板灰色
#
# # 绘制纬线（横向）
# for phi in np.linspace(0, np.pi, 24):  # 增加纬线密度
#     x = np.cos(u) * np.sin(phi)
#     y = np.sin(u) * np.sin(phi)
#     z = np.cos(phi) * np.ones_like(u)
#     ax.plot(x, y, z, color='#2F4F4F', lw=1.2, alpha=0.7)
#
# # 生成簇数据（改进生成方式）
# np.random.seed(42)
# clusters = []
# for _ in range(5):
#     # 均匀分布初始点
#     theta = np.random.uniform(0, 2 * np.pi)
#     phi = np.arccos(2 * np.random.random() - 1)
#     center = np.array([
#         np.sin(phi) * np.cos(theta),
#         np.sin(phi) * np.sin(theta),
#         np.cos(phi)
#     ])
#
#     # von Mises-Fisher分布生成簇
#     kappa = 15  # 集中度参数
#     points = np.random.randn(100, 3)
#     points = points / np.linalg.norm(points, axis=1)[:, None]
#     points = center + 0.2 * points
#     points = points / np.linalg.norm(points, axis=1)[:, None]
#     clusters.append(points)
#
# # 绘制数据点
# colors = ['#E74C3C', '#2980B9', '#27AE60', '#F39C12', '#8E44AD']  # 现代配色
# for i, cluster in enumerate(clusters):
#     ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
#                c=colors[i], s=40, edgecolor='w', linewidth=0.1,
#                depthshade=False, alpha=0.8)
#
# # 绘制球心
# ax.scatter([0], [0], [0], c='k', s=100, edgecolor='w', linewidth=0.5, depthshade=False)  # 黑色球心
#
# # 连接球心与特征点
# for cluster in clusters:
#     for point in cluster:
#         ax.plot([0, point[0]], [0, point[1]], [0, point[2]], color='gray', lw=0.5, alpha=0.5)
#
# # 设置观察角度和比例
# ax.view_init(elev=45, azim=75)
# ax.set_box_aspect([1, 1, 1])  # 保持各轴比例一致
#
# # 隐藏坐标轴刻度
# # ax.set_xticks([])
# # ax.set_yticks([])
# # ax.set_zticks([])
#
# # 设置坐标轴物理范围
# ax.set_xlim(-1.0, 1.0)  # 5%边界缓冲
# ax.set_ylim(-1.0, 1.0)
# ax.set_zlim(-1.0, 1.0)
#
# # 保持各向同性缩放（关键修正）
# ax.set_box_aspect([1,1,1], zoom=0.95)  # zoom参数微调显示比例
#
# plt.tight_layout()
#
# # 保存图像
# # plt.savefig('sphere_with_center.png', format='png', dpi=600)
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================== 核心修改部分 ==================
# 自定义簇的中心坐标 (单位向量)
custom_centers = np.array([
    [-0.2, 0, 0.98],  # 北极点
    [-0.98, -0.2, 0],  # 赤道x轴正方向
    [0, 0.95, 0.3],  # 赤道y轴正方向
    [0.866, 0.3, 0.4],  # 南极点
    [0.707, 0.707, 0]  # 东北方向45度 (需满足 x² + y² + z² = 1)
])
# ================================================

# 创建画布
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 关闭坐标面板和网格
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# 绘制经纬线球体框架
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# 绘制经线（纵向）
for theta in np.linspace(0, 2 * np.pi, 24):
    x = np.cos(theta) * np.sin(v)
    y = np.sin(theta) * np.sin(v)
    ax.plot(x, y, np.cos(v), color='#2F4F4F', lw=1.2, alpha=0.7)

# 绘制纬线（横向）
for phi in np.linspace(0, np.pi, 24):
    x = np.cos(u) * np.sin(phi)
    y = np.sin(u) * np.sin(phi)
    z = np.cos(phi) * np.ones_like(u)
    ax.plot(x, y, z, color='#2F4F4F', lw=1.2, alpha=0.7)

# 生成簇数据（使用自定义中心）
np.random.seed(42)
clusters = []
for center in custom_centers:
    # 确保中心点归一化（单位向量）
    center = center / np.linalg.norm(center)

    # von Mises-Fisher分布生成簇
    points = np.random.randn(100, 3)
    points = points / np.linalg.norm(points, axis=1)[:, None]
    points = center + 0.3 * points
    points = points / np.linalg.norm(points, axis=1)[:, None]
    clusters.append(points)

# 绘制数据点
colors = ['#E74C3C', '#2980B9', '#27AE60', '#F39C12', '#8E44AD']
for i, cluster in enumerate(clusters):
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
               c=colors[i], s=40, edgecolor='w', linewidth=0.05,
               depthshade=False, alpha=1)

# 绘制球心
ax.scatter([0], [0], [0], c='k', s=26, edgecolor='w', linewidth=0.5, alpha=0.7)

# 连接球心与特征点
for cluster in clusters:
    for point in cluster:
        ax.plot([0, point[0]], [0, point[1]], [0, point[2]],
                color='gray', lw=0.5, alpha=0.3)

# 设置观察角度和比例
ax.view_init(elev=45, azim=75)
ax.set_box_aspect([1, 1, 1])
#
# 设置坐标轴物理范围
ax.set_xlim(-1.0, 1.0)  # 5%边界缓冲
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

# 保持各向同性缩放（关键修正）
ax.set_box_aspect([1,1,1], zoom=0.95)  # zoom参数微调显示比例

plt.tight_layout()

plt.savefig('sphere.png', format='png', dpi=600)

plt.show()