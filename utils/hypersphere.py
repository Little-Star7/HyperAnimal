import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建画布
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 关闭坐标面板和网格
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.grid(False)

# 仅绘制经纬线球体框架
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# 绘制经线（纵向）
for theta in np.linspace(0, 2 * np.pi, 24):  # 增加经线密度
    x = np.cos(theta) * np.sin(v)
    y = np.sin(theta) * np.sin(v)
    ax.plot(x, y, np.cos(v), color='#2F4F4F', lw=1.2, alpha=0.7)  # 石板灰色

# 绘制纬线（横向）
for phi in np.linspace(0, np.pi, 24):  # 增加纬线密度
    x = np.cos(u) * np.sin(phi)
    y = np.sin(u) * np.sin(phi)
    z = np.cos(phi) * np.ones_like(u)
    ax.plot(x, y, z, color='#2F4F4F', lw=1.2, alpha=0.7)

# 生成簇数据（改进生成方式）
np.random.seed(42)
clusters = []
for _ in range(5):
    # 均匀分布初始点
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(2 * np.random.random() - 1)
    center = np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

    # von Mises-Fisher分布生成簇
    kappa = 15  # 集中度参数
    points = np.random.randn(100, 3)
    points = points / np.linalg.norm(points, axis=1)[:, None]
    points = center + 0.2 * points
    points = points / np.linalg.norm(points, axis=1)[:, None]
    clusters.append(points)

# 绘制数据点
colors = ['#E74C3C', '#2980B9', '#27AE60', '#F39C12', '#8E44AD']  # 现代配色
for i, cluster in enumerate(clusters):
    ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2],
               c=colors[i], s=40, edgecolor='w', linewidth=0.01,
               depthshade=False)

# 设置观察角度和比例
ax.view_init(elev=45, azim=60)
ax.set_box_aspect([1, 1, 1])  # 保持各轴比例一致

# 隐藏坐标轴刻度
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# 在plt.show()之前添加以下配置
# -------------------------------
# # 设置坐标轴物理范围
ax.set_xlim(-1.0, 1.0)  # 5%边界缓冲
ax.set_ylim(-1.0, 1.0)
ax.set_zlim(-1.0, 1.0)

# 保持各向同性缩放（关键修正）
ax.set_box_aspect([1,1,1], zoom=0.95)  # zoom参数微调显示比例

# 添加不可见边界点（解决边缘裁剪问题）
# ax.scatter([-1.0,1.0], [0,0], [0,0], alpha=0)
# ax.scatter([0,0], [-1.0,1.0], [0,0], alpha=0)
# ax.scatter([0,0], [0,0], [-1.0,1.0], alpha=0)

plt.tight_layout()

# plt.savefig('sphere.png', format='png', dpi=600)

plt.show()