# -*- coding: utf-8 -*-


from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 各省经纬度 https://blog.csdn.net/weixin_42060598/article/details/129876634
# 各城市经纬度 https://blog.csdn.net/wjm901215/article/details/83800447
# 定义中国部分城市的坐标
cities = {
    '北京': (39.9042, 116.4074),
    '上海': (31.2304, 121.4737),
    '广州': (23.1291, 113.2644),
    '深圳': (22.5431, 114.0579),
    '成都': (30.5728, 104.0060),
}

# 将坐标转换为 numpy 数组
city_names = list(cities.keys())
coords = np.array(list(cities.values()))

# 使用距离矩阵求解 TSP
dist_matrix = distance_matrix(coords, coords)
row_ind, col_ind = linear_sum_assignment(dist_matrix)

# 创建 TSP 游览路径
tsp_path = list(row_ind)

# 使用 Cartopy 绘制
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# 添加地图功能
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)

# 绘制城市
for city, (lat, lon) in cities.items():
    ax.plot(lon, lat, 'o', color='red', transform=ccrs.PlateCarree())
    ax.text(lon, lat, city, fontsize=12, ha='right', transform=ccrs.PlateCarree())

# 绘制 TSP 路径
path_coords = np.array([coords[i] for i in tsp_path] + [coords[tsp_path[0]]])  # Include return to start
path_lon_lat = path_coords[:, [1, 0]]  # Swap for plotting
ax.plot(path_lon_lat[:, 0], path_lon_lat[:, 1], 'gray', lw=2, transform=ccrs.PlateCarree())

plt.title('中国地图上的 TSP 路径')
plt.show()
