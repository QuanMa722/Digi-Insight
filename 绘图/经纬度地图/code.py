# -*- coding: utf-8 -*-
# 各省经纬度 https://blog.csdn.net/weixin_42060598/article/details/129876634
# 各城市经纬度 https://blog.csdn.net/wjm901215/article/details/83800447

import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 城市及其坐标
cities = {
    '上海': (31.231706, 121.472644),
    '北京': (39.904989, 116.405285),
    '广州': (23.125178, 113.280637),
    '成都': (30.659462, 104.065735),
    '南京': (32.041544, 118.767413),
    '济南': (36.675807, 117.000923),
    '重庆': (29.533155, 106.504962),
    '西安': (34.263161, 108.948024),
    '福州': (26.075302, 119.306239),
    '厦门': (24.490474, 118.11022),
    '丽水': (28.451993, 119.921786),
    '武汉': (30.584355, 114.298572),
    '抚州': (27.98385, 116.358351),
    '杭州': (30.287459, 120.153576),
    '韶关': (24.801322, 113.591544),
}
# 获取城市的经纬度
lats, lons = zip(*cities.values())

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

# 添加中国地图的特征
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, edgecolor='black')

# 设置显示区域以集中在中国
ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

# 绘制城市位置
ax.scatter(lons, lats, color='red', s=20, transform=ccrs.PlateCarree())  # 使用 scatter 绘制点
for city, (lat, lon) in cities.items():
    ax.text(lon, lat, city, fontsize=12, ha='right', transform=ccrs.PlateCarree())

# 连接点的顺序
# ax.plot(lons, lats, linestyle='--', marker='o', color='red', transform=ccrs.PlateCarree())

# 显示图形
plt.show()

