def WatershedSeg(pcd):
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage.filters import gaussian_filter
    import numpy as np
    from scipy import spatial
    import pandas as pd
    import math
    from matplotlib import pyplot as plt

    # 提取边界
    bound = pcd.get_axis_aligned_bounding_box()
    x_max, y_max = bound.get_max_bound()[0:2]
    x_min, y_min = bound.get_min_bound()[0:2]
    # 分辨率
    pixelWidth, pixelHeight = 0.3, -0.3
    ### calculate
    xOrigin = x_min
    yOrigin = y_max

    cols = int(math.ceil(((x_max - x_min) / pixelWidth)))
    rows = int(math.ceil(((y_max - y_min) / abs(pixelHeight))))
    xyz = np.asarray(pcd.points)
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]

    xOffset = (x - xOrigin) / pixelWidth
    xOffset = xOffset.astype(int)
    yOffset = (y - yOrigin) / pixelHeight
    yOffset = yOffset.astype(int)
    print(cols, rows)
    x_index, y_index = np.meshgrid(np.arange(rows), np.arange(cols))
    fill_value = np.full_like(x_index.flatten(), np.nan, dtype='double')
    filled_points = np.column_stack([y_index.flatten(), x_index.flatten(), fill_value])
    # for keep the exent consisited with las data
    z_update = z
    real_points = np.column_stack([xOffset, yOffset, z_update])
    points = np.row_stack([real_points, filled_points])
    pts = pd.DataFrame(points, columns=['Off_x', 'Off_y', 'z'])
    min_raster = np.asarray(pts.pivot_table(index='Off_x', columns='Off_y', values='z', aggfunc='min', dropna=False))

    # calculate the grid center distance
    # x_index,y_index=np.meshgrid(np.arange(rows),np.arange(cols))
    x_index = x_index.flatten() * pixelWidth + xOrigin
    y_index = y_index.flatten() * pixelHeight + yOrigin

    z_min_index = min_raster.flatten()
    z_min_tmp = z_min_index
    # 封装x,y
    griddata = np.array(list(zip(x_index, y_index)))
    tree = spatial.cKDTree(griddata)
    # 领域点数
    K = 200
    # 查找空值点
    nan_index = np.where(np.isnan(z_min_tmp))
    dist, index = tree.query(griddata[nan_index], K)
    ###去除查询的网格中心点
    dist = dist[:, 1:K - 1]
    index = index[:, 1:K - 1]
    # 计算权重
    w = 1.0 / dist ** 2
    # 筛除NAN值
    w_nan_index = np.where(np.isnan(z_min_tmp[index]))
    w[w_nan_index] = 0
    z_idw = np.nansum(w * z_min_tmp[index], axis=1) / np.sum(w, axis=1)
    z_min_tmp[nan_index] = z_idw

    max_raster = np.asarray(pts.pivot_table(index='Off_x', columns='Off_y', values='z', aggfunc='max', dropna=False))
    z_max_index = max_raster.flatten()
    z_max_tmp = z_max_index
    # 封装x,y
    griddata = np.array(list(zip(x_index, y_index)))
    tree = spatial.cKDTree(griddata)
    # 领域点数
    K = 200
    # 查找空值点
    nan_index = np.where(np.isnan(z_max_tmp))
    dist, index = tree.query(griddata[nan_index], K)
    ###去除查询的网格中心点
    dist = dist[:, 1:K - 1]
    index = index[:, 1:K - 1]
    # 计算权重
    w = 1.0 / dist ** 2
    # 筛除NAN值
    w_nan_index = np.where(np.isnan(z_max_tmp[index]))
    w[w_nan_index] = 0
    z_idw = np.nansum(w * z_max_tmp[index], axis=1) / np.sum(w, axis=1)
    z_max_tmp[nan_index] = z_idw

    # CHM
    chm_raster = z_max_tmp.reshape(cols, rows) - z_min_tmp.reshape(cols, rows)

    dataraster_gau = gaussian_filter(chm_raster, sigma=1)
    # 寻找CHM中的局部最大值作为分水岭的标记点
    local_maxi = peak_local_max(dataraster_gau, indices=False)
    markers = ndi.label(local_maxi)[0]
    # 利用分水岭算法进行分割，labels变量中存储了每个分割结果
    labels = watershed(-dataraster_gau, markers, mask=dataraster_gau[:] > 3)
    # 利用matplotlib出图查看初步结果
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    # 绘制CHM
    ax[0].imshow(-chm_raster)
    ax[0].set_title('CHM')
    # 绘制高斯滤波后的CHM
    ax[1].imshow(dataraster_gau)
    ax[1].set_title('chm_gs')
    # 绘制分割结果
    ax[2].imshow(labels)
    ax[2].set_title('Segment')
    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()





