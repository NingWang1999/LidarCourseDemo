
def ReadPointCloud(datapth):
    """
    :param datapth: 点云数据路径
    :return:
    """
    import laspy
    pcd = laspy.read(datapth)
    
    print('Read point cloud:', datapth)
    return pcd

def laspy2o3d(las):
    """
    :param las: laspy读取的点云数据
    :return:
    """
    import open3d as o3d
    import numpy as np

    x = np.float64(las.x)
    y = np.float64(las.y)
    z = np.float64(las.z)
    x = x - 440303
    y = y - 4427159

    point_data = np.stack([x, y, z], axis=0).transpose((1, 0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_data)
    return pcd

def ShowwithColors(pcd):
    """
    :param pcd: 点云文件(Open3D)
    :return:
    """
    # coding:utf-8
    import numpy as np
    import open3d as o3d

    pts = np.asarray(pcd.points)

    # 根据高度生成色彩
    colors = np.zeros([pts.shape[0], 3])
    height_max = np.max(pts[:, 2])
    height_min = np.min(pts[:, 2])
    delta_c = abs(height_max - height_min) / (255 * 2)
    for j in range(pts.shape[0]):
        color_n = (pts[j, 2] - height_min) / delta_c
        if color_n <= 255:
            colors[j, :] = [0, 1 - color_n / 255, 1]
        else:
            colors[j, :] = [(color_n - 255) / 255, 0, 1]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# datapth = r'E:/Lidar/2025/1112/Data/TLS_Clip/TLS_FARO_Clip_Resample01.las'
# spth = r'E:/Lidar/2025/1112/Output/FARO'

# ##基于laspy读取点云,并转成open3D格式
# las = ReadPointCloud(datapth)
# pcd = laspy2o3d(las)