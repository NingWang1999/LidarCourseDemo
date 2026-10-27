def voxel_down_sample(pcd, voxel_size=0.01):
    """
    :param pcd: 点云数据
    :param voxel_size: 抽稀大小
    :return:
    """
    pcd_downsample = pcd.voxel_down_sample(voxel_size=voxel_size)
    print('Point Cloud Subsample:', voxel_size)
    return pcd_downsample