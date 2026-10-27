def statistical_outlier_removal(pcd,neighbors=20,ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=ratio)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    print("Set outliers (red) and inliers (gray)")

    outlier_cloud.paint_uniform_color([1, 0, 0])
    print('----outlier_cloud: ', outlier_cloud.points)
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    print('----inlier_cloud: ', inlier_cloud.points)
    return outlier_cloud,inlier_cloud

def radius_outlier_removal(pcd, nb_points=16, radius=0.05):
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    print("Set outliers (red) and inliers (gray)")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    print('----outlier_cloud: ', outlier_cloud.points)
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    print('----inlier_cloud: ', inlier_cloud.points)
    return outlier_cloud,inlier_cloud
