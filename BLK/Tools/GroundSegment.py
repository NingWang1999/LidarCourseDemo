def PlaneSegmentation(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,ransac_n=5,num_iterations=10000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    print("Set ground (gray) and aboveground (red)")
    inlier_cloud = pcd.select_by_index(inliers)
    print('----inlier_cloud: ', inlier_cloud.points)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    print('----outlier_cloud: ', outlier_cloud.points)
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    return outlier_cloud,inlier_cloud

def FoundZmin(xyz,pillar_size):
    import numpy as np
    print('Find the lowest point in pillars')
    rows = [x for x in np.arange(np.min(xyz[:, 0]), np.max(xyz[:, 0]), pillar_size)]
    cols = [y for y in np.arange(np.min(xyz[:, 1]), np.max(xyz[:, 1]), pillar_size)]
    ground = []

    for i,r in enumerate(rows):
        for j,c in enumerate(cols):
            index = (xyz[:, 0] >= r) & (xyz[:, 0] < r + pillar_size) & (xyz[:, 1] >= c) & (xyz[:, 1] <= c + pillar_size)
            pillar = xyz[index]
            try:
                index = pillar[:, 2] == np.min(pillar[:, 2])
                downpoint = pillar[index]
                ground.extend(downpoint)
            except:
                continue


    return np.array(ground)

def PoissonSurfaceFilter(pcd,pillar_size,threshold=0.3):
    import numpy as np
    import open3d as o3d
    xyz = np.asarray(pcd.points)

    ground = FoundZmin(xyz,pillar_size)
    new_z = ground[:,2]
    index_z = (new_z <= np.mean(new_z) + 2 * np.std(new_z)) & (new_z >= np.mean(new_z) - 2 * np.std(new_z))
    groundsurface = o3d.geometry.PointCloud()
    groundsurface.points = o3d.utility.Vector3dVector(ground[index_z])
    groundsurface.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([groundsurface])

    # ------------------------- Ball pivoting --------------------------
    print('run Poisson surface reconstruction')
    radius = 0.001
    max_nn = 30
    groundsurface.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))  # 执行法线估计

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(groundsurface, depth=12)

    ground = mesh.sample_points_poisson_disk(number_of_points=10000, init_factor=10)
    ground.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([ground])

    dist = np.asarray(pcd.compute_point_cloud_distance(ground))
    idx = np.where(np.abs(dist)>threshold)[0]
    inlier_cloud = pcd.select_by_index(idx)
    inlier_dist = np.array(np.abs(dist)[idx])

    colors = np.zeros([inlier_dist.shape[0], 3])
    height_max = np.max(inlier_dist)
    height_min = np.min(inlier_dist)
    delta_c = abs(height_max - height_min) / (255 * 2)
    color_n = (inlier_dist-height_min) / delta_c
    color1 = np.vstack(([0]*len(color_n[color_n<=255]),1-color_n[color_n<=255]/255,[1]*len(color_n[color_n<=255])))
    color2 = np.vstack(((color_n[color_n>255]-255)/255,[0]*len(color_n[color_n>255]),[1]*len(color_n[color_n>255])))
    colors[color_n<=255,:] = color1.T
    colors[color_n>255,:] = color2.T

    colors = colors
    inlier_cloud.colors = o3d.utility.Vector3dVector(colors)

    print('----inlier_cloud: ', inlier_cloud.points)
    outlier_cloud = pcd.select_by_index(idx, invert=True)
    outlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    print('----outlier_cloud: ', outlier_cloud.points)


    return outlier_cloud,inlier_cloud




