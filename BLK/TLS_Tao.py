#%%
## 打开代码后，Tools文件就在同目录下，所以不用设置Tools路径
from Tools.ReadPointCloud import *
import open3d as o3d
from Tools.Treeseg import WatershedSeg
from Tools.Down_Sample import *
from Tools.GroundSegment import *
from Tools.Clean import *
from Tools.Treeseg import *
from Tools.TLS2tree import TLSSeg,MergePoint
import os

#%%
#点云数据路径
datapth = r"D:\Desktop\LiDAR_course\1119\my\FARO_register_Resample_offset_clip_trim.las"
spth = r'D:\Desktop\LiDAR_course\1119\my'

# ##基于laspy读取点云,并转成open3D格式
las = ReadPointCloud(datapth)
pcd = laspy2o3d(las)
#%%
##点云下采样
voxel_size = 0.02
pcd_down_sample = voxel_down_sample(pcd,voxel_size=voxel_size)
o3d.io.write_point_cloud(os.path.join(spth,'Down_sample.pcd'), pcd_down_sample)
#%%
#点云去噪
# pcd_radius_outlier, pcd_radius_inlier = radius_outlier_removal(pcd_down_sample,nb_points=6,radius=0.2)
pcd_statistical_outlier, pcd_statistical_inlier = statistical_outlier_removal(pcd_down_sample,neighbors=5,ratio=10.0)
o3d.io.write_point_cloud(os.path.join(spth,'pcd_radius_outlier.pcd'), pcd_statistical_outlier)
o3d.io.write_point_cloud(os.path.join(spth,'pcd_radius_inlier.pcd'), pcd_statistical_inlier)
#%%
#地面点分离， 会自动出现可视化，第一个是最低点，关闭后出现第二个试图，最终的点
pcd_statistical_inlier = o3d.io.read_point_cloud(os.path.join(spth,'pcd_radius_inlier.pcd'))
# ground, aboveground = PlaneSegmentation(pcd)
ground, aboveground = PoissonSurfaceFilter(pcd_statistical_inlier,pillar_size=1,threshold=0.3)
o3d.io.write_point_cloud(os.path.join(spth,'ground.pcd'), ground)
o3d.io.write_point_cloud(os.path.join(spth,'aboveground.pcd'), aboveground)
#%%
##基于Lewos进行枝叶分离
##单木分割

o3d.visualization.draw_geometries([ground, aboveground])
### add this two lines to avoid argparse error in jupyternotebook
import sys
sys.argv = ['']

pcd = MergePoint(groundpth=os.path.join(spth,'ground.txt'),leafwoodpth=os.path.join(spth,'leaf_wood_results.txt'))
TLSSeg(pcd,spth)


##点云可视化
#o3d.io.write_point_cloud(spth, data)

