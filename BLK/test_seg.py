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
### add this two lines to avoid argparse error in jupyternotebook
import sys
sys.argv = ['']

my_spth = r"D:\Desktop\LiDAR_course\1119\help\lxh"
pcd = MergePoint(groundpth=os.path.join(my_spth,'ground.pcd'),leafwoodpth=os.path.join(my_spth,'leaf_wood_results.txt'))
TLSSeg(pcd,my_spth)


##点云可视化
# o3d.visualization.draw_geometries([ground, aboveground])
# o3d.io.write_point_cloud(spth, data)

