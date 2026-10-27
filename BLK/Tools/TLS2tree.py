import datetime
start = datetime.datetime.now()
import argparse
from scipy.spatial import ConvexHull
import networkx as nx
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation
from scipy.stats import variation
import numpy as np
from matplotlib.patches import Circle
from tqdm.auto import tqdm
import warnings
import os
import shutil
import glob
import string
import itertools
import threading
import laspy
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy import ndimage
from tqdm import tqdm
import sys
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

def MergePoint(groundpth,leafwoodpth):
    print('Merge Point Cloud ....')
    import open3d as o3d
    ground = o3d.io.read_point_cloud(groundpth)
    groundwithlabel = np.zeros((len(ground.points),4),np.float32)
    groundwithlabel[:,:3] = np.asarray(ground.points)
    groundwithlabel[:,-1] = 2
    Leafwood = np.loadtxt(leafwoodpth)
    index = Leafwood[:,-1] == 1
    Leafwood[:,-1] = 1
    Leafwood[index,-1] = 3
    pc = np.vstack([Leafwood,groundwithlabel])
    return pc


def read_ply(fp, newline=None):
    line = open(fp, encoding='ISO-8859-1').readline()
    newline = '\n' if line == 'ply\n' else None

    return read_ply_(fp, newline)

def read_pcd(fp):

    if (sys.version_info > (3, 0)):
        open_file = open(fp, encoding='ISO-8859-1')
    else:
        open_file = open(fp)

    with open_file as pcd:

        length = 0

        for i, line in enumerate(pcd.readlines()):
            length += len(line)
            if 'WIDTH' in line: N = int(line.split()[1])
            if 'FIELDS' in line: F = line.split()[1:]
            if 'DATA' in line:
                fmt = line.split()[1]
                break

        if fmt == 'binary':
            pcd.seek(length)
            arr = np.fromfile(pcd, dtype='f')

            arr = arr[:N*len(F)].reshape(-1, len(F))
            df = pd.DataFrame(arr, columns=F)

    if fmt == 'ascii':
        df = pd.read_csv(fp, sep=' ', names=F, skiprows=11)

    return df

def read_ply_(fp, newline):
    open_file = open(fp,
                     encoding='ISO-8859-1',
                     newline=newline)

    with open_file as ply:

        length = 0
        prop = []
        dtype_map = {'uint16': 'uint16', 'uint8': 'uint8', 'double': 'd', 'float64': 'f8',
                     'float32': 'f4', 'float': 'f4', 'uchar': 'B', 'int': 'i'}
        dtype = []
        fmt = 'binary'

        for i, line in enumerate(ply.readlines()):
            length += len(line)
            if i == 1:
                if 'ascii' in line:
                    fmt = 'ascii'
            if 'element vertex' in line: N = int(line.split()[2])
            if 'property' in line:
                dtype.append(dtype_map[line.split()[1]])
                prop.append(line.split()[2])
            if 'element face' in line:
                raise Exception('.ply appears to be a mesh')
            if 'end_header' in line: break

        ply.seek(length)

        if fmt == 'binary':
            arr = np.fromfile(ply, dtype=','.join(dtype))
        else:
            arr = np.loadtxt(ply)
        df = pd.DataFrame(data=arr)
        df.columns = prop

    return df


def write_ply(output_name, pc, comments=[]):
    cols = ['x', 'y', 'z']
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']].astype('f8')

    with open(output_name, 'w') as ply:

        ply.write("ply\n")
        ply.write('format binary_little_endian 1.0\n')
        ply.write("comment Author: Phil Wilkes\n")
        for comment in comments:
            ply.write("comment {}\n".format(comment))
        ply.write("obj_info generated with pcd2ply.py\n")
        ply.write("element vertex {}\n".format(len(pc)))
        ply.write("property float64 x\n")
        ply.write("property float64 y\n")
        ply.write("property float64 z\n")
        if 'red' in pc.columns:
            cols += ['red', 'green', 'blue']
            pc[['red', 'green', 'blue']] = pc[['red', 'green', 'blue']].astype('i')
            ply.write("property int red\n")
            ply.write("property int green\n")
            ply.write("property int blue\n")
        for col in pc.columns:
            if col in cols: continue
            try:
                pc[col] = pc[col].astype('f8')
                ply.write("property float64 {}\n".format(col))
                cols += [col]
            except:
                pass
        ply.write("end_header\n")

    with open(output_name, 'ab') as ply:
        ply.write(pc[cols].to_records(index=False).tobytes())

class dict2class:

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def make_folder_structure(params):
    if params.odir == None:
        params.odir = os.path.join(params.directory, params.filename + '_FSCT_output')

    params.working_dir = os.path.join(params.odir, params.basename + '.tmp')

    if not os.path.isdir(params.odir):
        os.makedirs(params.odir)

    if not os.path.isdir(params.working_dir):
        os.makedirs(params.working_dir)
    else:
        shutil.rmtree(params.working_dir, ignore_errors=True)
        os.makedirs(params.working_dir)

    if params.verbose:
        print('output directory:', params.odir)
        print('scratch directory:', params.working_dir)

    return params


def voxelise(tmp, length, method='random', z=True):
    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z: tmp.loc[:, 'zz'] = tmp.z // length * length

    if method == 'random':

        code = lambda: ''.join(np.random.choice([x for x in string.ascii_letters], size=8))

        xD = {x: code() for x in tmp.xx.unique()}
        yD = {y: code() for y in tmp.yy.unique()}
        if z: zD = {z: code() for z in tmp.zz.unique()}

        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD)
        if z: tmp.VX += tmp.zz.map(zD)

    elif method == 'bytes':

        code = lambda row: np.array([row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = tmp.apply(code, axis=1)

    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')

    return tmp


def downsample(pc, vlength,
               accurate=False,
               keep_columns=[],
               keep_points=False,
               voxel_method='random',
               return_VX=False,
               verbose=False):
    """
    Downsamples a point cloud so that there is one point per voxel.
    Points are selected as the point closest to the median xyz value

    Parameters
    ----------

    pc: pd.DataFrame with x, y, z columns
    vlength: float


    Returns
    -------

    pd.DataFrame with boolean downsample column

    """

    pc = pc.drop(columns=[c for c in ['downsample', 'VX'] if c in pc.columns])
    columns = pc.columns.to_list() + keep_columns  # required for tidy up later
    if return_VX: columns += ['VX']
    pc = voxelise(pc, vlength, method=voxel_method)

    if accurate:
        # groubpy to find central (closest to median) point
        groupby = pc.groupby('VX')
        pc.loc[:, 'mx'] = groupby.x.transform(np.median)
        pc.loc[:, 'my'] = groupby.y.transform(np.median)
        pc.loc[:, 'mz'] = groupby.z.transform(np.median)
        pc.loc[:, 'dist'] = np.linalg.norm(pc[['x', 'y', 'z']].to_numpy(dtype=np.float32) -
                                           pc[['mx', 'my', 'mz']].to_numpy(dtype=np.float32), axis=1)
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.sort_values(['VX', 'dist']).duplicated('VX'), 'downsample'] = True

    else:
        pc.loc[:, 'downsample'] = False
        pc.loc[~pc.VX.duplicated(), 'downsample'] = True

    if keep_points:
        return pc[columns + ['downsample']]
    else:
        return pc.loc[pc.downsample][columns]


def compute_plot_centre(pc):
    """calculate plot centre"""
    plot_min, plot_max = pc[['x', 'y']].min(), pc[['x', 'y']].max()
    return (plot_min + ((plot_max - plot_min) / 2)).values


def compute_bbox(pc):
    bbox_min = pc.min().to_dict()
    bbox_min = {k + 'min': v for k, v in bbox_min.items()}
    bbox_max = pc.max().to_dict()
    bbox_max = {k + 'max': v for k, v in bbox_max.items()}
    return dict2class({**bbox_min, **bbox_max})


def load_file(filename, additional_headers=False, verbose=False):
    file_extension = os.path.splitext(filename)[1]
    headers = ['x', 'y', 'z']

    if file_extension == '.las' or file_extension == '.laz':

        import laspy

        inFile = laspy.read(filename)
        pc = np.vstack((inFile.x, inFile.y, inFile.z)).T
        # for header in additional_fields:
        #    if header in list(inFile.point_format.dimension_names):
        #        pc = np.vstack((pc, getattr(inFile, header)))
        #    else:
        #        headers.drop(header)
        pc = pd.DataFrame(data=pc, columns=['x', 'y', 'z'])

    elif file_extension == '.ply':
        pc = read_ply(filename)
        # pc = pc[headers]

    elif file_extension == '.pcd':
        pc = read_pcd(filename)
        # pc = pc[headers]

    else:
        raise Exception('point cloud format not recognised' + filename)

    original_num_points = len(pc)

    if verbose: print(f'read in {filename} with {len(pc)} points')

    if additional_headers:
        return pc, [c for c in pc.columns if c not in ['x', 'y', 'z']]
    else:
        return pc


def save_file(filename, pointcloud, additional_fields=[], verbose=False):
    #     if pointcloud.shape[0] == 0:
    #         print(filename, 'is empty...')
    #     else:
    if verbose:
        print('Saving file:', filename)

    cols = ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = np.min(pointcloud[:, :3], axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if len(additional_fields) != 0:
            additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            col_idxs = list(range(3, pointcloud.shape[1]))
            additional_fields.reverse()

            col_idxs.reverse()
            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud[:, i]
                if header in ['red', 'green', 'blue']:
                    setattr(las, header, column)
                else:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
        las.write(filename)
        if not verbose:
            print("Saved.")

    elif filename.endswith('.csv'):
        pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
        print("Saved to:", filename)

    elif filename.endswith('.ply'):

        if not isinstance(pointcloud, pd.DataFrame):
            cols = list(set(cols))
            pointcloud = pd.DataFrame(pointcloud, columns=cols)

        write_ply(filename, pointcloud[cols])
        print("Saved to:", filename)


def make_dtm(params):
    """
    This function will generate a Digital Terrain Model (dtm) based on the terrain labelled points.
    """

    if params.verbose: print("Making dtm...")

    params.grid_resolution = .5
    params.terrain_class = 2

    ### voxelise, identify lowest points and create DTM
    params.pc = voxelise(params.pc, params.grid_resolution, z=False)
    VX_map = params.pc.loc[~params.pc.VX.duplicated()][['xx', 'yy', 'VX']]
    ground = params.pc.loc[params.pc.label == params.terrain_class]
    ground.loc[:, 'zmin'] = ground.groupby('VX').z.transform(np.median)
    ground = ground.loc[ground.z == ground.zmin]
    ground = ground.loc[~ground.VX.duplicated()]

    X, Y = np.meshgrid(
        np.arange(params.pc.xx.min(), params.pc.xx.max() + params.grid_resolution, params.grid_resolution),
        np.arange(params.pc.yy.min(), params.pc.yy.max() + params.grid_resolution, params.grid_resolution))

    ground_arr = pd.DataFrame(data=np.vstack([X.flatten(), Y.flatten()]).T, columns=['xx', 'yy'])
    ground_arr = pd.merge(ground_arr, VX_map, on=['xx', 'yy'], how='outer')  # map VX to ground_arr
    ground_arr = pd.merge(ground[['z', 'VX']], ground_arr, how='right', on=['VX'])  # map z to ground_arr
    ground_arr.sort_values(['xx', 'yy'], inplace=True)

    # loop over incresing size of window until no cell are nan
    ground_arr.loc[:, 'ZZ'] = np.nan
    size = 3
    while np.any(np.isnan(ground_arr.ZZ)):
        ground_arr.loc[:, 'ZZ'] = ndimage.generic_filter(ground_arr.z.values.reshape(*X.shape),  # create raster,
                                                         lambda z: np.nanmedian(z), size=size).flatten()
        size += 2

    # ground_arr[['xx', 'yy', 'ZZ']].to_csv(os.path.join(params.odir, f'{params.basename}.dem.csv'), index=False)

    # apply to all points
    MAP = ground_arr.set_index('VX').ZZ.to_dict()
    params.pc.loc[:, 'n_z'] = params.pc.z - params.pc.VX.map(MAP)

    return params


def chunk_pc(pc, out_dir, params):
    def save_pts(pc, I, bx, by, bz, working_dir, params):

        pc = pc.loc[(pc.x.between(bx, bx + 6)) &
                    (pc.y.between(by, by + 6)) &
                    (pc.z.between(bz, bz + 6))]

        if len(pc) > 1000:

            if len(pc) > 20000:
                pc = pc.sample(n=20000)

            np.save(os.path.join(working_dir, f'{I:07}'), pc[['x', 'y', 'z', 'label']].values)

    if not os.path.isdir(out_dir): os.makedirs(out_dir)

    # apply global shift
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']] - pc[['x', 'y', 'z']].mean()

    pc.reset_index(inplace=True)
    pc.loc[:, 'pid'] = pc.index

    # generate bounding boxes
    xmin, xmax = np.floor(pc.x.min()), np.ceil(pc.x.max())
    ymin, ymax = np.floor(pc.y.min()), np.ceil(pc.y.max())
    zmin, zmax = np.floor(pc.z.min()), np.ceil(pc.z.max())

    box_dims = 6
    box_overlap = 0.5

    box_overlap = box_dims * box_overlap

    x_cnr = np.arange(xmin - box_overlap, xmax + box_overlap, box_overlap)
    y_cnr = np.arange(ymin - box_overlap, ymax + box_overlap, box_overlap)
    z_cnr = np.arange(zmin - box_overlap, zmax + box_overlap, box_overlap)

    # multithread segmenting points into boxes and save
    threads = []
    for i, (bx, by, bz) in enumerate(itertools.product(x_cnr, y_cnr, z_cnr)):
        threads.append(threading.Thread(target=save_pts, args=(pc, i, bx, by, bz, out_dir, params)))

    for x in tqdm(threads, desc='generating data blocks', disable=False if params.verbose else True):
        x.start()

    for x in threads:
        x.join()


def other_cylinder_fit2(xyz, xm=0, ym=0, xr=0, yr=0, r=1):
    from scipy.optimize import leastsq

    """
    https://stackoverflow.com/a/44164662/1414831

    This is a fitting for a vertical cylinder fitting
    Reference:
    http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf
    xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder
    th, threshold for the convergence of the least squares
    """

    x = xyz.x
    y = xyz.y
    z = xyz.z

    p = np.array([xm, ym, xr, yr, r])

    fitfunc = lambda p, x, y, z: (- np.cos(p[3]) * (p[0] - x) - z * np.cos(p[2]) * np.sin(p[3]) - np.sin(p[2]) * np.sin(
        p[3]) * (p[1] - y)) ** 2 + (z * np.sin(p[2]) - np.cos(p[2]) * (p[1] - y)) ** 2  # fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4] ** 2  # error function

    est_p, success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)

    return est_p


def RANSACcylinderFitting4(xyz_, iterations=50, plot=False):
    # if plot:
    #    ax = plt.subplot(111)

    bestFit, bestErr = None, np.inf
    xyz_mean = xyz_.mean(axis=0)
    xyz_ -= xyz_mean

    #     for i in tqdm(range(iterations), total=iterations, display=plot):
    for i in range(iterations):

        xyz = xyz_.copy()

        # prepare sample
        sample = xyz.sample(n=20)
        xyz = xyz.loc[~xyz.index.isin(sample.index)]

        x, y, a, b, radius = other_cylinder_fit2(sample, 0, 0, 0, 0, 0)
        centre = (x, y)
        if not np.all(np.isclose(centre, 0, atol=radius * 1.05)): continue

        MX = Rotation.from_euler('xy', [a, b]).inv()
        xyz[['x', 'y', 'z']] = MX.apply(xyz)
        xyz.loc[:, 'error'] = np.linalg.norm(xyz[['x', 'y']] - centre, axis=1) / radius
        idx = xyz.loc[xyz.error.between(.8, 1.2)].index  # 40% of radius is prob quite large

        # select points which best fit model from original dataset
        alsoInliers = xyz_.loc[idx].copy()
        if len(alsoInliers) < len(xyz_) * .2: continue  # skip if no enough points chosen

        # refit model using new params
        x, y, a, b, radius = other_cylinder_fit2(alsoInliers, x, y, a, b, radius)
        centre = [x, y]
        if not np.all(np.isclose(centre, 0, atol=radius * 1.05)): continue

        MX = Rotation.from_euler('xy', [a, b]).inv()
        alsoInliers[['x', 'y', 'z']] = MX.apply(alsoInliers[['x', 'y', 'z']])
        # calculate error for "best" subset
        alsoInliers.loc[:, 'error'] = np.linalg.norm(alsoInliers[['x', 'y']] - centre, axis=1) / radius

        if variation(alsoInliers.error) < bestErr:
            # for testing uncomment
            c = Circle(centre, radius=radius, facecolor='none', edgecolor='g')

            bestFit = [radius, centre, c, alsoInliers, MX]
            bestErr = variation(alsoInliers.error)

    if bestFit == None:
        # usually caused by low number of ransac iterations
        return np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz_)

    radius, centre, c, alsoInliers, MX = bestFit
    centre[0] += xyz_mean.x
    centre[1] += xyz_mean.y
    centre = centre + [xyz_mean.z]

    # for testing uncomment
    if plot:
        ax = plot
        radius, Centre, c, alsoInliers, MX = bestFit

        xyz_[['x', 'y', 'z']] = MX.apply(xyz_)
        xyz_ += xyz_mean
        ax.scatter(xyz_.x, xyz_.y, s=1, c='grey')

        alsoInliers[['x', 'y', 'z']] += xyz_mean
        cbar = ax.scatter(alsoInliers.x, alsoInliers.y, s=10, c=alsoInliers.error)
        # ax.colorbar(cbar)

        ax.scatter(Centre[0], Centre[1], marker='+', s=100, c='r')
        ax.add_patch(c)

        return [radius, centre, bestErr, len(xyz_), ax]

    return [radius, centre, bestErr, len(xyz_)]


def NotRANSAC(xyz):
    try:
        xyz = xyz[['x', 'y', 'z']]
        pca = PCA(n_components=3, svd_solver='auto').fit(xyz)
        xyz[['x', 'y', 'z']] = pca.transform(xyz)
        radius, centre = other_cylinder_fit2(xyz)

        if xyz.z.min() - radius < centre[0] < xyz.z.max() + radius or \
                xyz.y.min() - radius < centre[1] < xyz.y.max() + radius:
            centre = np.hstack([xyz.x.mean(), centre])
        else:
            centre = xyz.mean().values

        centre = pca.inverse_transform(centre)
    except:
        radius, centre = np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values

    return [radius, centre, np.inf, len(xyz)]


def RANSAC_helper(xyz, ransac_iterations, plot=False):
    #     try:
    if len(xyz) == 0:  # don't think this is required but....
        cylinder = [np.nan, np.array([np.inf, np.inf, np.inf]), np.inf, len(xyz)]
    elif len(xyz) < 10:
        cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0).values, np.inf, len(xyz)]
    elif len(xyz) < 50:
        cylinder = NotRANSAC(xyz)
    else:
        cylinder = RANSACcylinderFitting4(xyz[['x', 'y', 'z']], iterations=ransac_iterations, plot=plot)
    #             if cylinder == None: # again not sure if this is necessary...
    #                 cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0)]

    #     except:
    #         cylinder = [np.nan, xyz[['x', 'y', 'z']].mean(axis=0), np.inf, np.inf]

    return cylinder


def generate_path(samples, origins, n_neighbours=200, max_length=0,base=-1):
    # compute nearest neighbours for each vertex in cluster convex hull
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(samples[['x', 'y', 'z']])
    distances, indices = nn.kneighbors()
    from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.clstr.values, n_neighbours),
                                          samples.iloc[indices.ravel()].clstr.values,
                                          distances.ravel()]).T,
                               columns=['source', 'target', 'length'])

    # remove X-X connections
    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]

    # and build edge database where edges with min distance between clusters persist
    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    # remove edges that are likely leaps between trees
    edges = edges.loc[edges.length <= max_length]

    # removes isolated origin points i.e. > edge.length
    origins = [s for s in origins if s in edges.source.values]

    # compute graph
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    distance, shortest_path = nx.multi_source_dijkstra(G,sources=origins,weight='length')

    paths = pd.DataFrame(index=distance.keys(), data=distance.values(), columns=['distance'])
    paths.loc[:, 'base'] = base
    for p in paths.index: paths.loc[p, 'base'] = shortest_path[p][0]
    paths.reset_index(inplace=True)
    paths.columns = ['clstr', 'distance', 't_clstr']

    # identify nodes that are branch tips
    node_occurance = {}
    for v in shortest_path.values():
        for n in v:
            if n in node_occurance.keys():
                node_occurance[n] += 1
            else:
                node_occurance[n] = 1

    tips = [k for k, v in node_occurance.items() if v == 1]

    paths.loc[:, 'is_tip'] = False
    paths.loc[paths.clstr.isin(tips), 'is_tip'] = True

    return paths

def cube(pc):
    try:
        if len(pc) > 5:
            vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
            idx = np.random.choice(vertices, size=len(vertices), replace=False)
            return pc.loc[pc.index[idx]]
    except:
        pass
    return pc

def TLSSeg(pcd,spth):

    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', '-o', type=str, default=spth, help='output directory')

    parser.add_argument('--tindex', type=str, default=None, help='path to tile index')
    parser.add_argument('--n-tiles', default=3, type=int,
                        help='enlarges the number of tiles i.e. 3x3 or tiles or 5 x 5 tiles')
    parser.add_argument('--n-zeros', default=3, type=int, help='leading zeros for tile names')
    parser.add_argument('--overlap', default=False, type=float, help='buffer to crop adjacent tiles')
    parser.add_argument('--slice-thickness', default=.2, type=float, help='slice thickness for constructing graph')
    parser.add_argument('--find-stems-boundary', default=[1.5, 2.], nargs=2, type=float,
                        help='boundary height for slice used for identifying stems: default [1.5, 2.]')
    parser.add_argument('--find-stems-min-radius', default=.025, type=float, help='minimum radius of found stems')
    parser.add_argument('--find-stems-min-points', default=50, type=int,
                        help='minimum number of points for found stems')
    parser.add_argument('--graph-edge-length', default=1, type=float,
                        help='maximum distance used to connect points in graph')
    parser.add_argument('--graph-maximum-cumulative-gap', default=np.inf, type=float,
                        help='maximum cumulative distance between a base and a cluster')
    parser.add_argument('--min-points-per-tree', default=0, type=int,
                        help='minimum number of points for a identified tree')
    parser.add_argument('--add-leaves', default=1, action='store_true', help='add leaf points')
    parser.add_argument('--add-leaves-voxel-length', default=.5, type=float, help='voxel sixe when add leaves')
    parser.add_argument('--add-leaves-edge-length', default=1, type=float,
                        help='maximum distance used to connect points in leaf graph')
    parser.add_argument('--save-diameter-class', action='store_true', help='save into dimater class directories')
    parser.add_argument('--ignore-missing-tiles', action='store_true', help='ignore missing neighbouring tiles')
    parser.add_argument('--pandarallel', action='store_true', help='use pandarallel')
    parser.add_argument('--verbose', default=1,action='store_true', help='print something')
    # params = parser.parse_args()
    params, unknown = parser.parse_known_args()

    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True if params.verbose else False)
        except:
            print('--- pandarallel not installed ---')
            params.pandarallel = False



    if params.verbose:
        print('---- parameters ----')
        for k, v in params.__dict__.items():
            print(f'{k:<35}{v}')

    params.not_base = -1
    xyz = ['x', 'y', 'z']  # shorthand

    # params.dir, params.fn = os.path.split(params.tile)
    if params.tindex is not None:
        params.n = params.fn.split('.')[0]
    else:
        params.n = -1

    pc = pd.DataFrame(pcd,columns=['x','y','z','label'])
    params.pc = pc
    params = make_dtm(params)

    params.pc.loc[:, 'buffer'] = False
    params.pc.loc[:, 'fn'] = params.n

    bbox = {}
    bbox['xmin'], bbox['xmax'] = params.pc.x.min(), params.pc.x.max()
    bbox['ymin'], bbox['ymax'] = params.pc.y.min(), params.pc.y.max()
    bbox = dict2class(bbox)


    params.pc.reset_index(inplace=True)

    # save space
    params.pc = params.pc[[c for c in ['x', 'y', 'z', 'n_z', 'label', 'buffer', 'fn']]]
    params.pc[['x', 'y', 'z', 'n_z']] = params.pc[['x', 'y', 'z', 'n_z']].astype(np.float32)
    params.pc.label = params.pc.label.astype(np.int16)

    ### generate skeleton points
    if params.verbose: print('\n----- skeletonisation started -----')

    # extract stems points and slice slice
    stem_pc = params.pc.loc[params.pc.label == 3]

    # slice stem_pc
    stem_pc.loc[:, 'slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)

    # cluster within height slices
    stem_pc.loc[:, 'clstr'] = -1
    label_offset = 0

    for slice_height in tqdm(np.sort(stem_pc.n_slice.unique()),
                             disable=False if params.verbose else True,
                             desc='slice data vertically and clustering'):

        new_slice = stem_pc.loc[stem_pc.n_slice == slice_height]

        if len(new_slice) > 100:
            dbscan = DBSCAN(eps=.1, min_samples=20).fit(new_slice[xyz])
            new_slice.loc[:, 'clstr'] = dbscan.labels_
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = stem_pc.clstr.max() + 1

    # group skeleton points
    grouped = stem_pc.loc[stem_pc.clstr != -1].groupby('clstr')
    if params.verbose: print('fitting convex hulls to clusters')
    if params.pandarallel:
        chull = grouped.parallel_apply(cube)  # parallel_apply only works witn pd < 1.3
        print(chull)
    else:
        chull = grouped.apply(cube)  # don't think works with Jasmin or parallel_apply only works witn pd < 1.3
    chull = chull.reset_index(drop=True)

    ### identify possible stems ###
    if params.verbose: print('identifying stems...')
    skeleton = grouped[xyz + ['n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton.loc[:, 'dbh_node'] = False

    # dbh_nodes = skeleton.loc[skeleton.n_slice == params.slice_height].clstr
    find_stems_min = int(params.find_stems_boundary[0] // params.slice_thickness)
    find_stems_max = int(params.find_stems_boundary[1] // params.slice_thickness) + 1
    dbh_nodes_plus = skeleton.loc[skeleton.n_slice.between(find_stems_min, find_stems_max)].clstr
    dbh_slice = stem_pc.loc[stem_pc.clstr.isin(dbh_nodes_plus)]

    if len(dbh_slice) > 0:

        # remove noise from dbh slice
        nn = NearestNeighbors(n_neighbors=10).fit(dbh_slice[xyz])
        distances, indices = nn.kneighbors()
        dbh_slice.loc[:, 'nn'] = distances[:, 1:].mean(axis=1)
        dbh_slice = dbh_slice.loc[dbh_slice.nn < dbh_slice.nn.quantile(q=.9)]

        # run dbscan over dbh_slice to find potential stems
        dbscan = DBSCAN(eps=.2, min_samples=50).fit(dbh_slice[['x', 'y']])
        dbh_slice.loc[:, 'clstr_db'] = dbscan.labels_
        dbh_slice = dbh_slice.loc[dbh_slice.clstr_db > -1]
        dbh_slice.loc[:, 'cclstr'] = dbh_slice.groupby('clstr_db').clstr.transform('min')

        if len(dbh_slice) > 10:

            # ransac cylinder fitting
            if params.verbose: print('fitting cylinders to possible stems...')
            if params.pandarallel:
                dbh_cylinder = dbh_slice.groupby('cclstr').parallel_apply(RANSAC_helper, 100, ).to_dict()
            else:
                dbh_cylinder = dbh_slice.groupby('cclstr').apply(RANSAC_helper, 100, ).to_dict()
            dbh_cylinder = pd.DataFrame(dbh_cylinder).T
            dbh_cylinder.columns = ['radius', 'centre', 'CV', 'cnt']
            dbh_cylinder.loc[:, ['x', 'y', 'z']] = [[*row.centre] for row in dbh_cylinder.itertuples()]
            dbh_cylinder = dbh_cylinder.drop(columns=['centre']).astype(float)

            # identify clusters where cylinder CV <= .75 and label as nodes
            skeleton.loc[skeleton.clstr.isin(dbh_cylinder.loc[(dbh_cylinder.radius > params.find_stems_min_radius) &
                                                              (dbh_cylinder.cnt > params.find_stems_min_points) &
                                                              (
                                                                          dbh_cylinder.CV <= .15)].index.values), 'dbh_node'] = True

    in_tile_stem_nodes = skeleton.loc[(skeleton.dbh_node) &
                                      (skeleton.x.between(bbox.xmin, bbox.xmax)) &
                                      (skeleton.y.between(bbox.ymin, bbox.ymax))].clstr

    # generates paths through all stem points
    if params.verbose: print('generating graph, this may take a while...')
    wood_paths = generate_path(chull,
                               skeleton.loc[skeleton.dbh_node].clstr,
                               n_neighbours=200,
                               max_length=params.graph_edge_length,base=params.not_base)

    # removes paths that are longer for same clstr
    wood_paths = wood_paths.sort_values(['clstr', 'distance'])
    wood_paths = wood_paths.loc[~wood_paths['clstr'].duplicated()]

    # remove clusters that are linked to a base by a cumulative
    # distance greater than X
    wood_paths = wood_paths.loc[wood_paths.distance <= params.graph_maximum_cumulative_gap]

    if params.verbose: print('merging skeleton points with graph')
    stems = pd.merge(skeleton, wood_paths, on='clstr', how='left')

    # give a unique colour to each tree (helps with visualising)
    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or
                        c.startswith('green') or
                        c.startswith('blue')], inplace=True)

    # generate unique RGB for each stem
    unique_stems = stems.t_clstr.unique()
    RGB = pd.DataFrame(data=np.vstack([unique_stems,
                                       np.random.randint(0, 255, size=(3, len(unique_stems)))]).T,
                       columns=['t_clstr', 'red', 'green', 'blue'])
    RGB.loc[RGB.t_clstr == params.not_base, :] = [np.nan, 211, 211, 211]  # color unassigned points grey
    stems = pd.merge(stems, RGB, on='t_clstr', how='right')

    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(stem_pc,
                     stems[['clstr', 't_clstr', 'distance', 'red', 'green', 'blue']],
                     on='clstr')
    trees.loc[:, 'cnt'] = trees.groupby('t_clstr').t_clstr.transform('count')
    trees = trees.loc[trees.cnt > params.min_points_per_tree]

    in_tile_stem_nodes = trees.loc[trees.t_clstr.isin(in_tile_stem_nodes)].t_clstr.unique()

    # write out all trees
    params.base_I, I = {}, 0
    for i, b in tqdm(enumerate(dbh_cylinder.loc[in_tile_stem_nodes].sort_values('radius', ascending=False).index),
                     total=len(in_tile_stem_nodes),
                     desc='writing stems to file',
                     disable=False if params.verbose else True):


        if b == params.not_base:
            continue

        n = str(params.n)  # .zfill(params.n_zeros)

        trees.loc[trees.t_clstr == b,'Segmentation'] = I
        params.base_I[b] = I
        I += 1

    savetrees = trees[['x', 'y', 'z', 'Segmentation', 'red', 'green', 'blue']]
    savetrees.to_csv(os.path.join(params.odir, 'Segmentation_leafoff.txt'), sep='\t', index=False)

    if params.add_leaves:

        if params.verbose: print('adding leaves to stems, this may take a while...')

        # link stem number to clstr
        stem2tlsctr = stems[['clstr', 't_clstr']].loc[stems.t_clstr != params.not_base].set_index('clstr').to_dict()[
            't_clstr']
        chull.loc[:, 'stem'] = chull.clstr.map(stem2tlsctr)

        # identify unlabelled woody points to add back to leaves
        unlabelled_wood = chull.loc[[True if np.isnan(s) else False for s in chull.stem]]
        unlabelled_wood = stem_pc.loc[stem_pc.clstr.isin(unlabelled_wood.clstr.to_list() + [-1])]
        unlabelled_wood = unlabelled_wood.loc[unlabelled_wood.n_z >= 2]

        # extract wood points that are attributed to a base and that are the
        # the last clstr of the graph i.e. a tip
        is_tip = wood_paths.set_index('clstr')['is_tip'].to_dict()
        chull = chull.loc[[False if np.isnan(s) else True for s in chull.stem]]
        chull.loc[:, 'is_tip'] = chull.clstr.map(is_tip)
        chull = chull.loc[(chull.is_tip) & (chull.n_z > params.find_stems_boundary[0])]
        chull.loc[:, 'xlabel'] = 2

        # process leaf points
        lvs = params.pc.loc[(params.pc.label == 1) & (params.pc.n_z >= 2)].copy()
        lvs = pd.concat([lvs, unlabelled_wood])
        lvs.reset_index(inplace=True)

        # voxelise
        lvs = voxelise(lvs, length=params.add_leaves_voxel_length)
        lvs_gb = lvs.groupby('VX')[xyz]
        lvs_min = lvs_gb.min()
        lvs_max = lvs_gb.max()
        lvs_med = lvs_gb.median()

        # find faces of leaf voxels and create database
        cnrs = np.vstack([lvs_min.x, lvs_med.y, lvs_med.z]).T
        clstr = np.tile(np.arange(len(lvs_min.index)) + 1 + chull.clstr.max(), 6)
        VX = np.tile(lvs_min.index, 6)
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_med.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_max.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_max.z]).T])
        cnrs = pd.DataFrame(cnrs, columns=['x', 'y', 'z'])
        cnrs.loc[:, 'xlabel'] = 1
        cnrs.loc[:, 'clstr'] = clstr
        cnrs.loc[:, 'VX'] = VX

        # and combine leaves and wood
        branch_and_leaves = pd.concat([cnrs, chull[['x', 'y', 'z', 'label', 'stem', 'xlabel', 'clstr']]])
        branch_and_leaves.reset_index(inplace=True, drop=True)

        # find neighbouring branch and leaf points - used as entry points
        nn = NearestNeighbors(n_neighbors=2).fit(branch_and_leaves[xyz])
        distances, indices = nn.kneighbors()
        closest_point_to_leaf = indices[:len(cnrs), :].flatten()  # only leaf points
        idx = np.isin(closest_point_to_leaf, branch_and_leaves.loc[branch_and_leaves.xlabel == 2].index)
        close_branch_points = closest_point_to_leaf[idx]  # points where the branch is closest

        # remove all branch points that are not close to leaves
        idx = np.hstack([branch_and_leaves.iloc[:len(cnrs)].index.values, close_branch_points])
        bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(idx))]

        # generate a leaf paths graph
        leaf_paths = generate_path(bal,
                                   bal.loc[bal.xlabel == 2].clstr.unique(),
                                   max_length=1,  # i.e. any leaves which are separated by greater are ignored
                                   n_neighbours=20,base=params.not_base)

        leaf_paths = leaf_paths.sort_values(['clstr', 'distance'])
        leaf_paths = leaf_paths.loc[~leaf_paths['clstr'].duplicated()]  # removes duplicate paths
        leaf_paths = leaf_paths.loc[leaf_paths.distance > 0]  # removes within cluseter paths

        # linking indexs to stem number
        top2stem = branch_and_leaves.loc[branch_and_leaves.xlabel == 2].set_index('clstr')['stem'].to_dict()
        leaf_paths.loc[:, 't_clstr'] = leaf_paths.t_clstr.map(top2stem)
        #     paths.loc[:, 'stem'] = paths.stem_.map(base2i)

        # linking index to VX number
        index2VX = branch_and_leaves.loc[branch_and_leaves.xlabel == 1].set_index('clstr')['VX'].to_dict()
        leaf_paths.loc[:, 'VX'] = leaf_paths['clstr'].map(index2VX)

        # colour the same as stem
        lvs = pd.merge(lvs, leaf_paths[['VX', 't_clstr', 'distance']], on='VX', how='left')


        # and save

        for lv in tqdm(in_tile_stem_nodes):

            I = params.base_I[lv]
            trees.loc[trees['Segmentation']==I, 'wood'] = 1

            l2a = lvs.loc[lvs.t_clstr == lv]
            if len(l2a) > 0:
                l2a.loc[:, 'wood'] = 0

                rgb = RGB.loc[RGB.t_clstr == lv][['red', 'green', 'blue']].values[0] * 1.2
                lvs.loc[lvs.t_clstr == lv, ['red', 'green', 'blue']] = [c if c <= 255 else 255 for c in rgb]
                lvs.loc[lvs.t_clstr == lv,'Segmentation'] = I

        savelvs = lvs[['x', 'y', 'z','red', 'green', 'blue','Segmentation']]
        savelvs = pd.concat([savelvs,savetrees])

        savelvs.to_csv(os.path.join(params.odir, 'Segmentation_leafon.txt'), sep='\t', index=False)
    print(f'runtime: {(datetime.datetime.now() - start).seconds}')
