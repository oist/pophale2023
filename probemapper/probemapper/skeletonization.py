import numpy as np
from skan import Skeleton, summarize, draw
from skimage import morphology
from scipy import interpolate
from scipy import ndimage as ndi
import pandas as pd
import nrrd

def skeletonize(image, um_per_voxel):
    skeleton3d = morphology.skeletonize_3d(image)
    skeleton3d = cleanup_skeleton(skeleton3d, thresh=10)

    sk = Skeleton(skeleton3d)
    print(sk.n_paths)

    coords = []
    for i in range(sk.n_paths):
        coords.append(sk.path_coordinates(i) * um_per_voxel)
    coords = np.concatenate(coords, axis=0)
    coords = coords.T

    return coords

def cleanup_skeleton(skeleton_image, thresh):
    out = np.array(skeleton_image)
    sk = Skeleton(skeleton_image)
    branch_data = summarize(sk)

    for i, row in branch_data.iterrows():
        if row["branch-distance"] < thresh and row["branch-type"] == 1:
            out[tuple(sk.path_coordinates(i).astype("int").T)] = False

    out = morphology.binary_dilation(out)
    out = morphology.skeletonize(out)
    
    return out

def fit_with_spline(coords, channel_spacing, extrapolate_channels):
    """
    Parameters
    ----------
    coords: np.ndarray, shape = (N, 3)
    """

    # 3rd order spline fit
    u = np.linspace(0,1,coords[0].shape[0])
    tck_o3, _ = interpolate.splprep(coords, u=u, k=3, s=coords[0].shape[0]*50)

    # 1st order spline fit
    u = np.linspace(0,1,coords[0].shape[0])
    tck_o1, _ = interpolate.splprep(coords, u=u, k=1, s=coords[0].shape[0]*50)

    # Total length of the spline
    fitted1 = interpolate.splev(np.linspace(0, 1.0, 1000), tck_o3, der=0)
    dist = np.diff(np.array(fitted1).T, axis=0)
    dist = np.power(dist, 2).sum(axis=1)
    dist = np.sqrt(dist).sum()

    num_channels = int(dist / channel_spacing)
    margin = extrapolate_channels
    s1 = interpolate.splev(np.linspace(margin*1.0/num_channels, 1.0, num_channels-margin), tck_o3, der=0)
    s2 = interpolate.splev(np.linspace(-margin*1.0/num_channels, margin*1.0/num_channels, 2*margin), tck_o1, der=0)
    s1 = np.array(s1)
    s2 = np.array(s2)
    s = np.concatenate((s2, s1), axis=1)

    return s, dist, num_channels

def query_region_id(coords_in_um, atlas_path, margin, num_channels, atlas_um_per_pixel=10):
    ann, _ = nrrd.read(atlas_path)
    ann = np.swapaxes(ann, 0, 2)

    regions = ndi.map_coordinates(ann, coords_in_um/atlas_um_per_pixel, order=0, mode="constant", cval=0)
    df = pd.DataFrame()
    df["channel"] = np.arange(-margin, num_channels)
    df["X"] = coords_in_um[2]
    df["Y"] = coords_in_um[1]
    df["Z"] = coords_in_um[0]
    df["Region"] = regions

    return df

def query_region_id2(coords_in_um, atlas_path, margin, num_channels, atlas_um_per_pixel=10):
    ann, _ = nrrd.read(atlas_path)
    ann = np.swapaxes(ann, 0, 2)
    regions = ndi.map_coordinates(ann, coords_in_um/atlas_um_per_pixel, order=0, mode="constant", cval=0)
    return regions

def generate_skeleton_image(coords, atlas_path, atlas_um_per_pixel=10):
    ann, _ = nrrd.read(atlas_path)
    ann = np.swapaxes(ann, 0, 2)

    out = np.zeros_like(ann)
    for e in coords.T:
        idx = (e/atlas_um_per_pixel).astype("int")
        out[idx[0], idx[1], idx[2]] = 1
    # dilation
    out = ndi.binary_dilation(out, iterations=2)
    
    return out
