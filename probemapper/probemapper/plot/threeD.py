from mayavi import mlab
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
import nrrd

def _scalar2rgb(v, vmin, vmax, cmap="jet"):
    cmap = mpl.cm.get_cmap(cmap)
    v = (v - vmin) / (vmax - vmin)
    v = np.clip(v, 0, 1)
    return cmap(v)

def render_probes_on_surface(df, samples, column, atlas_dir, vmin, vmax,
                             surface_coloring_by="region", probe_colormap="jet", surface_colormap="Reds",
                             surface_vmin=None, surface_vmax=None, surface_opacity=0.15,
                             savename=None):
    """
    df: pandas.DataFrame
    samples: List of str
    column: str
    atlas_dir: str
    vmin: float
    vmax: float
    """

    if surface_vmin is None: surface_vmin = vmin
    if surface_vmax is None: surface_vmax = vmax

    # load atlas data
    atlas_lut = pd.read_csv(os.path.join(atlas_dir, "region_LUT.csv"))
    atlas_lut["region_name2"] = ["VL", "Others", "sFL", "iFL", "Buc", "Subfr", "aBL", "dBL", "Subv", "Prec"]

    if samples == "all":
        samples = list(df["probe"].unique())

    # compute region mean
    filtered = df[df["probe"].isin(samples)]
    atlas_lut["mean"] = [filtered[filtered["Region"]==i][column].mean() for i in atlas_lut["ID"]]
    atlas_lut["mean"] = [0 if np.isnan(e) else e for e in atlas_lut["mean"]]

    mlab.figure(size=(1500, 1500), bgcolor=(1, 1, 1))

    # render brain surface
    for index, row in atlas_lut.iterrows():
        region_id = row["ID"]
        if region_id in [3,12]:
            continue
        if surface_coloring_by == "region":
            color = [int(x) for x in row["region_color"].split(',')]
            color = np.array([v/255 for v in color]).clip(0,1)
            color = tuple(color.tolist())
        elif surface_coloring_by == "intensity":
            color = _scalar2rgb(row["mean"], vmin=surface_vmin, vmax=surface_vmax, cmap=surface_colormap)
            color = tuple(color[0:3])
        elif surface_coloring_by == "gray":
            color = (0.8, 0.8, 0.8)

        d = mlab.pipeline.open(os.path.join(atlas_dir, "Slicer_3D", f"{region_id}.ply"))
        mlab.pipeline.surface(d, color=color, opacity=surface_opacity)

    for sample in samples:
        data = df[df["probe"]==sample]
        x, y, z = data["X"], data["Y"], data["Z"]
        v = data[column]
        mlab.plot3d(x, y, z, v, tube_radius=1, colormap=probe_colormap, vmin=vmin, vmax=vmax)

    mlab.view(0, 90, 800, focalpoint=(225, 240, 275), roll=240) # saggital view
    if savename is not None:
        mlab.savefig(savename)
    mlab.show()

def render_probes_on_slice(df, samples, column, atlas_dir, vmin, vmax, savename=None):
    """
    df: pandas.DataFrame
    """
    # load atlas data
    atlas_lut = pd.read_csv(os.path.join(atlas_dir, "region_LUT.csv"))
    atlas_lut["region_name2"] = ["VL", "Others", "sFL", "iFL", "Buc", "Subfr", "aBL", "dBL", "Subv", "Prec"]

    if samples == "all":
        samples = list(df["probe"].unique())

    # compute region mean
    filtered = df[df["probe"].isin(samples)]
    atlas_lut["mean"] = [filtered[filtered["Region"]==i][column].mean() for i in atlas_lut["ID"]]

    # prepave volume data for 2D slice plot
    annotation, _ = nrrd.read(os.path.join(atlas_dir, "Slicer_3D", "Segmentation.seg.nrrd"))
    vol = np.zeros_like(annotation, dtype="float")
    for (i, row) in atlas_lut.iterrows():
        if row["ID"] in [3,12]:
            continue
        vol[annotation==row["ID"]] = row["mean"]
    x, y, z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

    mlab.figure(size=(1500, 1500), bgcolor=(1, 1, 1))

    # Slice
    sl = mlab.volume_slice(x, y, z, vol,
                           plane_orientation='x_axes', slice_index=210, colormap="Reds", vmin=vmin, vmax=vmax)
    lut = sl.module_manager.scalar_lut_manager.lut.table.to_array()
    lut[:, 3] = int(0.4 * 255) # opacity = 0.4
    lut[0] = (255,255,255,0) # 0 = white
    sl.module_manager.scalar_lut_manager.lut.table = lut

    for sample in samples:
        data = df[df["probe"]==sample]
        x, y, z = data["X"], data["Y"], data["Z"]
        v = data[column]
        mlab.plot3d(x, y, z, v, tube_radius=1, colormap="jet", vmin=vmin, vmax=vmax)

    mlab.view(0, 90, 800, focalpoint=(225, 240, 275), roll=240) # saggital view
    if savename is not None:
        mlab.savefig(savename)
    mlab.show()
