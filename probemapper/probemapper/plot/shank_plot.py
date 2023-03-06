import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd

mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42})

def shank_plot(data, sample, columns, vranges=None, show_region=True, show_title=False, savename=None, margin=8, arrow_size=15, figsize="auto"):
    """
    data: pd.DataFrame
    """
    num_panels = len(columns) + show_region
    if num_panels == 1:
        fig, axs = plt.subplots(1, num_panels, figsize=(0.5*num_panels, 8))
        axs = [axs]
    else:
        fig, axs = plt.subplots(1, num_panels, figsize=(0.5*num_panels, 8))
    
    if vranges is None:
        vranges = [None for c in columns]

    for (j, column) in enumerate(columns):
        v = data[data["probe"]==sample][column].to_numpy()

        # draw 1D heatmap
        if vranges[j]:
            vmin, vmax = vranges[j]
        else:
            vmin = np.percentile(v, 1)
            vmax = np.percentile(v, 99)

        axs[j].imshow(v[:, np.newaxis], cmap="jet", vmin=vmin, vmax=vmax,
                      origin="lower", extent=(-4, 4, 0, v.size))
        axs[j].set_xlim([-margin, margin])
        axs[j].set_ylim([-arrow_size,v.size+margin])
        axs[j].set_xticks([])
        if j == 0:
            axs[j].spines.left.set_position(('outward', 10))
        else:
            axs[j].spines.left.set_visible(False)
            axs[j].set_yticks([])
        axs[j].spines.right.set_visible(False)
        axs[j].spines.top.set_visible(False)
        axs[j].spines.bottom.set_visible(False)
        if show_title:
            axs[j].set_title(column, loc="center", rotation=70)

        # Draw background polygon
        xy = [
            [-margin, v.size+margin],
            [-margin, 0],
            [0, -arrow_size],
            [margin, 0],
            [margin, v.size+margin]
        ]
        polygon = plt.Polygon(xy, closed=True, fill=True, facecolor="lightgray", edgecolor="black", lw=0,
                              antialiased=True, zorder=-1,)
        axs[j].add_patch(polygon)

    if show_region:
        region_colors = data[data["probe"]==sample]["Region_color"].to_numpy()
        region_colors = np.array([e for e in region_colors])
        axs[-1].imshow(region_colors[:, np.newaxis, :], cmap="jet", origin="lower", extent=(-2, 2, 0, v.size))
        axs[-1].set_xlim([-margin, margin])
        axs[-1].set_ylim([-arrow_size,v.size+margin])
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
        axs[-1].spines.left.set_visible(False)
        axs[-1].spines.right.set_visible(False)
        axs[-1].spines.top.set_visible(False)
        axs[-1].spines.bottom.set_visible(False)

    if savename is not None:
        plt.savefig(savename, bbox_inches="tight")
    plt.show()
