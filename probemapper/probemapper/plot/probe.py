from matplotlib import pyplot as plt
import numpy as np

def plot_overlay(t, lfp, mantle, savename=None):
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(t, lfp, color="blue")
    ax.set_xlabel("sec")
    ax.set_ylabel("V")

    ax2 = ax.twinx()
    ax2.plot(t, mantle, color="orange")
    ax2.set_ylabel("a.u.")
    
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    
    plt.show()

def plot_stacked(t, lfp, mantle, lfp_range=None, savename=None):
    fig, axs = plt.subplots(2, 1, figsize=(6,4))

    axs[0].plot(t, lfp, color="blue")
    if lfp_range is not None:
        axs[0].set_ylim(lfp_range)
    axs[0].set_xticks([]) # remove the axis
    axs[0].set(frame_on=False)

    axs[1].plot(t, mantle, color="orange")
    axs[1].set_xlabel("sec")
    axs[1].set(frame_on=False)
    
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    
    plt.show()

def voltage_lines(t, lfp, lfp_range=None, savename=None, vertical_scaling=1.0):
    
    num_channels = lfp.shape[1]
    fig, axs = plt.subplots(num_channels, 1, figsize=(15,num_channels*vertical_scaling*0.1))
    
    for i in range(num_channels):
        axs[i].plot(t, lfp[:, i], color="k")
        axs[i].set_xticks([]) # remove the axis
        axs[i].set_yticks([]) # remove the axis
        axs[i].set(frame_on=False)
        if lfp_range is not None:
            axs[i].set_ylim(lfp_range)
    
    # Last plot should have labels
    axs[-1].set_xlabel("sec")
    axs[-1].set_xticks(np.linspace(t[0], t[-1], 5))

    #plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    
    plt.show()

def voltage_heatmap2(t, lfp, lut, colormap, offset, sampling=10, vmin=-1e-4, vmax=1e-4, savename=None):
    
    num_channels = lfp.shape[1]
    ann = np.zeros((num_channels,1), dtype="int")
    tmp = lut[["Region"]].loc[100-offset:].to_numpy()
    ann[0:tmp.shape[0]] = tmp
    tmin, tmax = t[0], t[-1]
    tlabels = np.linspace(0, t[-1]-t[0], 3)
    tlabels = np.round(tlabels, decimals=1)
    
    grid = lfp.T[:, ::sampling]
    ann_image = colormap[ann]
        
    fig, axs = plt.subplots(1, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [9, 1]})
    im = axs[0].imshow(grid, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    #im = ax.imshow(grid, cmap="plasma", vmin=0, vmax=0.0001, aspect="auto")
    axs[0].set_ylabel("Channel")
    axs[0].set_xlabel("Time (sec)")
    axs[0].invert_yaxis()
    axs[0].set_xticks(np.linspace(0, grid.shape[1], 3))
    axs[0].set_xticklabels(tlabels)
    #cbar = axs[0].figure.colorbar(im, ax=axs[0], location="left")
    #cbar.ax.set_ylabel("Voltage", rotation=-90, va="bottom")
    
    axs[1].imshow(ann_image, aspect=1/10.0, interpolation="nearest")
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].invert_yaxis()
    #axs[1].set(frame_on=False)
    
    plt.tight_layout()
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    plt.show()
