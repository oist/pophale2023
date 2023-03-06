import numpy as np
import os
from matplotlib import pyplot as plt
from .helper import run_shell_command
from ..io import save_nifti

def show_brain_side_by_side(im1, im2, num=10, savename=None):
    # merge two images along x axis
    merged = np.concatenate((im1, im2), axis=2)

    start = int(im1.shape[0]*0.05)
    end = int(im1.shape[0]*0.95)
    indice = np.linspace(start, end, num).astype("int")

    for i in indice:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.imshow(merged[i], cmap="gray", interpolation="nearest")
        ax.axis("off")
        plt.show()

    if savename is not None:
        save_nifti(merged, savename, 1, 1, 1)
        print("Merged image was saved as", savename)

def overlay_brains(im1, im2, vrange1, vrange2, gamma=1.0):
    rgb = np.zeros((*im1.shape, 3))
    # RED
    rgb[:,:,:,0] = 255 * (im1 - vrange1[0]) / (vrange1[1] - vrange1[0])
    # GREEN
    rgb[:,:,:,1] = 255 * (im2 - vrange2[0]) / (vrange2[1] - vrange2[0])
    # Gamma correction
    rgb = 255 * np.power(rgb/255, 1/gamma)

    return rgb.clip(0,255).astype("uint8")

def create_warped_grid_image(ants_path, deformation_field, output_image, directions="1x1x0", gs=50):
    cmd = [str(os.path.join(ants_path, "CreateWarpedGridImage"))]
    cmd.extend(["3", deformation_field, output_image])
    cmd.extend([directions])
    cmd.extend([f"{gs}x{gs}x{gs}"])

    print(cmd)
    r = run_shell_command(cmd, verbose=True)
    print(r)

def create_jacobian_determinant_image(ants_path, deformation_field, output_image):
    cmd = [str(os.path.join(ants_path, "CreateJacobianDeterminantImage"))]
    cmd.extend(["3", deformation_field, output_image])

    print(cmd)
    r = run_shell_command(cmd, verbose=True)
    print(r)
