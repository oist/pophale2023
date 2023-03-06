import numpy as np
from scipy import ndimage as ndi

def NCC_3D(I, J, ws):
    """
    I, J: np.ndarray (3D)
    ws: local window size
    """
    # Use float64
    I = I.astype(np.float64)
    J = J.astype(np.float64)

    kernel = np.ones((ws,)*I.ndim)

    I_bar = I - ndi.convolve(I, kernel, mode="reflect") / kernel.sum()
    J_bar = J - ndi.convolve(J, kernel, mode="reflect") / kernel.sum()

    I_norm = ndi.convolve(I_bar ** 2, kernel, mode="reflect")
    I_norm = np.sqrt(I_norm)
    J_norm = ndi.convolve(J_bar ** 2, kernel, mode="reflect")
    J_norm = np.sqrt(J_norm)
    I_norm[I_norm==0] = 1
    J_norm[J_norm==0] = 1

    prod = ndi.convolve(I_bar * J_bar, kernel, mode="reflect")

    return prod / (I_norm * J_norm)
