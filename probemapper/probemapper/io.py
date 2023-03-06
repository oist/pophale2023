import os, glob
import tifffile
import numpy as np
import nibabel as nb

def load_tiff_sequence(imgdir, imgtype='tif', zrange=None):
	"""
	Load image sequence stored in the same directory.
    Only monochrome image can be loaded with this function.
	e.g.
	vol = load_tiff_sequence(imgdir, 'tif', zrange=[10,100])
	"""

	imglist = glob.glob(os.path.join(imgdir, f"*.{imgtype}"))
	imglist.sort() # sort numerically

	if zrange is not None:
		imglist = imglist[zrange[0]:zrange[1]]

	# get image properties by reading the first image
	im = tifffile.imread(imglist[0])
	imsize = (len(imglist), *im.shape)
	imtype = im.dtype

	stack = np.zeros(imsize, dtype=imtype)
	for (i,impath) in enumerate(imglist):
		stack[i,:,:] = tifffile.imread(impath, maxworkers=1)

	return stack

def load_nifti(src: str):
    """
    Load nifti image. Array is rearranged so that it aligns with TIFF or HDF5 format.
    Parameters
    ----------
    src : str
    """
    stack = nb.load(src).get_data()
    # the below line is needed to make it consistent with TIFF array order
    stack = np.swapaxes(stack, 0, 2)
    return stack

def save_nifti(stack, niftiname, spx, spy, spz):
    """
    Write array as NIfTI format.
    Note: array layout is reordered so that tiff and nifti can be treated equally;
    In tiff, z is 0th index, i.e. to make xy slice,
    slice = stack[ i, :, : ]
    On the other hand, in nifti, z is the 2nd index, i.e. to make xy slice,
    slice = stack[ :, :, i ]
    Parameters
    ----------
    stack : np.ndarray
    niftiname : str
    spx : float
    spy : float
    spz : float
    Returns
    -------
    None
    Examples
    --------
    vol = save_nifti(stack, 'image.nii.gz', 3, 3, 3)
    """
    # swap axis!
    stack = np.swapaxes(stack,0,2)
    nim = nb.Nifti1Image(stack, affine=None)
    # define voxel spacing and orientation
    # ANTS uses qform when reading NIFTI-1 images
    aff = np.diag([-spx,-spy,spz,1]) # sign is due to the inconsistency between nibabel and ANTS (or ITK)
    nim.header.set_qform(aff, code=2)
    # write!
    nim.to_filename(niftiname)
