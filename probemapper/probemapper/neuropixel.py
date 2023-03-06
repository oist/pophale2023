import h5py
import numpy as np
from scipy import signal

def load_data(filename, start, end, step, channels):
    """
    Load neuropixel data in HDF5 format.
    Parameters
    ----------
    filename: str
        HDF5 file path
    start: int
    end: int
    step: int
    channels: List[int]
    Returns
    -------
    t : Time stamp (in seconds)
    lfp_val : LFP signal (in V)
    mantle_val: mantle intensity (a.u.)
    """
    with h5py.File(filename, 'r') as f:
        lfp = f["lfpMS"]
        mantle = f["mantle"]
        
        if channels == "all":
            channels = np.arange(lfp.shape[1])
        
        t = np.arange(start, end, step)
        t = t / 1000 # convert to sec
        lfp_val = lfp[start:end:step, channels]
        mantle_val = mantle[start:end:step, 0]
    
    return t, lfp_val, mantle_val

def print_info(filename):
    """
    Print the information about the neuropixel recording data stored in HDF5.
    """
    with h5py.File(filename, 'r') as f:
        print("Available keys", list(f.keys()))
        print("LFP array shape", f["lfpMS"].shape)
        print("Mantle array shape", f["mantle"].shape)

def filter_no1(lfp):
    """
    Given a LFP input, this function applies 1-150 Hz bandpass filter and 55-65 Hz bandstop filter.
    Parameters
    ----------
    lfp: np.ndarray
    Returns
    -------
    out: np.ndarray
        Filtered LFP signal
    """
    bandpass = signal.butter(3, [1,150], 'bandpass', fs=1000, output='sos')
    bandstop = signal.butter(3, [55, 65], 'bandstop', fs=1000, output='sos')

    if lfp.ndim == 1:
        out = signal.sosfilt(bandpass, lfp)
        out = signal.sosfilt(bandstop, out)
    else:
        out = np.zeros_like(lfp)
        for i in range(lfp.shape[1]):
            v = lfp[:, i]
            v = signal.sosfilt(bandpass, v)
            v = signal.sosfilt(bandstop, v)
            out[:, i] = v
    
    return out
