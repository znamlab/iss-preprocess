from tifffile import TiffWriter
import numpy as np


def write_stack(stack, fname, bigtiff=False):
    """
    Write a stack to file as a multipage TIFF

    Args:
        stack (numpy.ndarray): X x Y x ... array (can have multiple channels /
            zplanes, etc.)
        fname (str): save path for the TIFF

    """
    stack = stack.reshape((stack.shape[0], stack.shape[1], -1))
    stack[stack < 0] = 0

    with TiffWriter(fname, bigtiff=bigtiff) as tif:
        for frame in range(stack.shape[2]):
            tif.write(np.uint16(stack[:, :, frame]), contiguous=True)
