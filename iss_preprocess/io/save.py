from tifffile import TiffWriter
import numpy as np


def write_stack(stack, fname, bigtiff=False, dtype="uint16", clip=True):
    """
    Write a stack to file as a multipage TIFF

    Args:
        stack (numpy.ndarray): X x Y x ... array (can have multiple channels /
            zplanes, etc.)
        fname (str): save path for the TIFF
        dtype (str): datatype of the output image. Default to 'uint16'
        clip (bool): clip negative values before convertion
    """
    stack = stack.reshape((stack.shape[0], stack.shape[1], -1))
    if clip:
        stack = np.clip(stack, 0)

    with TiffWriter(fname, bigtiff=bigtiff) as tif:
        for frame in range(stack.shape[2]):
            tif.write(stack[:, :, frame].astype(dtype), contiguous=True)
