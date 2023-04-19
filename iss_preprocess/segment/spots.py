import pandas as pd
import numpy as np
from skimage.feature import blob_log
from scipy.signal import medfilt2d
from scipy.ndimage import grey_dilation
import cv2
import scipy
from ..coppafish import annulus


def detect_isolated_spots(
    im, detection_threshold=40, isolation_threshold=30, annulus_r=(3, 7)
):
    """
    Detect spots that are isolated from their neighbors.
    
    For each spot, we compute the average intensity in a circular annulus around
    the spot. If the average intensity is below a threshold, we consider the spot
    to be isolated.
    
    Args:
        im (numpy.ndarray): X x Y image
        detection_threshold (float): threshold for initial spot detection
        isolation_threshold (float): threshold for spot isolation. Lower values
            in fewer spots considered isolated.
        annulus_r (tuple): inner and outer radii of the annulus used to compute
            the average intensity around each spot.

    Returns:
        pandas.DataFrame of spots
    
    """
    spots = detect_spots(im, threshold=detection_threshold)
    strel = annulus(annulus_r[0], annulus_r[1])
    strel = strel / np.sum(strel)
    annulus_image = scipy.ndimage.correlate(im, strel)
    isolated = annulus_image[spots["y"], spots["x"]] < isolation_threshold
    return spots.iloc[isolated]


def detect_spots(im, threshold=100, spot_size=2):
    """
    Detect peaks in an image.

    TODO: no point assigning size here.

    Args:
        stack (numpy.ndarray): X x Y x C image stack
        threshold (float): spot detection threshold
        spot_size (float): spot size in pixels. This value is simply assigned to
            the "size" column of the output DataFrame.

    Returns:
        pandas.DataFrame of spot location, including x, y, and size.

    """
    dilate = grey_dilation(im, size=(4, 4))
    small = 1e-6
    spots = np.logical_and(im + small > dilate, im > threshold)
    coors = np.where(spots)
    spots = pd.DataFrame(
        {"y": coors[0], "x": coors[1], "size": np.ones(len(coors[0])) * spot_size}
    )

    return spots

def make_spot_image(spots, gaussian_width=30, dtype="single", output_shape=None):
    """Make an image by convolving spots with a gaussian

    A single isolated rolony results in a gaussian with sd about `kernel_size / 3` and
    an amplitude of 1

    Args:
        spots (pandas.DataFrame): Spots DataFrame. Must have `x` and `y` columns
        gaussian_width (int, optional): Width of the gaussian in pixels. Defaults to 30.
        dtype (str, optional): Datatype for computation. Defaults to "single".
        output_shape (tuple, optional): Shape of the output image. If None, the smallest
            shape fitting all spots + kernel will be used. Defaults to None.

    Returns:
        numpy.ndarray: Convolution results
        
    """
    kernel_size = gaussian_width * 8
    kernel_size += 1 - kernel_size % 2  # kernel shape must be odd
    kernel = cv2.getGaussianKernel(kernel_size, sigma=int(gaussian_width))
    # set the initial value so that single pixels after convolution have a peak of 1
    kernel /= kernel.max()
    kernel = kernel.astype(dtype)
    if output_shape is None:
        output_shape = np.array(
            (spots.y.max() + kernel_size + 1, spots.x.max() + kernel_size + 1)
        ).astype(int)
    spot_image = np.zeros(output_shape, dtype=dtype)
    spot_image[spots.y.values.astype(int), spots.x.values.astype(int)] = 1
    return cv2.sepFilter2D(src=spot_image, kernelX=kernel, kernelY=kernel, ddepth=-1)
