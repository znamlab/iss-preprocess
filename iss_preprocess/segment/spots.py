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
    spots = detect_spots(im, threshold=detection_threshold)
    strel = annulus(annulus_r[0], annulus_r[1])
    strel = strel / np.sum(strel)
    annulus_image = scipy.ndimage.correlate(im, strel)
    isolated = annulus_image[spots["y"], spots["x"]] < isolation_threshold
    return spots.iloc[isolated]


def detect_gene_spots(im, median_filter=False, min_size=1.0, max_sigma=4.0):
    """
    Detect spots corresponding to single rolonies from OMP coefficient images.

    Args:
        im (numpy.ndarray): X x Y image of OMP coefficients for a single gene.
        median_filter (bool): whether to apply a 3x3 median filter before spot
            detection. Can be helpful to deal with single noise pixels.
        min_size (float): minimum size threshold for spots. Helps avoid spurious
            mini-spots next to real ones.
        max_sigma (float): maximum sigma for the spot detection algorithm.

    Returns:
        pandas.DataFrame of spots containing 'x', 'y', and 'size' columns.

    """
    if median_filter:
        im = medfilt2d(im, kernel_size=3)
    spots_array = blob_log(
        im,
        max_sigma=max_sigma,
        min_sigma=0.5,
        num_sigma=10,
        log_scale=True,
        overlap=0.9,
        exclude_border=10,
    )
    gene_spots = pd.DataFrame(spots_array, columns=["y", "x", "size"])
    gene_spots = gene_spots[gene_spots["size"] >= min_size]
    return gene_spots


def detect_spots(im, threshold=100, spot_size=2):
    """
    Detect peaks in an image.

    Args:
        stack (numpy.ndarray): X x Y x C image stack
        threshold (float): spot detection threshold

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


def filter_spots(spots, min_dist):
    """
    Eliminate duplicate spots closer than a distance threshold.

    Args:
        spots: pandas.DataFrame containing spot coordinates
        min_dist: minimum distance threshold

    Returns:
        pandas.DataFrame after filtering.

    """
    clean_spots = spots.copy()
    for ispot, spot in spots.iterrows():
        dist = np.sqrt((clean_spots.x - spot.x) ** 2 + (clean_spots.y - spot.y) ** 2)
        if np.sum(dist < min_dist) > 1:
            # set x to nan to skip this spot
            clean_spots.iloc[ispot].x = np.nan
    return clean_spots[clean_spots.x.notna()]


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
