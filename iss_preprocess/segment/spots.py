import cv2
import numpy as np
import pandas as pd
import scipy
from scipy.ndimage import grey_dilation

from ..coppafish import annulus

import iss_preprocess as iss


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
    if not np.any(isolated):
        raise ValueError("No isolated spots found")
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


def make_spot_image(
    spots, gaussian_width=30, dtype="single", output_shape=None, x_col="x", y_col="y"
):
    """Make an image by convolving spots with a gaussian

    A single isolated rolony results in a gaussian with sigma `gaussian_width`
    and an amplitude of 1

    Args:
        spots (pandas.DataFrame): Spots DataFrame. Must have `x` and `y` columns
        gaussian_width (int, optional): Width of the gaussian in pixels. Defaults to 30.
        dtype (str, optional): Datatype for computation. Defaults to "single".
        output_shape (tuple, optional): Shape of the output image. If None, the smallest
            shape fitting all spots + kernel will be used. Defaults to None.
        x_col (str, optional): Column name for x coordinates. Defaults to "x".
        y_col (str, optional): Column name for y coordinates. Defaults to "y".

    Returns:
        numpy.ndarray: Convolution results

    """
    kernel_size = gaussian_width * 20
    kernel_size += 1 - kernel_size % 2  # kernel shape must be odd
    kernel = cv2.getGaussianKernel(kernel_size, sigma=int(gaussian_width))
    # set the initial value so that single pixels after convolution have a peak of 1
    kernel /= kernel.max()
    kernel = kernel.astype(dtype)
    if output_shape is None:
        output_shape = np.array(
            (spots[y_col].max() + kernel_size + 1, spots[x_col].max() + kernel_size + 1)
        ).astype(int)
    spot_image = np.zeros(output_shape, dtype=dtype)
    spot_image[spots[y_col].values.astype(int), spots[x_col].values.astype(int)] = 1
    return cv2.sepFilter2D(
        src=spot_image,
        kernelX=kernel,
        kernelY=kernel,
        ddepth=-1,
        borderType=cv2.BORDER_ISOLATED,
    )


def convolve_spots(
    data_path,
    roi,
    kernel_um,
    prefix="barcode_round",
    dot_threshold=None,
    tile=None,
    output_shape=None,
):
    """Generate an image of spot density by convolution

    Args:
        data_path (str): Relative path to data
        roi (int): Roi ID
        kernel_um (float): Width of the kernel for convolution in microns
        prefix (str, optional): Prefix of the spots to load. Defaults to 'barcode_round'
        dot_threshold (float, optional): Threshold on the barcode dot_product_score to
            select spots to use. Defaults to None.
        tile (tuple, optional): Tile to use. Defaults to None.
        output_shape (tuple, optional): Shape of the output image. If not provided will
            return the smallest shape that includes (0,0) and all spots. Defaults to
            None.

    Returns:
        numpy.ndarray: 2D image of roi density

    """
    if tile is None:
        spots = pd.read_pickle(
            iss.io.get_processed_path(data_path) / f"{prefix}_spots_{roi}.pkl"
        )
    else:
        spots = pd.read_pickle(
            iss.io.get_processed_path(data_path)
            / "spots"
            / f"{prefix}_spots_{roi}_{tile[0]}_{tile[1]}.pkl"
        )
    if dot_threshold is not None:
        spots = spots[spots.dot_product_score > dot_threshold]

    # load barcode_round_1_1 but anything should work
    pixel_size = iss.io.get_pixel_size(data_path)
    gaussian_width = int(kernel_um / pixel_size)

    return make_spot_image(
        spots, gaussian_width=gaussian_width, dtype="single", output_shape=output_shape
    )
