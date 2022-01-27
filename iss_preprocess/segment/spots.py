import trackpy
import pandas as pd
import numpy as np
from skimage.feature import blob_log
from scipy.signal import medfilt2d


def detect_gene_spots(im, median_filter=False, min_size=1., max_sigma=4.):
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
        min_sigma=.5,
        num_sigma=10,
        log_scale=True,
        overlap=0.9,
        exclude_border=10
    )
    gene_spots = pd.DataFrame(spots_array, columns=['y', 'x', 'size'])
    gene_spots = gene_spots[gene_spots['size'] >= min_size]
    return gene_spots


def detect_spots(stack, method='trackpy', separation=4, diameter=9, threshold=100,
                 max_sigma=10):
    """
    Detect spots in a multichannel image based on standard deviation across channels
    using the selected detection method.

    Args:
        stack (numpy.ndarray): X x Y x C image stack
        method (str): detection method, either `trackpy` or `skimage`
        separation: minimum separation for spot detection, only applies to
            `trackpy` method
        diameter (int): spot diameter, only applies to `trackpy` method
        threshold (float): spot detection threshold
        max_sigma (float): maximum spot STD, only applies to `skimage` method

    Returns:
        pandas.DataFrame of spot location, including x, y, and size.

    """
    im = np.std(stack, axis=2)
    if method == 'trackpy':
        spots = trackpy.locate(
            im,
            separation=separation,
            diameter=diameter,
            threshold=threshold
        )
    elif method == 'skimage':
        spots_array = blob_log(
            im,
            max_sigma=max_sigma,
            threshold=threshold,
            min_sigma=1.,
            num_sigma=20,
        )
        spots = pd.DataFrame(spots_array, columns=['y', 'x', 'size'])
    else:
        raise(ValueError(f'Unknown spot detection method "{method}"'))

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
        dist = np.sqrt((clean_spots.x - spot.x)**2 + (clean_spots.y - spot.y)**2)
        if np.sum(dist < min_dist) > 1:
            # set x to nan to skip this spot
            clean_spots.iloc[ispot].x = np.nan
    return clean_spots[clean_spots.x.notna()]
