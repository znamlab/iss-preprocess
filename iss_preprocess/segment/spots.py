import trackpy
import pandas as pd
import numpy as np
from skimage.feature import blob_log


def detect_spots(stack, method='trackpy', separation=4, diameter=9, threshold=100,
                 min_dist=5., max_sigma=10):
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
        min_dist (float): minimum distance between spots to avoid duplicate spots
            when combining across channels
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
