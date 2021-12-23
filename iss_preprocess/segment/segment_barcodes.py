import numpy as np
from skimage.morphology import binary_dilation
from .roi import ROI


def correlation_map(frames):
    """
    Compute the correlation map based on the provided fluorescence stack.

    The correlation map value for each pixel is defined as its Pearson
    correlation coefficient with the sum of surrounding pixels.

    Args:
        frames (numpy.ndarray): Z x X x Y stack. For the puposes of ROI detection
            both the channels and rounds dimensions of the stack can be collapsed
            into one.

    Returns:
        numpy.ndarray: X x Y correlation map.

    """
    cmap = np.zeros(frames.shape[1:])
    for i in range(frames.shape[1]):
        if 0 < i < (frames.shape[1] - 1):
            for j in range(frames.shape[2]):
                if 0 < j < (frames.shape[2] - 1):
                    # centre pixel trace
                    this_pixel = np.squeeze(frames[:, i, j])
                    # trace of sum of surround pixels
                    surr_pixels = np.squeeze(
                        np.sum(np.sum(np.squeeze(
                            frames[:, i - 1:i + 1, j - 1:j + 1]), 2), 1)) - this_pixel
                    C = corrcoef(this_pixel, surr_pixels)
                    cmap[i, j] = C
    return cmap


def corrcoef(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)
    x = x - meanx
    y = y - meany
    return np.sum(x * y) / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


def extract_roi(stackmap, frames, threshold=0.8, max_size=200):
    """
    Extract an ROI by iteratively growing it from a seed pixel defined by the
    provided map image.

    Args:
        stackmap (numpy.ndarray): the max value this image is used to select the
            seed pixel.
        frames (numpy.ndarray): Z x X x Y stack.
        threshold (float): threshold for correlation between ROI and surrounding
            pixels used to determine, which pixels to add.
        max_size (int): maximum number of pixels in an ROI.

    Returns:
        numpy.ndarray: X x Y binary mask of pixels in the ROI.
        numpy.array: Z timeseries of average fluorescence values in the ROI.
        int: number of pixels in the ROI.
        numpy.ndarray: the provided map with pixels in the ROI set to 0 so that
            they are not used as seed pixels in the next round.

    """
    roi = np.zeros(stackmap.shape, dtype=bool)
    # position of pixel with highest value in the input map
    seed_pixel = np.unravel_index(np.argmax(stackmap), shape=stackmap.shape)
    # seed pixel is always included
    roi[seed_pixel[0], seed_pixel[1]] = True
    npix = 1
    # repeat cycles of growing the ROI until either the maximum number of pixels
    # is reached or no more pixels are added on a given cycle
    pixels_added = True
    while pixels_added and npix <= max_size:
        # get the fluorescence trace of pixels currently included
        roi_trace = np.mean(frames[:, roi], axis=1)
        # select candidate pixels by dilating the ROI
        candidate_pixels = np.bitwise_and(binary_dilation(roi), np.invert(roi))
        pixels_added = False
        # for each candidate pixel, check if it passes the threshold for inclusion
        for candidate_pixel in np.argwhere(candidate_pixels):
            this_pixel = np.squeeze(frames[:, candidate_pixel[0], candidate_pixel[1]])
            if corrcoef(this_pixel, roi_trace) > threshold:
                roi[candidate_pixel[0], candidate_pixel[1]] = True
                npix += 1
                pixels_added = True
    # zero the map values for pixels included in the ROI
    stackmap[roi] = 0

    roi_trace = np.mean(frames[:, roi], axis=1)
    return (roi, roi_trace, npix)


def detect_rois(stack, stackmap, min_size=4, max_size=500, threshold=0.5, nsteps=1000):
    """
    Iteratively detect ROIs using the `extract_roi` function.

    Args:
        stack (numpy.ndarray): Z x X x Y image stack. For the puposes of ROI detection
            both the channels and rounds dimensions of the stack can be collapsed
            into one.
        stackmap (numpy.ndarray): X x Y map image, such as a correlation or
            variance map, used to seed ROI detection.
        min_size (int): minimum number of pixels per ROI. Smaller ROIs are
            discarded.
        max_size (int): maximum number of pixels per ROI. The ROI detection
            algorithm stops growing ROIs when this number is reached.
        nsteps (int): number of times to call the `extract_roi` function.

    Returns:
        list: list of ROI objects
        numpy.ndarray: X x Y sum of all ROI masks.

    """

    rois = []
    for i in range(nsteps):
        (roi_mask, trace, roi_size) = extract_roi(
            stackmap,
            stack,
            threshold=threshold,
            max_size=max_size
        )
        if roi_size >= min_size:
            rois.append(ROI(mask=roi_mask, trace=trace))

    return rois


def create_overlap_map(rois):
    """
    Create an overlap map counting the number of ROIs that include each pixel.

    Args:
        rois (list): list of ROI object

    Returns:
        numpy.ndarray

    """
    overlap_map = np.zeros(rois[0].shape)
    for roi in rois:
        overlap_map += roi.mask
    return overlap_map


def find_overlappers(rois, max_overlap=0.5):
    """
    Find overlapping ROIs and get rid of them.

    Args:
        rois (list): list of binary ROI masks.
        max_overlap (float): maximum overlap for inclusion.

    Returns:
        list: list of booleans corresponding to ROIs to keep.

    """

    overlap_map = create_overlap_map(rois)
    keep_rois = []
    for roi in reversed(rois):  # start from the end of the list
        if np.mean(overlap_map[roi.mask] > 1) > max_overlap:
            keep_rois.append(False)
            overlap_map[roi.mask] -= 1
        else:
            keep_rois.append(True)

    return keep_rois[::-1]


def remove_overlaps(rois):
    """

    Get rid of overlaps between roi masks.

    Args:
        rois:

    """
    overlap_map = create_overlap_map(rois)
    for roi in rois:
        roi.mask = np.logical_and(roi.mask, np.logical_not(overlap_map>1))