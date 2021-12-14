import numpy as np
from skimage.morphology import binary_dilation
from scipy.stats import pearsonr


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
    cmap=np.zeros(frames.shape[1:])
    for i in range(frames.shape[1]):
        if i>0 and i<(frames.shape[1]-1):
            for j in range(frames.shape[2]):
                if j>0 and j<(frames.shape[2]-1):
                    # centre pixel trace
                    this_pixel = np.squeeze(frames[:,i,j])
                    # trace of sum of surround pixels
                    surr_pixels = np.squeeze(
                        np.sum(np.sum(np.squeeze(
                            frames[:,i-1:i+1,j-1:j+1]),2),1)) - this_pixel
                    C, _ = pearsonr(this_pixel, surr_pixels)
                    cmap[i,j]=C
    return cmap


def extract_roi(map, frames, threshold=0.8, max_size=200):
    """
    Extract an ROI by iteratively growing it from a seed pixel defined by the
    provided map image.

    Args:
        map (numpy.ndarray): the max value this image is used to select the
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
    roi = np.zeros(map.shape, dtype=bool)
    # position of pixel with highest value in the input map
    seed_pixel = np.unravel_index(np.argmax(map), shape=map.shape)
    # seed pixel is always included
    roi[seed_pixel[0], seed_pixel[1]] = True
    npix = 1
    # repeat cycles of growing the ROI until either the maximum number of pixels
    # is reached or no more pixels are added on a given cycle
    pixels_added = True
    while pixels_added and npix<=max_size:
        # get the fluorescence trace of pixels currently included
        roi_trace = np.mean(frames[:, roi], axis=(1))
        # select candidate pixels by dilating the ROI
        candidate_pixels = np.bitwise_and(binary_dilation(roi), np.invert(roi))
        pixels_added = False
        # for each candidate pixel, check if it passes the threshold for inclusion
        for candidate_pixel in np.argwhere(candidate_pixels):
            this_pixel = np.squeeze(frames[:, candidate_pixel[0], candidate_pixel[1]])
            if pearsonr(this_pixel, roi_trace)[0] > threshold:
                roi[candidate_pixel[0], candidate_pixel[1]] = True
                npix +=1
                pixels_added = True
    # zero the map values for pixels included in the ROI
    map[roi] = 0

    roi_trace = np.mean(frames[:, roi], axis=1)
    return (roi, roi_trace, npix, map)


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
        list: list of X x Y binary masks for each ROI.
        list: list of Z numpy.arrays with average fluorescence for each ROI.
        list: list of ROI sizes
        numpy.ndarray: X x Y sum of all ROI masks.

    """

    rois = []
    traces = []
    sizes = []
    for i in range(nsteps):
      (roi, trace, roi_size, stackmap) = extract_roi(
          stackmap,
          stack,
          threshold=threshold,
          max_size=max_size
        )
      if roi_size >= min_size:
        rois.append(roi)
        traces.append(trace)
        sizes.append(roi_size)

    all_rois = np.zeros(stackmap.shape)
    for roi in rois:
        all_rois = all_rois + roi

    return rois, traces, sizes, all_rois
