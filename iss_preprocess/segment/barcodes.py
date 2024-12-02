import cv2
import numpy as np


def segment_spot_image(
    spot_image, binarise_threshold=5.0, distance_threshold=10.0, debug=False
):
    """Segment a spot image using opencv

    Args:
        spot_image (numpy.ndarray): 2D image to segment
        binarise_threshold (float, optional): Threshold for initial binarisation. Will
            cut isolated rolonies. Defaults to 5.
        distance_threshold (float, optional): Distance threshold for initial
            segmentation. Defaults to 10.
        debug (bool, optional): Return intermediate results if True. Defaults to False.

    Returns:
        numpy.ndarray or dict: Segmented image. Background is 0, borders -1, other
            numbers label individual cells. If debug is True, return a dictionary with
            intermediate results

    """
    # binarise
    mask = 255 * (spot_image > binarise_threshold).astype("uint8")
    kernel = np.ones((5, 5), dtype="uint8") * 255
    background = cv2.dilate(mask, kernel, iterations=10)
    dst2nonzero = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5)
    is_cell = 255 * (dst2nonzero > distance_threshold).astype("uint8")
    ret, markers = cv2.connectedComponents(is_cell)
    # make the background to 1
    markers += 1
    # and part to watershed to 0
    markers[np.bitwise_xor(background, is_cell).astype(bool)] = 0
    # watershed required a rgb image
    stack = cv2.normalize(spot_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    stack = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
    water = cv2.watershed(stack, markers)
    water -= 1  # put the background seed to 0.
    water[water < 0] = 0  # put borders into background
    if debug:
        return dict(
            binary=mask,
            background=background,
            seeds=is_cell,
            distance=dst2nonzero,
            initial_labels=markers,
            watershed=water,
        )
    return water
