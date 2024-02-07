import numpy as np
import scipy.fft
from math import cos, sin, radians
from skimage.transform import SimilarityTransform, warp


def transform_image(im, scale=1, angle=0, shift=(0, 0), cval=0.0):
    """
    Transform an image using provided scale, rotation angle, and shift.

    Args:
        im (numpy.ndarray): image to transform
        scale (float): scale factor
        angle (float): rotation angle in degrees
        shift (tuple): shift in x and y
        cval (float): value to fill in for pixels outside of the image

    Returns:
        numpy.ndarray: transformed image

    """
    tform = SimilarityTransform(matrix=make_transform(scale, angle, shift, im.shape))
    return warp(im, tform.inverse, preserve_range=True, cval=cval)


def make_transform(s, angle, shift, shape):
    """
    Make a transformation matrix using provided scale, rotation angle, and shift.

    Args:
        s (float): scale factor
        angle (float): rotation angle in degrees
        shift (tuple): shift in x and y
        shape (tuple): shape of the image

    Returns:
        numpy.ndarray: transformation matrix

    """
    angle = -radians(angle)
    center_x = float(shape[1]) / 2 - 0.5
    center_y = float(shape[0]) / 2 - 0.5
    shift_x = shift[1]
    shift_y = shift[0]
    tform = [
        [
            cos(angle) * s,
            -sin(angle) * s,
            shift_x + (center_x - s * (center_x * cos(angle) - center_y * sin(angle))),
        ],
        [
            sin(angle) * s,
            cos(angle) * s,
            shift_y + (center_y - s * (center_x * sin(angle) + center_y * cos(angle))),
        ],
        [0, 0, 1],
    ]
    return np.array(tform)


def phase_corr(
    reference: np.ndarray,
    target: np.ndarray,
    max_shift=None,
    min_shift=None,
    whiten=True,
    fft_ref=True,
) -> np.ndarray:
    """
    Compute phase correlation of two images.

    Args:
        reference (numpy.ndarray): reference image
        target (numpy.ndarray): target image
        max_shift (int): the range over which to search for the maximum of the
            cross-correlogram
        whiten (bool): whether or not to whiten the FFTs of the images. If True,
            the method performs phase correlation, otherwise cross correlation
        fft_ref (bool): whether to compute the FFT transform of the reference

    Returns:
        shift: numpy.array of the location of the peak of the cross-correlogram
        cc: numpy.ndarray of the cross-correlagram itself.

    """
    if fft_ref:
        f1 = scipy.fft.fft2(reference)
    else:
        f1 = reference
    f2 = scipy.fft.fft2(target)
    if whiten:
        f1 = f1 / np.abs(f1)
        f2 = f2 / np.abs(f2)
    cc = np.abs(scipy.fft.ifft2(f1 * np.conj(f2)))
    if max_shift:
        cc[max_shift:-max_shift, :] = 0
        cc[:, max_shift:-max_shift] = 0
    if min_shift:
        cc[:min_shift, :min_shift] = 0
        cc[-min_shift:, -min_shift:] = 0
        cc[-min_shift:, :min_shift] = 0
        cc[:min_shift, -min_shift:] = 0
    cc = scipy.fft.fftshift(cc)

    shift = (
        np.unravel_index(np.argmax(cc), reference.shape) - np.array(reference.shape) / 2
    )
    return shift, cc
