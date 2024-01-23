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


def masked_phase_corr(  
    reference: np.ndarray,
    target: np.ndarray,
    reference_mask: np.ndarray,
    target_mask: np.ndarray,
    max_shift: int=None,
    min_shift: int=None,
    fft_ref: bool=True,
    reference_mask_fft: np.ndarray = None,
    overlap_ratio: float = 0.3,
) -> np.ndarray:
    """
    Compute phase correlation of two images.

    masks must be boolean arrays of the same shape as the images.
    all images must be the same shape.

    """

    float_dtype = reference.dtype
    eps = np.finfo(float_dtype).eps

    if fft_ref:
        reference[np.logical_not(reference_mask)] = 0.0
        fixed_fft = scipy.fft.fft2(reference)
    
    if reference_mask_fft is None:
        fixed_mask_fft = scipy.fft.fft2(reference_mask.astype(float_dtype))
    else:
        fixed_mask_fft = reference_mask_fft

    # all computation are made on the rotated image
    rotated_moving_image = target[::-1, ::-1]
    rotated_moving_mask = target_mask[::-1, ::-1]
    # zero out the masked pixels
    rotated_moving_image[np.logical_not(rotated_moving_mask)] = 0.0

    rotated_moving_fft = scipy.fft.fft2(rotated_moving_image)
    rotated_moving_mask_fft = scipy.fft.fft2(rotated_moving_mask.astype(float_dtype))

    # Calculate overlap of masks at every point in the convolution.
    # Locations with high overlap should not be taken into account.
    number_overlap_masked_px = scipy.fft.ifft2(rotated_moving_mask_fft * fixed_mask_fft).real
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)
    masked_correlated_fixed_fft = scipy.fft.ifft2(rotated_moving_mask_fft * fixed_fft).real
    masked_correlated_rotated_moving_fft = scipy.fft.ifft2(
        fixed_mask_fft * rotated_moving_fft).real
    
    numerator = scipy.fft.ifft2(rotated_moving_fft * fixed_fft).real
    numerator -= masked_correlated_fixed_fft * \
        masked_correlated_rotated_moving_fft / number_overlap_masked_px

    fixed_squared_fft = scipy.fft.fft2(np.square(reference))
    fixed_denom = scipy.fft.ifft2(rotated_moving_mask_fft * fixed_squared_fft).real
    fixed_denom -= np.square(masked_correlated_fixed_fft) / \
        number_overlap_masked_px
    fixed_denom[:] = np.fmax(fixed_denom, 0.0)

    rotated_moving_squared_fft = scipy.fft.fft2(np.square(rotated_moving_image))
    moving_denom = scipy.fft.ifft2(fixed_mask_fft * rotated_moving_squared_fft).real
    moving_denom -= np.square(masked_correlated_rotated_moving_fft) / \
        number_overlap_masked_px
    moving_denom[:] = np.fmax(moving_denom, 0.0)

    denom = np.sqrt(fixed_denom * moving_denom)

    tol = 1e3 * eps * np.max(np.abs(denom), keepdims=True)
    nonzero_indices = denom > tol

    # explicitly set out dtype for compatibility with SciPy < 1.4, where
    # fftmodule will be numpy.fft which always uses float64 dtype.
    cc = np.zeros_like(denom, dtype=float_dtype)
    cc[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
    np.clip(cc, a_min=-1, a_max=1, out=cc)

    # Apply overlap ratio threshold
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px,
                                                 keepdims=True)
    cc[number_overlap_masked_px < number_px_threshold] = 0.0

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
