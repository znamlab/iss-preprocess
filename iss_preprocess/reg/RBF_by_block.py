
from typing import Optional

import numpy as np
import numpy.typing as npt
from skimage.transform import AffineTransform, warp
from sklearn.linear_model import HuberRegressor, LinearRegression
from scipy.interpolate import RBFInterpolator

from image_tools.registration import phase_correlation as _pc

## Still in development, net yet implemented in the main pipeline

def find_RBF_by_block(
    reference: npt.NDArray,
    target: npt.NDArray,
    block_size: int = 256,
    overlap: float = 0.5,
    max_shift: Optional[int] = None,
    min_shift: int = 0,
    binarise_quantile: Optional[float] = None,
    correlation_threshold: Optional[float] = None,
    whiten: bool = True,
    max_residual: float = 2,
    min_valid_block_fraction: float = 0.1,
    min_valid_block_number: int = 3,
    debug: bool = False,
    stored: str = "centers_and_shifts",
):
    """Find affine transformation between two images by dividing them into blocks and
    estimating translation for each block.

    Shifts are estimated with phase correlation on each block and then we fit:
    x_coords = a_x * x + b_x * y + c_x
    y_coords = a_y * x + b_y * y + c_y

    The fit is done in two steps:
    1. Fit all shifts for which the correlation coefficient is above threshold with
    HuberRegressor.
    2. Exclude points with residuals above max_residual and refit with least squares.

    Notes that max_shift and min_shifts are applied to the shifts after the phase
    correlation, not to the fit (and the fit output can be outside of these limits).

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int, optional): size of the blocks, defaults to 256
        overlap (float, optional): fraction of overlap between blocks, defaults to 0.5
        max_shift (int, optional): maximum shift to consider, defaults to None
        min_shift (int, optional): minimum shift to consider, defaults to 0
        binarise_quantile (float, optional): quantile to use for binarisation,
            optional, defaults to None
        correlation_threshold (float, optional): minimum correlation threshold to
            include shift in the initial fit. If None, all shifts are included, defaults
            to None
        whiten (bool, optional): if True, perform phase rather than cross-correlation,
            defaults to True
        max_residual (float, optional): maximum residual to include shift in the final
            fit, defaults to 2
        min_valid_block_fraction (float, optional): minimum fraction of valid blocks to
            consider the fit valid, defaults to 0.1
        min_valid_block_number (int, optional): minimum number of valid blocks to
            consider the fit valid, defaults to 3
        debug (bool, optional): if True, return additional information, defaults to
            False

    Returns:
        params (np.array): parameters (a_x, b_x, c_x, a_y, b_y, c_y) of the affine
            transformation
        db (dict): dictionary with additional information if debug is True
    """

    # first perform phase correlation by block
    shifts, corr, centers = phase_correlation_by_block(
        reference,
        target,
        block_size=block_size,
        overlap=overlap,
        max_shift=max_shift,
        min_shift=min_shift,
        binarise_quantile=binarise_quantile,
        whiten=whiten,
    )
    shape = shifts.shape
    # then fit affine transformation to the shifts
    shifts = shifts.reshape(-1, 2)
    centers = centers.reshape(-1, 2)
    # shifts are in the form (row, col), but we need (x, y)
    # so we need to swap the columns
    shifts = shifts[:, [1, 0]]
    centers = centers[:, [1, 0]]

    if correlation_threshold is not None:
        corr = corr.reshape(-1)
        valid_shifts = shifts[corr > correlation_threshold]
        valid_centers = centers[corr > correlation_threshold]
        if not len(valid_shifts):
            raise ValueError("No valid shifts found")
    else:
        valid_shifts = shifts
        valid_centers = centers

    if stored == "centers_and_shifts":
        return valid_centers, valid_shifts
    

    # now fit RBF interpolator to the shifts
    x_RBFInterpolator = RBFInterpolator(valid_centers, valid_shifts[:, 0], kernel="thin_plate_spline")
    y_RBFInterpolator = RBFInterpolator(valid_centers, valid_shifts[:, 1], kernel="thin_plate_spline")


    params = np.hstack(
        [huber_x.coef_, huber_x.intercept_, huber_y.coef_, huber_y.intercept_]
    )

    # Now to improve the fit we will exclude the point with residuals above 2px and
    # refit with least squares
    residuals = np.abs(
        valid_shifts - (affine_transform(valid_centers, params) - valid_centers)
    ).sum(axis=1)
    nvalid = np.sum(residuals < max_residual)
    nblocks = len(residuals)
    threshold = max(min_valid_block_number, min_valid_block_fraction * nblocks)
    if nvalid < threshold:
        raise ValueError(
            f"Only {nvalid} shifts ({nvalid/nblocks*100:.2f}%)"
            + f" with residual below {max_residual}."
        )
    valid_shifts = valid_shifts[residuals < max_residual]
    valid_centers = valid_centers[residuals < max_residual]
    # Use scikit-learn LinearRegression just to have the same syntax
    leastsq_x = LinearRegression(fit_intercept=True).fit(
        valid_centers, valid_shifts[:, 0] + valid_centers[:, 0]
    )
    leastsq_y = LinearRegression(fit_intercept=True).fit(
        valid_centers, valid_shifts[:, 1] + valid_centers[:, 1]
    )
    params = np.hstack(
        [leastsq_x.coef_, leastsq_x.intercept_, leastsq_y.coef_, leastsq_y.intercept_]
    )

    if debug:
        db = dict(
            fit_x=leastsq_x,
            fit_y=leastsq_y,
            shifts=shifts,
            centers=centers,
            corr=corr,
            nblocks=shape,
            block_size=block_size,
            overlap=overlap,
            correlation_threshold=correlation_threshold,
        )
        return params, db
    return params


def transform_image(image: npt.NDArray, params: npt.NDArray, cval: float = 0):
    matrix = np.zeros((3, 3))
    # add x/ y transform
    matrix[0] = params[:3]
    matrix[1] = params[3:]
    # add the last row
    matrix[2] = [0, 0, 1]

    tform = AffineTransform(matrix=matrix)
    return warp(image, tform.inverse, preserve_range=True, cval=cval)


def affine_transform(point: npt.NDArray, params: npt.NDArray):
    """Affine transformation for a point or array of points.

    Transforms coordinates of point in the reference image to the coordinates in the
    target image.

    Args:
        point (np.array): point or array of points, (x, y) not (row, col)
        params (np.array): parameters of the affine transformation,
            (a_x, b_x, c_x, a_y, b_y, c_y)

    Returns:
        new_point (np.array): transformed point or array of points
    """
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    if point.ndim == 1:
        point = point.reshape(1, -1)
    (a_x, b_x, c_x, a_y, b_y, c_y) = params
    x_coords = a_x * point[:, 0] + b_x * point[:, 1] + c_x
    y_coords = a_y * point[:, 0] + b_y * point[:, 1] + c_y

    new_point = np.hstack(
        [
            x_coords.reshape(-1, 1),
            y_coords.reshape(-1, 1),
        ]
    )
    return new_point


def inverse_map(point: npt.NDArray, params: npt.NDArray):
    """Apply the inverse affine transformation for a point or array of points.

    Transforms coordinates of point in the target image to the coordinates in the
    reference image.

    Args:
        point (np.array): point or array of points, (x, y) not (row, col)
        params (np.array): parameters of the affine transformation,
            (a_x, b_x, c_x, a_y, b_y, c_y)

    Returns:
        new_point (np.array): transformed point or array of points
    """
    if not isinstance(point, np.ndarray):
        point = np.array(point)
    if point.ndim == 1:
        point = point.reshape(1, -1)
    # inverse the affine transformation
    """
    We fit:
    x_coords = a_x * x + b_x * y + c_x
    y_coords = a_y * x + b_y * y + c_y

    The inverse is:
    x = (x_coords - c_x)/a_x - b_x /a_x * y
    y = (y_coords - c_y)/b_y - a_y / b_y * x

    with Ax = (x_coords - c_x)/a_x and Bx = - b_x /a_x
    and Ay = (y_coords - c_y)/b_y and By = - a_y / b_y
    x = Ax + Bx . y
    y = Ay + By . x

    so we can solve for x:
    x = Ax + Bx(Ay + By * x)
    x = (Ax + Bx . Ay) / (1 - Bx.By)
    and for y:
    y = Ay + By(Ax + Bx * y)
    y = (Ay + By . Ax) / (1 - Bx.By)

    """
    (a_x, b_x, c_x, a_y, b_y, c_y) = params
    Ax = (point[:, 0] - c_x) / a_x
    Bx = -b_x / a_x
    Ay = (point[:, 1] - c_y) / b_y
    By = -a_y / b_y
    x = (Ax + Bx * Ay) / (1 - Bx * By)
    y = (Ay + By * Ax) / (1 - Bx * By)
    return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])


def phase_correlation_by_block(
    reference: npt.NDArray,
    target: npt.NDArray,
    block_size: int = 256,
    overlap: float = 0.1,
    max_shift: Optional[int] = None,
    min_shift: int = 0,
    binarise_quantile: Optional[float] = None,
    whiten: bool = True,
):
    """Estimate translation between two images by dividing them into blocks and
    estimating translation for each block.

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int, optional): size of the blocks, defaults to 256
        overlap (float, optional): fraction of overlap between blocks, defaults to 0.1
        max_shift (int, optional): maximum shift to consider, defaults to None
        min_shift (int, optional): minimum shift to consider, defaults to 0
        binarise_quantile (float, optional): quantile to use for binarisation,
            optional, defaults to None
        whiten (bool, optional): if True, perform phase rather than cross-correlation,
            defaults to True

    Returns:
        shifts (np.array): array of shifts row/col for each block
        corrs (np.array): array of correlation coefficients for each block
        block_centers (np.array): array of centers of each block

    """
    step_size = int(block_size * (1 - overlap))
    nblocks = np.array(reference.shape) // step_size
    block_centers = np.zeros((*nblocks, 2))
    shifts = np.zeros((*nblocks, 2))
    corrs = np.zeros(nblocks)
    for row in range(nblocks[0]):
        for col in range(nblocks[1]):
            ref_block = reference[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            target_block = target[
                row * step_size : row * step_size + block_size,
                col * step_size : col * step_size + block_size,
            ]
            if binarise_quantile is not None:
                ref_block = ref_block > np.quantile(ref_block, binarise_quantile)
                target_block = target_block > np.quantile(
                    target_block, binarise_quantile
                )
            shift, corr = _pc.phase_correlation(
                ref_block.astype(reference.dtype),
                target_block.astype(target.dtype),
                max_shift=max_shift,
                min_shift=min_shift,
                whiten=whiten,
            )[:2]
            shifts[row, col] = shift
            corrs[row, col] = corr
            block_centers[row, col] = np.array(
                [row * step_size + block_size // 2, col * step_size + block_size // 2]
            )
    return shifts, corrs, block_centers
