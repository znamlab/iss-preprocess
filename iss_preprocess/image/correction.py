from functools import partial
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import disk
from sklearn.linear_model import LinearRegression

from ..coppafish import hanning_diff
from ..io import get_processed_path, load_ops, load_stack


def apply_illumination_correction(
    data_path, stack, prefix, projection=None, dtype=float
):
    """Apply illumination correction

    Use precomputed normalised and filtered averages to correct for inhomogeneous
    illumination

    Args:
        data_path (str): Relative path to the data to read ops and find averages
        stack (np.array): A 3 or 4D array with X x Y x Nchannels as first 3 dimensions
        prefix (str): Prefix name of the average, e.g. "barcode_round" for grand average
            or "barcode_round_1" for single round average.
        projection (str, optional): Name of the projection to use. If None, will try to
            get it from ops. Defaults to None.
        dtype (str or type, optional): data type of the output. Division is always
            performed as float

    Returns:
        stack (np.array): Normalised stack. Same shape as input.

    """
    raise DeprecationWarning(
        "This function is deprecated. Correction is done in load_tile_by_coors instead"
    )
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    fname = ops.get(f"{prefix}_average_for_correction", f"{prefix}_average.tif")
    average_image_fname = processed_path / "averages" / fname
    avg_image = load_stack(average_image_fname).astype(float)

    # find rounds that do not have data
    no_data = np.logical_not(np.any(stack, axis=(0, 1, 2)))
    if stack.ndim == 4:
        corrected = (
            stack - ops["black_level"][np.newaxis, np.newaxis, :, np.newaxis]
        ) / avg_image[:, :, :, np.newaxis]
        corrected[..., no_data] *= 0
    elif no_data:
        corrected = stack
    else:
        corrected = (stack - ops["black_level"][np.newaxis, np.newaxis, :]) / avg_image
    return corrected.astype(dtype)


def filter_stack(stack, r1=2, r2=4, dtype=float):
    """Filter stack with hanning window

    Convolve each image from the stack with a hanning kernel a la coppafish. The kernel
    is the sum of a negative outer circle and a positive inner circle.

    Args:
        stack (np.array): Stack to filter, either X x Y x Ch or X x Y x Ch x Round
        r1 (int, optional): Radius in pixels of central positive hanning convolve
            kernel. Defaults to 2.
        r2 (int, optional): Radius in pixels of outer negative hanning convolve kernel.
            Defaults to 4.
        dtype (str, optional): Datatype for performing the computation

    Returns:
        np.array: Filtered stack.

    """

    h = hanning_diff(r1, r2).astype(dtype)
    stack_filt = np.zeros(stack.shape, dtype=dtype)
    filt_func = partial(
        cv2.filter2D, ddepth=-1, kernel=np.flip(h), borderType=cv2.BORDER_REPLICATE
    )
    # TODO: check if we can get rid of the np.flip
    if stack.ndim == 2:
        stack_filt = filt_func(stack.astype(dtype))
        return stack_filt

    nchannels = stack.shape[2]
    for ich in range(nchannels):
        if stack.ndim == 4:
            nrounds = stack.shape[3]
            for iround in range(nrounds):
                stack_filt[:, :, ich, iround] = filt_func(
                    stack[:, :, ich, iround].astype(dtype)
                )
        else:
            stack_filt[:, :, ich] = filt_func(stack[:, :, ich].astype(dtype))
    return stack_filt


def calculate_unmixing_coefficient(
    signal_image,
    background_image,
    background_coef,
    threshold_background=None,
    threshold_signal=None,
    fit_intercept=True,
):
    """
    Unmixes two images: one with only background autofluorescence and another with both
    background and useful signal. Uses Linear regression for the unmixing process.

    Args:
        background_image: numpy array of the background image.
        signal_image: numpy array of the image with both signal and background.
        background_coef: Coefficient to multiply the background image by before
            subtraction.
        threshold_background (optional): Minimum value for a pixel to be considered
            background. If None will use the median. Default None.
        threshold_signal (optional): Minimum value on signal channel for a pixel to be
            considered background. If None will use the median. Default None.
        fit_intercept (optional): Whether to fit an intercept in the linear model.
            Default True.


    Returns:
        pure_signal_image: The isolated signal image after background subtraction.
        model_coef: Coefficient used to multiply the background image for unmixing.
        intercept: Intercept of the linear model used for unmixing.
        valid_pixel: Boolean array of pixels that passed the thresholding.

    """
    # Flatten to 1D arrays for the regression model
    background_flat = background_image.ravel()
    mixed_signal_flat = signal_image.ravel()

    # Remove pixels that are too dark or too bright
    # The max pixel value is 4096, remove only very close to saturation
    valid_pixel = (background_flat < 4090) & (mixed_signal_flat < 4090)
    valid_pixel &= background_flat > 0  # do that before for the median
    valid_pixel &= mixed_signal_flat > 0  # do that before for the median
    if threshold_signal is None:
        threshold_signal = np.nanmedian(mixed_signal_flat[valid_pixel])
    valid_pixel &= mixed_signal_flat > threshold_signal
    if threshold_background is None:
        threshold_background = np.nanmedian(background_flat[valid_pixel])
    valid_pixel &= background_flat > threshold_background

    # also remove pixels that are pure signal
    valid_pixel &= mixed_signal_flat < 20 * background_flat
    background_flat = background_flat[valid_pixel].reshape(-1, 1)
    mixed_signal_flat = mixed_signal_flat[valid_pixel]

    print("Parameters for unmixing:")
    print(f"Threshold background: {threshold_background}")
    print(f"Threshold signal: {threshold_signal}")
    print(f"Number of valid pixels: {valid_pixel.sum()}/{valid_pixel.size}")

    # Initialize and fit Linear model
    model = LinearRegression(positive=True, fit_intercept=fit_intercept)
    try:
        model.fit(background_flat, mixed_signal_flat)
        # Predict the background component in the mixed signal image
        predicted_background_flat = model.predict(
            background_image.ravel().reshape(-1, 1)
        )

        predicted_background = predicted_background_flat.reshape(background_image.shape)
        # TODO: Remove `background_coef`, the fudge factor
        pure_signal_image = signal_image - (predicted_background * background_coef)
        pure_signal_image = np.clip(pure_signal_image, 0, None)
        print(
            f"Image unmixed with coefficient: {model.coef_[0]:.2f},"
            + f" intercept: {model.intercept_:.2f}"
        )
    except ValueError:
        raise ValueError("Not enough data passing background threshold to fit model")

    return pure_signal_image, model.coef_[0], model.intercept_, valid_pixel


def unmix_images(
    background_image: np.ndarray,
    mixed_signal_image: np.ndarray,
    coef: float,
    intercept: float,
    background_coef: float = 1.0,
    offset: float = 10.0,
):
    """
    Unmixes two images

    One must contain only background autofluorescence and another with both
    background and useful signal.

    Args:
        background_image: numpy array of the background image.
        mixed_signal_image: numpy array of the image with both signal and background.
        coef: Coefficient to multiply the background image by before subtraction.
        intercept: Intercept of the linear model used for unmixing.
        background_coef: Fudge factor to increase the amount of background subtracted.
            Default 1.0.
        offset: Small value to add to the signal image to avoid negative values after
            subtraction. Default 10.0.

    Returns:
        signal_image: The isolated signal image after background subtraction.
    """
    predicted_background = (background_image * float(coef)) + float(intercept)
    signal_image = mixed_signal_image - (predicted_background * background_coef)
    # Clipping removes negative values, which can be an issue if the coefficient is
    # a bit off, add a small value to avoid this
    signal_image += offset
    signal_image = np.clip(signal_image, 0, None)
    return signal_image


def tilestats_and_mean_image(
    data_folder,
    prefix="",
    suffix="",
    n_batch=1,
    black_level=0,
    max_value=10000,
    verbose=False,
    median_filter_size=None,
    gaussian_filter_size=None,
    normalise=False,
    combine_tilestats=False,
    exclude_tiffs=None,
    row_filter=None,
):
    """
    Compute tile statistics and mean image to use for illumination correction.

    Args:
        data_folder (str): directory containing images
        prefix (str, optional): prefix to filter images to average. Defaults to "", no
            filter
        suffix (str, optional): suffix to filter images to average. Defaults to "", no
            filter
        n_batch (int, optional): If 1 average everything, otherwise makes `n_batch`
            averages and take the median of those. All averages must fit in RAM. If
            `None`, create as many batches as tiffs. Defaults to 1.
        black_level (float, optional): image black level to subtract before calculating
            each mean image. Defaults to 0
        max_value (float, optional): image values are clipped to this value *after*
            black level subtraction. This reduces the effect of extremely bright
            features skewing the average image. Defaults to 10000.
        verbose (bool, optional): whether to report on progress. Defaults to False
        median_filter (int, optional): size of median filter to apply to the correction
            image. If None, no median filtering is applied. Defaults to None.
        mean_filter (int, optional): size of mean filter to apply to the correction
            image. If None, no mean filtering is applied. Defaults to None.
        normalise (bool, optional): Divide each channel by its maximum value. Default to
            False
        combine_tilestats (bool, optional): If False, compute tilestats, if True, load
            already created tilestats for each tif and sum them.
        exclude_tiffs (list, optional): List of str filter to exclude tiffs from average
        row_filter (str, optional): If "even"/"odd", only average tiffs on even/odd rows

    Returns:
        numpy.ndarray: correction image
        dict: tile statistics of clipped and black subtracted images

    """
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ""
    data_folder = Path(data_folder)
    im_name = data_folder.name
    filt = f"{prefix}*{suffix}.tif"
    tiffs = list(data_folder.glob(filt))
    if exclude_tiffs is not None:
        tiffs = [t for t in tiffs if not any([f in t.name for f in exclude_tiffs])]
    if not len(tiffs):
        raise IOError(f"NO valid tifs in folder {data_folder}")
    # Filter tiffs to only even or odd numbered tiffs
    if row_filter == "even":
        # Search tif for even number in
        tiffs = [t for t in tiffs if int(t.stem.split("_")[-1]) % 2 == 0]
    elif row_filter == "odd":
        tiffs = [t for t in tiffs if int(t.stem.split("_")[-1]) % 2 == 1]

    black_level = np.asarray(black_level)  # in case we have just a float
    if verbose:
        print(f"Averaging {len(tiffs)} tifs in {im_name}.", flush=True)
    if n_batch is None:
        n_batch = len(tiffs)

    if n_batch == 1:
        mean_image, tilestats = _mean_tiffs(
            tiffs,
            black_level,
            max_value,
            verbose,
            median_filter_size,
            gaussian_filter_size,
            normalise,
            combine_tilestats,
        )
    else:
        means = []
        tilestats = None
        n_by_batch = int(np.floor(len(tiffs) / n_batch))
        if verbose:
            print(
                f"Averaging {n_batch} batch of {n_by_batch} tifs in {im_name}.",
                flush=True,
            )
        for ib in range(n_batch):
            print(f"Batch {ib+1} / {n_batch}", flush=True)
            batch = tiffs[n_by_batch * ib : min(n_by_batch * (ib + 1), len(tiffs))]
            batch_mean, batch_stats = _mean_tiffs(
                batch,
                black_level,
                max_value,
                verbose,
                median_filter_size,
                gaussian_filter_size,
                normalise,
                combine_tilestats,
            )
            if tilestats is None:
                tilestats = batch_stats
            else:
                tilestats += batch_stats
            means.append(batch_mean)
        means = np.stack(means, axis=3)
        mean_image = np.nanmedian(means, axis=3)
    return mean_image, tilestats


def _mean_tiffs(
    tiff_list,
    black_level,
    max_value,
    verbose,
    median_filter_size,
    gaussian_filter_size,
    normalise,
    combine_tilestats,
):
    """Inner function of tilestats_and_mean_image. See parent functions for docstring"""
    data = load_stack(tiff_list[0])
    assert data.ndim == 3
    if combine_tilestats:
        stats = tiff_list[0].with_name(
            tiff_list[0].name.replace("_average.tif", "_tilestats.npy")
        )
        if not stats.exists():
            raise IOError(
                "Tilestats files must exist to use `combine_tilestats`."
                + f"\n{stats} does not exists"
            )
        tilestats = np.load(stats)
    else:
        tilestats = compute_distribution(data, max_value=2**16)

    # initialise folder mean with first frame
    mean_image = np.array(data, dtype=float)
    mean_image = np.clip(mean_image - black_level.reshape(1, 1, -1), 0, max_value)
    mean_image /= len(tiff_list)

    for itile, tile in enumerate(tiff_list[1:]):  # processing the rest of the tiffs
        if verbose and not (itile % 10):
            print(f"   ...{itile + 1}/{len(tiff_list)}.", flush=True)

        data = load_stack(tile)
        if combine_tilestats:
            stats = tile.with_name(tile.name.replace("_average.tif", "_tilestats.npy"))
            tilestats += np.load(stats)
        else:
            tilestats += compute_distribution(data, max_value=2**16)

        data = np.clip(data.astype(float) - black_level.reshape(1, 1, -1), 0, max_value)
        mean_image += data / len(tiff_list)

    if median_filter_size is not None:
        mean_image = median_filter(
            mean_image, footprint=disk(median_filter_size), axes=(0, 1)
        )
    if gaussian_filter_size is not None:
        mean_image = gaussian_filter(
            mean_image, sigma=gaussian_filter_size, mode="nearest", axes=(0, 1)
        )

    if normalise:
        max_by_chan = np.nanmax(mean_image.reshape((-1, mean_image.shape[-1])), axis=0)
        mean_image /= max_by_chan.reshape((1, 1, -1))
    return mean_image, tilestats


def compute_distribution(stack, max_value=int(2**12 + 1)):
    """Compute simple tile statistics for one multichannel image

    Args:
        stack (np.array): An X x Y x Nch stack
        max_value (int): Maximum value to compute histogram, must be >= stack.max()

    Returns:
        np.array: Distribution of pixel values by channel. Shape (max_value + 1 , Nch)

    """
    distribution = np.zeros((max_value + 1, stack.shape[2]))
    for ich in range(stack.shape[2]):
        distribution[:, ich] = np.bincount(
            stack[:, :, ich].flatten().astype(np.uint16), minlength=max_value + 1
        )
    return distribution
