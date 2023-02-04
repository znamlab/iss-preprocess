import numpy as np
from pathlib import Path
import cv2
from sklearn.mixture import GaussianMixture
from skimage.exposure import match_histograms
from skimage.morphology import disk
from skimage.filters import median
from ..io.load import load_stack, load_ops
from ..coppafish import hanning_diff
from flexiznam.config import PARAMETERS
from pathlib import Path


def apply_illumination_correction(data_path, stack, prefix):
    """Apply illumination correction

    Use precomputed normalised and filtered averages to correct for inhomogeneous
    illumination

    Args:
        data_path (str): Relative path to the data to read ops and find averages
        stack (np.array): A 3 or 4D array with X x Y x Nchannels as first 3 dimensions
        prefix (str): Prefix name of the average, e.g. "barcode_round" for grand average
            or "barcode_round_1" for single round average.


    Returns:
        stack (np.array): Normalised stack. Same shape as input.
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    average_image_fname = (
        processed_path / data_path / "averages" / f"{prefix}_average.tif"
    )
    average_image = load_stack(average_image_fname).astype(float)
    average_image = (
        average_image / np.max(average_image, axis=(0, 1))[np.newaxis, np.newaxis, :]
    )
    if stack.ndim == 4:
        stack = (
            stack - ops["black_level"][np.newaxis, np.newaxis, :, np.newaxis]
        ) / average_image[:, :, :, np.newaxis]
    else:
        stack = (stack - ops["black_level"][np.newaxis, np.newaxis, :]) / average_image
    return stack


def filter_stack(stack, r1=2, r2=4):
    """Filter stack with hanning window

    Convolve each image from the stack with a hanning kernel a la coppafish. The kernel
    is the sum of a negative outer circle and a positive inner circle.

    Args:
        stack (np.array): Stack to filter, either X x Y x Ch or X x Y x Ch x Round
        r1 (int, optional): Radius in pixels of central positive hanning convolve
            kernel. Defaults to 2.
        r2 (int, optional): Radius in pixels of outer negative hanning convolve kernel.
            Defaults to 4.

    Returns:
        np.array: Filtered stack.
    """
    nchannels = stack.shape[2]
    h = hanning_diff(r1, r2)
    stack_filt = np.zeros(stack.shape)

    for ich in range(nchannels):
        if stack.ndim == 4:
            nrounds = stack.shape[3]
            for iround in range(nrounds):
                stack_filt[:, :, ich, iround] = cv2.filter2D(
                    stack[:, :, ich, iround].astype(float),
                    -1,
                    np.flip(h),
                    borderType=cv2.BORDER_REPLICATE,
                )
        else:
            stack_filt[:, :, ich] = cv2.filter2D(
                stack[:, :, ich].astype(float),
                -1,
                np.flip(h),
                borderType=cv2.BORDER_REPLICATE,
            )
    return stack_filt


# AB: Reviewed 10/01/23
def analyze_dark_frames(fname):
    """
    Get statistics of dark frames to use for black level correction

    Args:
        fname (str): path to dark frame TIFF file

    Returns:
        numpy.array: Average black level per channel
        numpy.array: Readout noise per channel

    """
    dark_frames = load_stack(fname)
    return dark_frames.mean(axis=(0, 1)), dark_frames.std(axis=(0, 1))


def tilestats_and_mean_image(
    data_folder,
    prefix="",
    suffix="",
    black_level=0,
    max_value=10000,
    verbose=False,
    median_filter=None,
    normalise=False,
    combine_tilestats=False,
):
    """
    Compute tile statistics and mean image to use for illumination correction.

    Args:
        data_folder (str): directory containing images
        prefix (str, optional): prefix to filter images to average. Defaults to "", no
            filter
        suffix (str, optional): suffix to filter images to average. Defaults to "", no
            filter
        black_level (float, optional): image black level to subtract before calculating
            each mean image. Defaults to 0
        max_value (float, optional): image values are clipped to this value *after*
            black level subtraction. This reduces the effect of extremely bright
            features skewing the average image. Defaults to 10000.
        verbose (bool, optional): whether to report on progress. Defaults to False
        median_filter (int, optional): size of median filter to apply to the correction
            image. If None, no median filtering is applied. Defaults to None.
        normalise (bool, optional): Divide each channel by its maximum value. Default to
            False
        combine_tilestats (bool, optional): If False, compute tilestats, if True, load
            already created tilestats for each tif and sum them.

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
    if not len(tiffs):
        raise IOError("NO valid tifs in folder %s" % data_folder, flush=True)

    if verbose:
        print("Averaging {0} tifs in {1}.".format(len(tiffs), im_name), flush=True)

    data = load_stack(tiffs[0])
    assert data.ndim == 3
    if combine_tilestats:
        stats = tiffs[0].with_name(
            tiffs[0].name.replace("_average.tif", "_tilestats.npy")
        )
        if not stats.exists():
            raise IOError(
                "Tilestats files must exist to use `combine_tilestats`."
                + f"\n{stats} does not exists"
            )
        tilestats = np.load(stats)
    else:
        tilestats = compute_distribution(data)

    black_level = np.asarray(black_level)  # in case we have just a float

    # initialise folder mean with first frame
    mean_image = np.array(data, dtype=float)
    mean_image = np.clip(mean_image - black_level.reshape(1, 1, -1), 0, max_value)
    mean_image /= len(tiffs)

    for itile, tile in enumerate(tiffs[1:]):  # processing the rest of the tiffs
        if verbose and not (itile % 10):
            print("   ...{0}/{1}.".format(itile + 1, len(tiffs)), flush=True)

        data = load_stack(tile)
        if combine_tilestats:
            stats = tile.with_name(tile.name.replace("_average.tif", "_tilestats.npy"))
            tilestats += np.load(stats)
        else:
            tilestats += compute_distribution(data)

        data = np.clip(data.astype(float) - black_level.reshape(1, 1, -1), 0, max_value)
        mean_image += data / len(tiffs)

    if median_filter is not None:
        for ic in range(mean_image.shape[2]):
            mean_image[:, :, ic] = median(mean_image[:, :, ic], disk(median_filter))

    if normalise:
        max_by_chan = np.nanmax(mean_image.reshape((-1, mean_image.shape[-1])), axis=0)
        mean_image /= max_by_chan.reshape((1, 1, -1))

    return mean_image, tilestats


def compute_distribution(stack, max_value=int(2**12 + 1)):
    """Compute simple tile statistics for one multichannel image

    Args:
        stack (np.array): An X x Y x Nch stack
        max_value (int): Maximum value to compute histogram

    Returns:
        np.array: Distribution of pixel values by channel. Shape (max_value + 1 , Nch)
    """
    distribution = np.zeros((max_value + 1, stack.shape[2]))
    for ich in range(stack.shape[2]):
        distribution[:, ich] = np.bincount(
            stack[:, :, ich].flatten().astype(np.uint16),
            minlength=max_value + 1,
        )
    return distribution
