import numpy as np
import glob
import os
import cv2
from sklearn.mixture import GaussianMixture
from skimage.exposure import match_histograms
from skimage.morphology import disk
from skimage.filters import median
from ..io.load import load_stack
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
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
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
    # reshape to get max/std accross all pixels for each channel
    return dark_frames.mean(axis=(0, 1)), dark_frames.std(axis=(0, 1))


def compute_mean_image(
    dir,
    suffix=None,
    black_level=0,
    max_value=1000,
    verbose=False,
    median_filter=None,
    normalise=False,
):
    """
    Compute mean image to use for illumination correction.

    Args:
        dir (str): directory containing images
        suffix (str): subdirectory inside each of `dirs` containing images
        black_level (float): image black level to subtract before calculating
            each mean image. Default to 0
        max_value (float): image values are clipped to this value. This reduces
            the effect of extremely bright features skewing the average image. Default
            to 1000.
        verbose (bool): whether to report on progress
        median_filter (int): size of median filter to apply to the correction image.
            If None, no median filtering is applied.
        normalise (bool): Divide each channel by its maximum value. Default to False


    Returns:
        numpy.ndarray correction image

    """

    if suffix:
        subdir = os.path.join(dir, suffix)
    else:
        subdir = dir
    im_name = os.path.split(dir)[1]
    tiffs = glob.glob(subdir + "/*.tif")
    if verbose:
        print("Averaging {0} tifs in {1}.".format(len(tiffs), im_name))

    data = load_stack(tiffs[0])

    # initialise folder mean with first frame
    mean_image = np.array(data, dtype=float)
    mean_image = np.clip(mean_image, None, max_value) - black_level
    mean_image /= len(tiffs)
    for itile, tile in enumerate(tiffs[1:]):  # processing the rest of the tiffs
        if verbose and not (itile % 10):
            print("   ...{0}/{1}.".format(itile + 1, len(tiffs)))
        data = np.array(load_stack(tile), dtype=float)
        data = np.clip(data, None, max_value) - black_level
        mean_image += data / len(tiffs)

    if median_filter is not None:
        mean_image = median(mean_image, disk(median_filter))

    if normalise:
        max_by_chan = np.nanmax(mean_image.reshape((-1, mean_image.shape[-1])), axis=0)
        mean_image /= max_by_chan.reshape((1, 1, -1))

    return mean_image


def correct_offset(tiles, method="metadata", metadata=None, n_components=5):
    """
    Estimate image offset for each channel as the minimum value or using a
    Gaussian mixture model and substract it from input images.

    Args:
        tiles (DataFrame): individual tiles
        method (str): method for determining the offset, one of either:
            `metadata`: uses the values recorded in the image metadata
            `min`: uses the minimum for each channel
            `gmm`: fits a Gaussian mixture model and uses the smallest mean
        metadata (ElementTree): XML element tree with

    """
    if metadata:
        channels_metadata = metadata.findall(
            "./Metadata/Information/Image/Dimensions/Channels/Channel"
        )

    channels = tiles.C.unique()
    for channel in channels:
        this_channel = tiles[(tiles["C"] == channel) & (tiles["Z"] == 0)]["data"]
        # Creating ragged nested ndarrays is deprecated. Suggested fix is to make dtype=object
        data = np.concatenate(this_channel.to_numpy(), dtype=object).reshape(-1, 1)
        if method == "metadata" and metadata:
            offset = float(
                channels_metadata[channel].find("./DetectorSettings/Offset").text
            )
        elif method == "min":
            offset = np.min(data)
        else:
            gm = GaussianMixture(n_components=n_components, random_state=0).fit(
                data[:10:, :]
            )
            offset = np.min(gm.means_)
        v = tiles[tiles["C"] == channel]["data"].transform(
            lambda x: x.astype(float) - offset
        )
        tiles.update(v)
    return tiles


def correct_levels(stacks, reference, method="histogram"):
    """
    Correct illumination levels of an image using a selected method.

    Args:
        stacks (list): list of X x Y x Z stacks to correct
        reference (numpy.ndarray): image to use as a template for correction
        method (str): correction method, one of:
            'histogram': match histograms
            'mean': match mean level
            'median': match median level
            'minmax': match minimum and maximum levels

    Returns:
        List of X x Y x Z stacks after correction
    """
    corrected_stacks = []
    reference_mean = np.mean(reference)
    reference_median = np.median(reference)
    reference_min = np.min(reference)
    reference_max = np.max(reference)
    reference_scale = reference_max - reference_min

    for stack in stacks:
        corrected_stack = np.empty(stack.shape)
        nchannels = stack.shape[2]
        for channel in range(nchannels):
            if method == "histogram":
                corrected_stack[:, :, channel] = match_histograms(
                    stack[:, :, channel], reference
                )
            elif method == "mean":
                corrected_stack[:, :, channel] = (
                    stack[:, :, channel]
                    / np.mean(stack[:, :, channel])
                    * reference_mean
                )
            elif method == "median":
                corrected_stack[:, :, channel] = (
                    stack[:, :, channel]
                    / np.median(stack[:, :, channel])
                    * reference_median
                )
            elif method == "minmax":
                im_min = np.min(stack[:, :, channel])
                im_max = np.max(stack[:, :, channel])
                corrected_stack[:, :, channel] = reference_min + reference_scale * (
                    stack[:, :, channel] - im_min
                ) / (im_max - im_min)
            else:
                raise (ValueError(f'Unknown correction method "{method}"'))
        corrected_stacks.append(corrected_stack)
    return corrected_stacks
