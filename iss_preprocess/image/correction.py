import numpy as np
import glob
import os
from tifffile import TiffFile
from sklearn.mixture import GaussianMixture
from skimage.exposure import match_histograms
from skimage.morphology import disk
from skimage.filters import median
from ..io import get_tiles_micromanager
from ..coppafish import hanning_diff
import cv2


def filter_stack(stack, r1=2, r2=4):
    nchannels, nrounds = stack.shape[2:]

    h = hanning_diff(r1, r2)
    stack_filt = np.zeros(stack.shape)

    for ich in range(nchannels):
        for iround in range(nrounds):
            stack_filt[:, :, ich, iround] = cv2.filter2D(
                stack[:, :, ich, iround].astype(float),
                -1,
                np.flip(h),
                borderType=cv2.BORDER_REPLICATE,
            )
    return stack_filt


def analyze_dark_frames(fname):
    """
    Get statistics of dark frames to use for black level correction

    Args:
        fname (str): path to dark frame TIFF file

    Returns:
        numpy.array: Average black level per channel
        numpy.array: Readout noise per channel

    """
    images = []
    with TiffFile(fname) as stack:
        for page in stack.pages:
            images.append(page.asarray())

    dark_frames = np.stack(images, axis=0)

    # reshape to get max/std accross all pixels for each channel
    dark_frames = dark_frames.reshape((-1, dark_frames.shape[-1]))
    return dark_frames.mean(axis=0), dark_frames.std(axis=0)


def compute_mean_image(
    dirs,
    tile_shape,
    suffix=None,
    black_level=300,
    max_value=1000,
    verbose=False,
    median_filter=None,
):
    """
    Compute mean image to use for illumination correction.

    Args:
        dirs (list): list of directories containing images
        tile_shape (tuple): shape of image tiles
        suffix (str): subdirectory inside each of `dirs` containing images
        black_level (float): image black level to subtract before calculating
            each mean image
        max_value (float): image values are clipped to this value. This reduces
            the effect of extremely bright features skewing the average image
        verbose (bool): whether to report on progress
        median_filter (int): size of median filter to apply to the correction image.
            If None, no median filtering is applied.

    Returns:
        numpy.ndarray correction image

    """
    mean_image = np.zeros(tile_shape)
    for dir in dirs:
        if suffix:
            subdir = os.path.join(dir, suffix)
        else:
            subdir = dir
        im_name = os.path.split(dir)[1]
        if verbose:
            print(im_name)
        tiffs = glob.glob(subdir + "/*.tif")
        tiles = get_tiles_micromanager(tiffs)
        this_mean_image = np.zeros(tiles.iloc[0]["data"].shape)
        for _, tile in tiles.iterrows():
            data = tile["data"]
            data[data > max_value] = max_value
            data = data - black_level
            this_mean_image += data
        this_mean_image = this_mean_image / np.max(this_mean_image)
        mean_image += this_mean_image
    if median_filter is not None:
        correction_image = median(mean_image, disk(median_filter))
    correction_image = correction_image / np.max(correction_image)
    return correction_image


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
