from tifffile import TiffWriter
import cv2
import numpy as np
import yaml
from pathlib import Path


def write_stack(stack, fname, bigtiff=False, dtype="uint16", clip=True):
    """
    Write a stack to file as a multipage TIFF

    Args:
        stack (numpy.ndarray): X x Y x ... array (can have multiple channels /
            zplanes, etc.)
        fname (str): save path for the TIFF
        dtype (str): datatype of the output image. Default to 'uint16'
        clip (bool): clip negative values before convertion
    """
    stack = stack.reshape((stack.shape[0], stack.shape[1], -1))
    if clip:
        stack = np.clip(stack, 0)

    with TiffWriter(fname, bigtiff=bigtiff) as tif:
        for frame in range(stack.shape[2]):
            tif.write(stack[:, :, frame].astype(dtype), contiguous=True)


def save_ome_tiff_pyramid(
    target,
    image,
    pixel_size,
    subresolutions=3,
    max_size=None,
    dtype="uint16",
    verbose=True,
    save_thumbnail=False
):
    """Write single image plane as pyramidal ome-tiff

    Args:
        target (str): Path to tif file
        image (np.array): 2D array with 8-bit image data
        pixel_size (float): Pixel size in microns
        subresolutions (int, optional): Number of pyramid levels. Defaults to 3.
        max_size (float, optional): Pixel size for the biggest layer. Data will be
            downsampled to that size first and then by a factor 2 every level. Defaults
            to None.
        dtype (str, optional): Image datatype, can be "uint16" or "uint8".
            Defaults to "uint16".
        verbose (bool, optional): Print progress. Defaults to True.
        save_thumbnail (bool, optional): Add a thumbnail image. Defaults to False.

    Returns:
        np.array: Last level of the pyramid, most downsampled image
    """

    if dtype not in ["uint8", "uint16"]:
        raise NotImplementedError("`dtype` must be uint8 or uint16")
    nbits = int(dtype[4:])
    max_val = 2**nbits - 1
    if verbose:
        print("... Clipping array")
    image = np.clip(image, 0, max_val).astype(dtype)  # clip to avoid overflow

    metadata = {
        "axes": "YX",
        "SignificantBits": nbits,
        "PhysicalSizeX": pixel_size,
        "PhysicalSizeXUnit": "Âµm",
        "PhysicalSizeY": pixel_size,
        "PhysicalSizeYUnit": "Âµm",
    }

    with TiffWriter(target, bigtiff=True) as tif:
        options = dict(
            photometric="minisblack", tile=(128, 128), resolutionunit="CENTIMETER"
        )
        if verbose:
            print("... writing full res image")
        tif.write(
            image,
            subifds=subresolutions,
            resolution=(1e4 / pixel_size, 1e4 / pixel_size),
            metadata=metadata,
            **options
        )
        for level in range(subresolutions):
            if verbose:
                print("... writing pyramidal layer %d" % (level + 1))

            mag = 2 ** (level + 1)
            image = cv2.resize(
                image,
                (image.shape[1] // 2, image.shape[0] // 2),
                interpolation=cv2.INTER_LINEAR,
            )
            tif.write(
                image,
                resolution=(1e4 / mag / pixel_size, 1e4 / mag / pixel_size),
                **options
            )
        if max(image.shape) < 200:
            skip = 1
        else:
            skip = int(max(image.shape) / 100)
        if save_thumbnail:
            if dtype == "uint16":
                # thumbnail are always uint8
                thumbnail = (image[::skip, ::skip] >> 2).astype("uint8")
            else:
                thumbnail = image[::skip, ::skip]
            # >> 2 if to shift bits before conversion to int8
            tif.write(thumbnail, metadata={"Name": "thumbnail"})
    return image
