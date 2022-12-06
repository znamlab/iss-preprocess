import numpy as np
import czifile
import pandas as pd
import xml.etree.ElementTree as ET
from tifffile import TiffFile
import json


def load_stack(fname):
    stack = TiffFile(fname)
    ims = []
    for page in stack.pages:
        ims.append(page.asarray())
    im_calibration = np.stack(ims, axis=2)
    return im_calibration


def get_tile_ome(fname, fmetadata):
    """
    Load OME TIFF tile.

    Args:
        fname (str): path to OME TIFF
        fmetadata (str): path to OME metadata file

    Returns:
        numpy.ndarray: X x Y x C x Z z-stack.

    """
    stack = TiffFile(fname)

    with open(fmetadata) as json_file:
        metadata = json.load(json_file)
    frame_keys = list(metadata.keys())[1:]

    zs = [metadata[frame_key]["ZPositionUm"] for frame_key in frame_keys]
    zs = sorted(list(set(zs)))
    channels = [int(metadata[frame_key]["Camera"][-1]) for frame_key in frame_keys]
    channels = sorted(list(set(channels)))
    nch = len(channels)
    nz = len(zs)
    xpix = stack.pages[0].tags["ImageWidth"].value
    ypix = stack.pages[0].tags["ImageLength"].value
    im = np.zeros((ypix, xpix, nch, nz))

    for page, frame_key in zip(stack.pages, frame_keys):
        z = zs.index(metadata[frame_key]["ZPositionUm"])
        ch = int(
            metadata[frame_key]["Camera"][-1]
        )  # channel id is the last digit of camera name
        im[:, :, ch, z] = page.asarray()

    return im


def get_tiles_micromanager(fnames, ch=0):
    """
    Load tiles from Micromanager TIFFs and return a nice DataFrame including tile
    coordinates

    Args:
        fnames (list): list of micromanager TIFF files

    Returns:
        pandas.DataFrame containing tile data.

    """
    page_dicts = []

    for fname in fnames:
        with TiffFile(fname) as stack:
            for page in stack.pages:
                page_dict = {}
                page_dict["data"] = page.asarray()
                page_dict["X"] = page.tags["MicroManagerMetadata"].value[
                    "XPosition_um_Intended"
                ]
                page_dict["Y"] = page.tags["MicroManagerMetadata"].value[
                    "YPosition_um_Intended"
                ]
                page_dict["Z"] = page.tags["MicroManagerMetadata"].value[
                    "ZPosition_um_Intended"
                ]
                page_dict["C"] = ch
                page_dicts.append(page_dict)

    df = pd.DataFrame.from_dict(page_dicts)
    return df


def get_tiles(fname):
    """
    Load tiles from CZI image file and return a nice DataFrame including tile
    coordinates and image metadata.

    Args:
        fname (str): path to CZI file

    Returns:
        pandas.DataFrame containing tile data.
        xml.etree.ElementTree with metadata.

    """
    with czifile.CziFile(fname, detectmosaic=False) as stack:
        subblock_dicts = []

        for subblock in stack.filtered_subblock_directory:
            subblock_dict = {}
            for dimension_entry in subblock.dimension_entries:
                subblock_dict[dimension_entry.dimension] = dimension_entry.start
            subblock_dict["data"] = (
                subblock.data_segment().data().squeeze().astype("float")
            )
            subblock_dicts.append(subblock_dict)

        df = pd.DataFrame.from_dict(subblock_dicts)
        metadata = ET.fromstring(stack.metadata())

    return df, metadata


def reorder_channels(stack, metadata):
    """
    Sorts channels of a stack by wavelength.

    Args:
        stack (numpy.ndarray): X x Y x C x Z image stack.
        metadata (xml.etree.ElementTree): stack metadata.

    Returns:
        Stack after sorting the channels.
    """
    channels_metadata = metadata.findall(
        "./Metadata/Information/Image/Dimensions/Channels/Channel"
    )
    wavelengths = []
    for channel in channels_metadata:
        wavelengths.append(float(channel.find("./EmissionWavelength").text))

    channel_order = np.argsort(np.array(wavelengths))
    return stack[:, :, channel_order, :]
