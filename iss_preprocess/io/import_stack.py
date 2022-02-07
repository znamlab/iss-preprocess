import sys, argparse
import glob
import numpy as np
import czifile
from skimage.io import ImageCollection
import pandas as pd
from tifffile import TiffWriter
import xml.etree.ElementTree as ET


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
            subblock_dict['data'] = subblock.data_segment().data().squeeze().astype('float')
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
        './Metadata/Information/Image/Dimensions/Channels/Channel'
    )
    wavelengths = []
    for channel in channels_metadata:
        wavelengths.append(float(channel.find('./EmissionWavelength').text))

    channel_order = np.argsort(np.array(wavelengths))
    return stack[:,:,channel_order,:]


def write_stack(stack, fname, bigtiff=False):
    """
    Write a stack to file as a multipage TIFF

    Args:
        stack (numpy.ndarray): X x Y x ... array (can have multiple channels /
            zplanes, etc.)
        fname (str): save path for the TIFF

    """
    stack = stack.reshape((stack.shape[0], stack.shape[1], -1))
    stack[stack<0] = 0

    with TiffWriter(fname, bigtiff=bigtiff) as tif:
        for frame in range(stack.shape[2]):
            tif.write(
                np.uint16(stack[:,:,frame]),
                contiguous=True
            )
