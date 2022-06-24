import numpy as np
import czifile
import pandas as pd
import xml.etree.ElementTree as ET
from tifffile import TiffFile


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
                page_dict['data'] = page.asarray()
                page_dict['X'] = page.tags['MicroManagerMetadata'].value['XPosition_um_Intended']
                page_dict['Y'] = page.tags['MicroManagerMetadata'].value['YPosition_um_Intended']
                page_dict['Z'] = page.tags['MicroManagerMetadata'].value['ZPosition_um_Intended']
                page_dict['C'] = ch
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


