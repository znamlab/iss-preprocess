import numpy as np
import czifile
import pandas as pd
import xml.etree.ElementTree as ET
from ..image.correction import correct_offset
from ..reg.tiles import register_tiles

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


def load_image(fname, ops):
    tiles, metadata = get_tiles(fname)
    tiles = correct_offset(tiles, method='metadata', metadata=metadata)
    im, tile_pos = register_tiles(
        tiles,
        reg_fraction=ops['tile_reg_fraction'],
        method='custom',
        max_shift=ops['max_shift'],
        ch_to_align=-1
    )
    im = reorder_channels(im, metadata)
    if im.ndim>3:
        im = np.mean(im, axis=3)
    return im

