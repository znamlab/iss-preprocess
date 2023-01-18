import numpy as np
import czifile
import pandas as pd
import xml.etree.ElementTree as ET
from tifffile import TiffFile
import json
from flexiznam.config import PARAMETERS
from pathlib import Path
import yaml


def load_hyb_probes_metadata():
    fname = Path(__file__).parent.parent / "call" / "hybridisation_probes.yml"
    with open(fname, "r") as f:
        hyb_probes = yaml.safe_load(f)
    return hyb_probes


# AB: LGTM 10/01/23
def load_metadata(data_path):
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    metadata_fname = raw_path / data_path / (Path(data_path).name + "_metadata.yml")
    with open(metadata_fname, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata


def load_single_acq_metdata(data_path, prefix):
    """Load the metadata.txt of a single acquisition round

    This is the detailled metadata from the microscope.

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, including round number if applicable

    Returns:
        metadata (dict): Content of the metadata file
    """

    acq_folder = Path(PARAMETERS["data_root"]["processed"]) / data_path / prefix
    # the metadata for the first ROI is always copied. Just in case the first ROI is not
    # ROI 1, we find whichever is available
    fmetadata = list(acq_folder.glob("*_metadata.txt"))
    assert len(fmetadata) == 1
    fmetadata = fmetadata[0]
    with open(fmetadata) as json_file:
        metadata = json.load(json_file)
    return metadata


# AB: LGTM 10/01/23
def load_tile_by_coors(
    data_path, tile_coors=(1, 0, 0), suffix="fstack", prefix="genes_round_1_1"
):
    """Load processed tile images

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        suffix (str, optional): File name suffix. Defaults to "fstack".
        prefix (str, optional): Full folder name prefix, including round number.
            Defaults to "genes_round_1_1"

    Returns:
        numpy.ndarray: X x Y x channels stack.

    """
    tile_roi, tile_x, tile_y = tile_coors
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    fname = (
        f"{prefix}_MMStack_{tile_roi}-"
        + f"Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_{suffix}.tif"
    )
    return load_stack(processed_path / data_path / prefix / fname)


# TODO: add shape check? What if pages are not 2D (rgb, weird tiffs)
def load_stack(fname):
    with TiffFile(fname) as stack:
        ims = []
        for page in stack.pages:
            ims.append(page.asarray())
    return np.stack(ims, axis=2)


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
