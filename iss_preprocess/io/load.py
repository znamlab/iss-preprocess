import json
import re
import warnings
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml
from flexiznam.config import PARAMETERS
from tifffile import TiffFile


def get_raw_path(data_path):
    """Return the path to the raw data.

    Args:
        data_path (str): Relative path to data

    Returns:
        pathlib.Path: Path to raw data

    """
    project = data_path.split("/")[0]
    if project in PARAMETERS["project_paths"].keys():
        raw_path = Path(PARAMETERS["project_paths"][project]["raw"])
    else:
        raw_path = Path(PARAMETERS["data_root"]["raw"])
    return raw_path / data_path


def get_processed_path(data_path):
    """Return the path to the processed data.

    Args:
        data_path (str): Relative path to data

    Returns:
        pathlib.Path: Path to processed data

    """
    project = data_path.split("/")[0]
    if project in PARAMETERS["project_paths"].keys():
        processed_path = Path(PARAMETERS["project_paths"][project]["processed"])
    else:
        processed_path = Path(PARAMETERS["data_root"]["processed"])
    return processed_path / data_path


def get_raw_filename(data_path, prefix, tile_coors):
    """Return the root name of raw data for a tile.

    Raw file names may take on different patterns depending on micromanager version.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition to load
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)

    Returns:
        str: Root name of the raw data file

    """
    data_dir = get_raw_path(data_path) / prefix

    # Define the patterns
    pattern1 = f"{prefix}_MMStack_{tile_coors[0]}-Pos{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}"
    pattern2 = f"{prefix}_MMStack_Pos-{tile_coors[0]}-{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}"
    # Search for files matching either pattern
    for p in data_dir.glob("*.tif"):
        if p.name.startswith(pattern1 + ".ome.tif"):
            return pattern1
        elif p.name.startswith(pattern2 + ".ome.tif"):
            return pattern2
    raise ValueError(
        f"Could not find any files matching the patterns {pattern1} or {pattern2} in {data_dir}"
    )


def load_hyb_probes_metadata():
    """Load the hybridisation probes metadata.

    Returns:
        dict: Contents of `hybridisation_probes.yml`

    """
    fname = Path(__file__).parent.parent / "call" / "hybridisation_probes.yml"
    with open(fname, "r") as f:
        hyb_probes = yaml.safe_load(f)
    return hyb_probes


def load_ops(data_path):
    """Load the ops.yaml file.

    This must be manually generated first. If it is not found, the default
    options are used.

    Args:
        data_path (str): Relative path to data

    Returns:
        dict: Options, see config/defaults_ops.yaml for description

    """

    def flatten_dict(d):
        flattened_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened_dict[subkey] = subvalue
            else:
                flattened_dict[key] = value
        return flattened_dict

    processed_path = get_processed_path(data_path)
    ops_fname = processed_path / "ops.yml"

    default_ops_fname = Path(__file__).parent.parent / "config" / "default_ops.yml"
    with open(default_ops_fname, "r") as f:
        default_ops = flatten_dict(yaml.safe_load(f))
    if not ops_fname.exists():
        print("ops.yml not found, using defaults")
        ops = default_ops
    else:
        with open(ops_fname, "r") as f:
            ops = flatten_dict(yaml.safe_load(f))
        # for any keys that are not in the ops file, use the defaults
        ops = dict(default_ops, **ops)

    black_level_fname = processed_path / "black_level.npy"
    if black_level_fname.exists():
        ops["black_level"] = np.load(black_level_fname)
    else:
        print("black level not found, computing from dark frame")
        dark_fname = get_processed_path(ops["dark_frame_path"])
        dark_frames = load_stack(dark_fname)
        ops["black_level"] = dark_frames.mean(axis=(0, 1))
        np.save(black_level_fname, ops["black_level"])

    try:
        metadata = load_metadata(data_path)
    except FileNotFoundError:
        metadata = {
            "camera_order": [1, 3, 4, 2],
            "genes_rounds": 7,
            "barcode_rounds": 10,
        }
        warnings.warn(f"Metadata file not found, using {metadata}.")
    ops.update(
        {
            "camera_order": metadata["camera_order"],
            "genes_rounds": metadata["genes_rounds"],
            "barcode_rounds": metadata["barcode_rounds"],
        }
    )

    return ops


def load_metadata(data_path):
    """Load the metadata.yml file

    This is the user generated file containing ROI and rounds informations

    Args:
        data_path (str): Relative path to data

    Returns:
        dict: Content of `{chamber}_metadata.yml`

    """
    process_fname = get_processed_path(data_path) / (
        Path(data_path).name + "_metadata.yml"
    )

    if not process_fname.exists():
        raw_fname = get_raw_path(data_path) / (Path(data_path).name + "_metadata.yml")
        if raw_fname.exists():
            print("Metadata not found in processed data, copying from raw data")
            shutil.copy(raw_fname, process_fname)
        else:
            raise FileNotFoundError(
                f"Metadata not found.\n{process_fname} does not exist"
            )
    with open(process_fname, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata


def get_pixel_size(data_path, prefix="genes_round_1_1"):
    """Get pixel size from MicroManager metadata.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Which acquisition prefix to use. Defaults to
            "genes_round_1_1".

    Returns:
        float: Pixel size in microns
    """
    acq_data = load_micromanager_metadata(data_path, prefix=prefix)
    pixel_size = acq_data["FrameKey-0-0-0"]["PixelSizeUm"]
    return pixel_size


def get_z_step(data_path, prefix="genes_round_1_1"):
    """Get z step size from MicroManager metadata.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Which acquisition prefix to use. Defaults to
            "genes_round_1_1".

    Returns:
        float: Z step size in microns
    """
    acq_data = load_micromanager_metadata(data_path, prefix=prefix)
    z_step = acq_data["Summary"]["z-step_um"]
    return z_step


def load_micromanager_metadata(data_path, prefix):
    """Load the metadata.txt of a single acquisition round

    This is the detailed metadata from the microscope.

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, including round number if applicable

    Returns:
        metadata (dict): Content of the metadata file

    """
    acq_folder = get_processed_path(data_path) / prefix
    # the metadata for the first ROI is always copied. Just in case the first ROI is not
    # ROI 1, we find whichever is available
    fmetadata = list(acq_folder.glob("*_metadata.txt"))
    assert (
        len(fmetadata) > 0
    ), f"No image metadata files found for {data_path} / {prefix}"
    fmetadata = fmetadata[0]
    with open(fmetadata) as json_file:
        metadata = json.load(json_file)
    return metadata


def load_section_position(data_path):
    """Load the section position information

    This is the same for all chambers and is contained in the parent folder of
    `data_path`

    Args:
        data_path (str): Relative path to dataset

    Returns:
        pandas.DataFrame: Slice position info

    """
    mouse_path = get_raw_path(data_path).parent
    csv_path = mouse_path / "section_position.csv"
    slice_info = pd.read_csv(csv_path, index_col=None)
    return slice_info


def load_tile_by_coors(
    data_path, tile_coors=(1, 0, 0), suffix="max", prefix="genes_round_1_1"
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
    if suffix != "max-median":
        fname = (
            f"{prefix}_MMStack_{tile_roi}-"
            + f"Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_{suffix}.tif"
        )
        stack = load_stack(get_processed_path(data_path) / prefix / fname)
    else:
        fname = (
            f"{prefix}_MMStack_{tile_roi}-"
            + f"Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_max.tif"
        )
        stack = load_stack(get_processed_path(data_path) / prefix / fname)
        fname = (
            f"{prefix}_MMStack_{tile_roi}-"
            + f"Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_median.tif"
        )
        stack -= load_stack(get_processed_path(data_path) / prefix / fname)
    return stack


# TODO: add shape check? What if pages are not 2D (rgb, weird tiffs)
def load_stack(fname):
    """
    Load TIFF stack.

    Args:
        fname (str): path to TIFF

    Returns:
        numpy.ndarray: X x Y x Z stack
    """
    with TiffFile(fname) as stack:
        ims = []
        for page in stack.pages:
            ims.append(page.asarray())
        return np.stack(ims, axis=2)


def get_tile_ome(fname, fmetadata=None, use_indexmap=None):
    """
    Load OME TIFF tile.

    Args:
        fname (str): path to OME TIFF
        fmetadata (str, optional): path to OME metadata file. Required if use_indexmap
            is False or None. Defaults to None.
        use_indexmap (bool, optional): Whether to use the indexmap from micromanager
            metadata. If True, the metadata file is not required. Defaults to None.

    Returns:
        numpy.ndarray: X x Y x C x Z z-stack.

    """
    with TiffFile(fname) as stack:
        if not use_indexmap:
            with open(fmetadata) as json_file:
                metadata = json.load(json_file)
            frame_keys = list(metadata.keys())[1:]

        if use_indexmap or (metadata[frame_keys[0]]["Core-Focus"] == "Piezo"):
            # THIS IS CRAP.
            # There is an issue with micromanager and the ome metadata are not always
            # correct use indexmap instead (which is from micromanager but is correct)
            umeta = stack.micromanager_metadata
            indexmap = umeta["IndexMap"]
            zs = indexmap[:, 1]
            channels = indexmap[:, 0]
            unique_channels = sorted(list(set(channels)))
            unique_zs = sorted(list(set(zs)))

        else:
            # metadata is now required since we have an upstairs style tiff
            z_ids = [metadata[frame_key]["ZPositionUm"] for frame_key in frame_keys]
            unique_zs = sorted(list(set(z_ids)))
            zs = [unique_zs.index(z) for z in z_ids]
            # Create channel and Z position arrays based on metadata
            channel_ids = [int(metadata[f_key]["Camera"][-1]) for f_key in frame_keys]
            unique_channels = sorted(list(set(channel_ids)))
            channels = [unique_channels.index(ch) for ch in channel_ids]

        nz = len(unique_zs)
        nch = len(unique_channels)
        xpix = stack.pages[0].tags["ImageWidth"].value
        ypix = stack.pages[0].tags["ImageLength"].value
        im = np.zeros((ypix, xpix, nch, nz), dtype=stack.pages[0].dtype)
        for ip, page in enumerate(stack.pages):
            im[:, :, channels[ip], zs[ip]] = page.asarray()
    return im


def get_roi_dimensions(data_path, prefix="genes_round_1_1", save=True):
    """Find imaging ROIs and determine their dimensions.

    The output is the maximum index of the file names, which are 0 based. It is therefore
    the number of tiles in each dimension minus 1.

    Create and/or load f"{prefix}_roi_dims.npy". The default ("genes_round_1_1") should
    be used for all acquisitions that have the same ROI dimensions (everything except
    overviews).

    Args:
        data_path (str): Relative path to data
        prefix (str, optional): Prefix of acquisition to load. Defaults to
            "genes_round_1_1"
        save (bool, optional): If True save roi dimensions if they are not already found
            on disk. Default to True

    Returns:
        numpy.ndarray: Nroi x 3 array of containing (roi_id, NtilesX, NtilesY) for each roi

    """
    processed_path = get_processed_path(data_path)
    roi_dims_file = processed_path / f"{prefix}_roi_dims.npy"
    if roi_dims_file.exists():
        return np.load(roi_dims_file)

    # file does not exist, let's find roi dims from filenames and create the file
    data_dir = get_raw_path(data_path) / prefix
    fnames = [p.name for p in data_dir.glob("*.tif")]
    if not fnames:
        warnings.warn(
            "Raw data has already been archived. Trying to use projected data"
        )
        ops = load_ops(data_path)
        data_dir = processed_path / prefix
        fnames = [p.name for p in data_dir.glob("*.tif")]
        proj = ops["genes_projection"]
        if proj == "max-median":
            proj = "max"
        pattern = rf"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d)_{proj}.tif"
    else:
        pattern = rf"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d).ome.tif"
    matcher = re.compile(pattern=pattern)
    matches = [matcher.match(fname) for fname in fnames]  # non match will be None
    if not any(matches):
        if not fnames:
            pattern = rf"{prefix}_MMStack_Pos-(\d*)-(\d\d\d)_(\d\d\d)_{proj}.tif"
        else:
            pattern = rf"{prefix}_MMStack_Pos-(\d*)-(\d\d\d)_(\d\d\d).ome.tif"
    matcher = re.compile(pattern=pattern)
    matches = [matcher.match(fname) for fname in fnames]  # non match will be None
    try:
        tile_coors = np.stack([np.array(m.groups(), dtype=int) for m in matches if m])
    except ValueError:
        raise ValueError(
            "Could not find any files matching the pattern " f"{pattern} in {data_dir}"
        )

    rois = np.unique(tile_coors[:, 0])
    roi_list = np.empty((len(rois), 3), dtype=int)
    for iroi, roi in enumerate(rois):
        roi_list[iroi, :] = [
            roi,
            np.max(tile_coors[tile_coors[:, 0] == roi, 1]),
            np.max(tile_coors[tile_coors[:, 0] == roi, 2]),
        ]
    if save:
        np.save(roi_dims_file, roi_list)
    return roi_list


def load_mask_by_coors(data_path, tile_coors, prefix, suffix="corrected"):
    """Load masks for a single tile.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)
        prefix (str): Prefix of acquisition to load
        suffix (str, optional): Suffix of the mask file. Defaults to "corrected".

    Returns:
        numpy.ndarray: Masks for the tile
        numpy.ndarray: Bad pixels for the tile (all True)
    """
    processed_path = get_processed_path(data_path)
    tname = "_".join(map(str, tile_coors))
    masks_file = processed_path / "cells" / f"{prefix}_{suffix}_{tname}.npy"
    masks = np.load(masks_file)

    return masks
