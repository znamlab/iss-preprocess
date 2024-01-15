import numpy as np
import pandas as pd
import warnings
from tifffile import TiffFile
import json
from flexiznam.config import PARAMETERS
from pathlib import Path
import yaml
import re


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


# AB: LGTM 10/01/23
def load_metadata(data_path):
    """Load the metadata.yml file

    This is the user generated file containing ROI and rounds informations

    Args:
        data_path (str): Relative path to data

    Returns:
        dict: Content of `{chamber}_metadata.yml`

    """
    metadata_fname = get_raw_path(data_path) / (Path(data_path).name + "_metadata.yml")
    if not metadata_fname.exists():
        metadata_fname = get_processed_path(data_path) / (
            Path(data_path).name + "_metadata.yml"
        )
        if metadata_fname.exists():
            print("Metadata not found in raw data, loading from processed data")
        else:
            raise FileNotFoundError(
                f"Metadata not found.\n{metadata_fname} does not exist"
            )
    with open(metadata_fname, "r") as f:
        metadata = yaml.safe_load(f)
    return metadata


def get_pixel_size(data_path, prefix="genes_round_1_1"):
    """Get pixel size from MicroManager metadata.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Which acquisition prefix to use. Defaults to "genes_round_1_1".

    """
    acq_data = load_micromanager_metadata(data_path, prefix=prefix)
    pixel_size = acq_data["FrameKey-0-0-0"]["PixelSizeUm"]
    return pixel_size


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
    assert len(fmetadata) == 1
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
        numpy.ndarray: X x Y x Z stack.
    """
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
    # Create channel and Z position arrays based on metadata
    channels = [int(metadata[frame_key]["Camera"][-1]) for frame_key in frame_keys]
    unique_channels = sorted(list(set(channels)))
    nch = len(unique_channels)
    # Create a mapping of channel IDs to a 0-based index
    channel_map = {ch_id: idx for idx, ch_id in enumerate(unique_channels)}
    # Determine Z positions
    if metadata[frame_keys[0]]["Core-Focus"] == "Piezo":
        zs = [metadata[frame_key]["ImageNumber"] for frame_key in frame_keys]
    else:
        zs = [metadata[frame_key]["ZPositionUm"] for frame_key in frame_keys]
    unique_zs = sorted(list(set(zs)))
    nz = len(unique_zs)
    xpix = stack.pages[0].tags["ImageWidth"].value
    ypix = stack.pages[0].tags["ImageLength"].value
    im = np.zeros((ypix, xpix, nch, nz))
    for page, frame_key in zip(stack.pages, frame_keys):
        ch_id = int(metadata[frame_key]["Camera"][-1])  # Actual channel ID
        ch = channel_map[ch_id]  # Mapped channel index
        if metadata[frame_keys[0]]["Core-Focus"] == "Piezo":
            z = unique_zs.index(metadata[frame_key]["ImageNumber"])
        else:
            z = unique_zs.index(metadata[frame_key]["ZPositionUm"])
    
        im[:, :, ch, z] = page.asarray()

    return im


def get_roi_dimensions(data_path, prefix="genes_round_1_1", save=True):
    """Find imaging ROIs and determine their dimensions.

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
        pattern = (
            rf"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d)_{ops['genes_projection']}.tif"
        )
    else:
        pattern = rf"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d).ome.tif"
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
