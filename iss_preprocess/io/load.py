import json
import re
import shutil
import warnings
from pathlib import Path

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
    project = Path(data_path).parts[0]
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
    project = Path(data_path).parts[0]
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
    tile_name = f"{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}"
    pattern1 = f"{prefix}_MMStack_{tile_coors[0]}-Pos{tile_name}"
    pattern2 = f"{prefix}_MMStack_Pos-{tile_coors[0]}-{tile_name}"
    # Search for files matching either pattern
    for p in data_dir.glob("*.tif"):
        if p.name.startswith(pattern1 + ".ome.tif"):
            return pattern1
        elif p.name.startswith(pattern2 + ".ome.tif"):
            return pattern2
    raise ValueError(
        f"Could not find any files matching the patterns {pattern1} or {pattern2} "
        + f"in {data_dir}"
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

    This is the user generated file containing ROI and rounds information

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
            process_fname.parent.mkdir(parents=True, exist_ok=True)
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
        # add back black level
        ops = load_ops(data_path)
        if np.any(stack):  # don't change corrupted stacks
            stack += ops["black_level"][None, None, :].astype(stack.dtype)
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


def get_zprofile(data_path, prefix, tile_coords):
    """Load the zprofile for a tile

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition to load
        tile_coords (tuple): Tile coordinates (roi, xpos, ypos)

    Returns:
        dict: Z profile for the tile, with 'std' and 'top_1permille' keys
    """
    folder = get_processed_path(data_path) / prefix
    r, x, y = tile_coords
    fname = f"{prefix}_MMStack_{r}-Pos{x:03}_{y:03}_zprofile.npz"
    zprofile = np.load(folder / fname)
    return dict(zprofile)


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


def get_roi_dimensions(data_path, prefix=None, save=True):
    """Find imaging ROIs and determine their dimensions.

    The output is the maximum index of the file names, which are 0 based. It is
    therefore the number of tiles in each dimension minus 1.

    Create and/or load f"{prefix}_roi_dims.npy". The default (None for
    ops['reference_prefix']) should be used for all acquisitions that have the same ROI
    dimensions (everything except overviews).

    Args:
        data_path (str): Relative path to data
        prefix (str, optional): Prefix of acquisition to load. Defaults to None.
        save (bool, optional): If True save roi dimensions if they are not already found
            on disk. Default to True

    Returns:
        numpy.ndarray: Nroi x 3 array of containing (roi_id, NtilesX, NtilesY) for each
            roi

    """
    processed_path = get_processed_path(data_path)
    roi_dims_file = processed_path / f"{prefix}_roi_dims.npy"
    if roi_dims_file.exists():
        return np.load(roi_dims_file)
    if prefix is None:
        ops = load_ops(data_path)
        prefix = ops["reference_prefix"]

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
        if not fnames:
            raise FileNotFoundError(
                f"Cannot get roi dimension for {data_path} "
                + f"using {prefix}. No file found in raw or processed data"
            )
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


def load_mask_by_coors(
    data_path,
    prefix,
    tile_coors,
    suffix="corrected",
):
    """Load masks for a single tile.

    If the corrected mask is not found, the raw mask is loaded instead.

    Args:
        data_path (str): relative path to dataset.
        prefix (str): Full folder name prefix, including round number.
        tile_coors (tuple): Coordinates of tile to load: ROI, Xpos, Ypos.
        suffix (str, optional): Suffix to add to the file name. Defaults to "corrected".

    Returns:
        numpy.ndarray: X x Y x channels stack.

    """
    processed_path = get_processed_path(data_path)
    tile_roi, tile_x, tile_y = tile_coors
    if suffix:
        suffix = f"_{suffix}"
    else:
        suffix = ""
    if "masks" in prefix:
        mask_name = suffix
        acq_prefix = prefix.split("_masks")[0]
    else:
        mask_name = f"_masks{suffix}"
        acq_prefix = prefix

    fname = f"{prefix}{mask_name}_{tile_roi}_{tile_x}_{tile_y}.npy"

    folder = processed_path / "cells"
    if (folder / f"{acq_prefix}_cells").exists():
        # if prefix specific subfolder exists, use that
        folder = folder / f"{acq_prefix}_cells"
    if (folder / fname).exists():
        masks = np.load(folder / fname, allow_pickle=True)
        return masks
    else:
        raise FileNotFoundError(f"Could not find mask file {folder / fname}")


def find_roi_position_on_cryostat(data_path):
    """Find the A/P position of each ROI relative to the first collected slice

    The section order is guess from the sign of `section_thickness_um`, positive for
    antero-posterior slicing (starting from the olfactory bulb), negative for opposite.

    Args:
        data_path (str): Relative path to the data

    Returns:
        roi_slice_pos_um (dict): For each ROI, the slice depth in um relative to the
            first collected slice
        min_step (float): Minimum thickness between two slices

    """
    metadata = load_metadata(data_path)
    rois = metadata["ROI"].keys()

    section_info = load_section_position(data_path)
    section_info.sort_values(by="absolute_section", inplace=True)
    constant_thickness = np.sum(np.diff(section_info.section_thickness_um)) == 0
    if any(np.diff(section_info.absolute_section) > 1):
        if not constant_thickness:
            raise IOError(
                "I need to know the thickness of all the slices.\n"
                + "Please add missing sections to `section_position.csv`"
            )
        # pos_um = section_info.absolute_section * section_info.section_thickness_um
    else:
        # we have all the slices in the csv, we can deal with variable thickness
        increase = section_info.section_thickness_um.values
        section_info["section_position"] = increase.cumsum()

    # find where is each slice of the chamber in the section order of the whole brain
    # the chamber folder should be called chamber_XX
    chamber = int(Path(data_path).name.split("_")[1])
    chamber_pos2section_order = {
        s.chamber_position: s.section_position
        for _, s in section_info[section_info.chamber == chamber].iterrows()
    }
    # find where is each roi in the chamber
    roi_id2chamber_pos = {roi: metadata["ROI"][roi]["chamber_position"] for roi in rois}
    # combine both to find for each ROI the distance sliced since the first
    roi_slice_pos_um = {
        roi: chamber_pos2section_order[roi_id2chamber_pos[roi]] for roi in rois
    }

    return roi_slice_pos_um, section_info.section_thickness_um.min()
