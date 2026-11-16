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

__all__ = [
    "get_raw_path",
    "get_processed_path",
    "get_mouse_path",
    "get_raw_filename",
    "raw_tile_exists",
    "PROJECTION_SUFFIXES",
    "tile_stems",
    "parse_tile_name",
    "get_projection_path",
    "raw_projection_suffixes",
    "is_preprojected",
    "tile_in_rois",
    "load_ops_file",
    "load_hyb_probes_metadata",
    "load_ops",
    "load_metadata",
    "get_pixel_size",
    "get_z_step",
    "load_micromanager_metadata",
    "load_section_position",
    "load_tile_by_coors",
    "load_correction_image",
    "load_stack",
    "get_zprofile",
    "get_tile_ome",
    "get_roi_dimensions",
    "load_mask_by_coors",
    "find_roi_position_on_cryostat",
    "get_channel_round_transforms",
    "load_sequencing_rounds",
]


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


def raw_tile_exists(data_path, prefix, tile_coors):
    """Return True if raw data for a tile exists on disk.

    Used to tell a never-acquired tile (e.g. a skipped no-tissue tile in a
    non-rectangular ROI) apart from a tile whose projection merely failed. Raw data is
    either the `.ome.tif` stack or, for acquisitions projected on the microscope, a
    projection saved in the raw folder.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition to load
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)

    Returns:
        bool: Whether raw data exists for the tile.
    """
    raw_dir = get_raw_path(data_path) / prefix
    for stem in tile_stems(prefix, tile_coors):
        if (raw_dir / f"{stem}.ome.tif").exists():
            return True
        for suffix in PROJECTION_SUFFIXES:
            if (raw_dir / f"{stem}_{suffix}.tif").exists():
                return True
    return False


# Projections that can be saved next to tiles, either by iss-preprocess in the
# processed folder or by the acquisition software in the raw folder.
PROJECTION_SUFFIXES = ("max", "median", "mean", "fstack")


def tile_stems(prefix, tile_coors):
    """Return the possible file name stems of a tile.

    Tiles are named either `{prefix}_MMStack_{roi}-Pos{x}_{y}` (canonical, used for all
    projections made by iss-preprocess) or `{prefix}_MMStack_Pos-{roi}-{x}_{y}` (the
    Micro-Manager position label, used by newer acquisitions).

    Args:
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)

    Returns:
        list: [canonical stem, position-label stem]
    """
    r, x, y = (int(c) for c in tile_coors)
    return [
        f"{prefix}_MMStack_{r}-Pos{x:03d}_{y:03d}",
        f"{prefix}_MMStack_Pos-{r}-{x:03d}_{y:03d}",
    ]


def _tile_name_regex(prefix):
    suffixes = "|".join(PROJECTION_SUFFIXES)
    return re.compile(
        rf"^{re.escape(prefix)}_MMStack_"
        r"(?:(\d+)-Pos(\d{3})_(\d{3})|Pos-(\d+)-(\d{3})_(\d{3}))"
        rf"(?:\.ome\.tif|_({suffixes})\.tif)$"
    )


def _match_tile(regex, fname):
    m = regex.match(fname)
    if m is None:
        return None
    g = m.groups()
    coors = g[0:3] if g[0] is not None else g[3:6]
    kind = g[6] if g[6] is not None else "ome"
    return (*(int(c) for c in coors), kind)


def parse_tile_name(fname, prefix):
    """Parse a raw or projected tile file name.

    Both naming conventions (see `tile_stems`) are accepted, for raw `.ome.tif` stacks
    and for projections (`_max.tif`, `_median.tif`, ...).

    Args:
        fname (str): File name, without folder.
        prefix (str): Acquisition prefix the file must belong to.

    Returns:
        tuple: (roi, xpos, ypos, kind), kind being "ome" for raw stacks or the
            projection suffix. None if the name does not match.
    """
    return _match_tile(_tile_name_regex(prefix), fname)


def _list_tiles(folder, prefix):
    """List (roi, xpos, ypos, kind) of all tile files of `prefix` in `folder`."""
    regex = _tile_name_regex(prefix)
    fnames = [p.name for p in Path(folder).glob(f"{prefix}_MMStack_*.tif")]
    tiles = [_match_tile(regex, fname) for fname in fnames]
    return [t for t in tiles if t is not None]


def get_projection_path(data_path, prefix, tile_coors, suffix="max"):
    """Find the projected image of a tile.

    Projections are searched first in the processed folder, then in the raw folder (for
    acquisitions projected on the microscope). In each folder the canonical name is
    tried before the position-label name.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)
        suffix (str, optional): Projection suffix. Defaults to "max".

    Returns:
        pathlib.Path: Path to the projection, or None if it does not exist.
    """
    stems = tile_stems(prefix, tile_coors)
    for root in (get_processed_path(data_path), get_raw_path(data_path)):
        for stem in stems:
            path = root / prefix / f"{stem}_{suffix}.tif"
            if path.exists():
                return path
    return None


def raw_projection_suffixes(data_path, prefix):
    """Return the projection suffixes found in the raw folder of an acquisition.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix.

    Returns:
        set: Projection suffixes (e.g. {"max"}), empty if there are none.
    """
    tiles = _list_tiles(get_raw_path(data_path) / prefix, prefix)
    return {t[3] for t in tiles if t[3] != "ome"}


def is_preprojected(data_path, prefix):
    """Whether an acquisition was projected on the microscope.

    This is the case when the raw folder contains projected tiles and no `.ome.tif`
    stacks. Such acquisitions need no projection by iss-preprocess and their
    projections are read directly from the raw folder.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix.

    Returns:
        bool: True if the acquisition is pre-projected.
    """
    kinds = {t[3] for t in _list_tiles(get_raw_path(data_path) / prefix, prefix)}
    return bool(kinds) and "ome" not in kinds


def load_hyb_probes_metadata():
    """Load the hybridisation probes metadata.

    Returns:
        dict: Contents of `hybridisation_probes.yml`

    """
    fname = Path(__file__).parent.parent / "call" / "hybridisation_probes.yml"
    with open(fname, "r") as f:
        hyb_probes = yaml.safe_load(f)
    return hyb_probes


def _flatten_dict(d):
    """Flatten the sections of an ops file into a single dict."""
    flattened_dict = {}
    for key, value in (d or {}).items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flattened_dict[subkey] = subvalue
        else:
            flattened_dict[key] = value
    return flattened_dict


def load_ops_file(data_path):
    """Load the options set in the `ops.yml` of a chamber, without the defaults.

    Use this to know whether an option was chosen for the dataset, `load_ops` cannot
    tell options from the file and defaults apart.

    Args:
        data_path (str): Relative path to data

    Returns:
        dict: Options in the file, empty if there is no `ops.yml`.
    """
    ops_fname = get_processed_path(data_path) / "ops.yml"
    if not ops_fname.exists():
        return {}
    with open(ops_fname, "r") as f:
        return _flatten_dict(yaml.safe_load(f))


def load_ops(data_path, warn_missing=True):
    """Load the ops.yaml file.

    This must be manually generated first. If it is not found, the default
    options are used.

    `ops["black_level"]` (one value per channel) is read from `black_level.npy`, or
    computed from `ops["dark_frame_path"]` and saved. If neither exists, it is assumed
    to be 0, with a warning.

    Args:
        data_path (str): Relative path to data
        warn_missing (bool, optional): Whether to warn if the ops or metadata files are
            not found. Defaults to True.

    Returns:
        dict: Options, see config/defaults_ops.yaml for description

    """
    processed_path = get_processed_path(data_path)
    ops_fname = processed_path / "ops.yml"

    default_ops_fname = Path(__file__).parent.parent / "config" / "default_ops.yml"
    with open(default_ops_fname, "r") as f:
        default_ops = _flatten_dict(yaml.safe_load(f))
    if not ops_fname.exists() and warn_missing:
        print(f"{ops_fname} not found, using defaults")
    # for any keys that are not in the ops file, use the defaults
    ops = dict(default_ops, **load_ops_file(data_path))

    try:
        metadata = load_metadata(data_path)
    except FileNotFoundError:
        metadata = {
            "camera_order": [1, 3, 4, 2],
            "genes_rounds": 7,
            "barcode_rounds": 10,
        }
        if warn_missing:
            warnings.warn(f"Metadata file not found, using {metadata}.")
    ops.update(
        {
            "camera_order": metadata["camera_order"],
            "genes_rounds": metadata["genes_rounds"],
            "barcode_rounds": metadata["barcode_rounds"],
        }
    )

    black_level_fname = processed_path / "black_level.npy"
    if black_level_fname.exists():
        ops["black_level"] = np.load(black_level_fname)
    elif ops["dark_frame_path"] is not None:
        print("black level not found, computing from dark frame")
        dark_fname = get_processed_path(ops["dark_frame_path"])
        dark_frames = load_stack(dark_fname)
        ops["black_level"] = dark_frames.mean(axis=(0, 1))
        np.save(black_level_fname, ops["black_level"])
    else:
        # The camera offset is not in the Micro-Manager metadata. Assume it is 0, one
        # value per camera. Not saved, so that a dark frame can still be added later.
        ops["black_level"] = np.zeros(len(ops["camera_order"]))
        warnings.warn(
            "No black_level.npy or dark_frame_path in ops, assuming black level is 0. "
            + "Set dark_frame_path in ops.yml to subtract the camera offset."
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


def get_pixel_size(data_path, prefix=None):
    """Get pixel size from MicroManager metadata.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Which acquisition prefix to use. Defaults to
            "genes_round_1_1".

    Returns:
        float: Pixel size in microns
    """
    if prefix is None:
        ops = load_ops(data_path)
        prefix = ops["reference_prefix"]
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
    # the metadata for the first ROI is always copied. Just in case the first ROI is not
    # ROI 1, we find whichever is available. Acquisitions projected on the microscope
    # are not copied to processed, so fall back to the raw folder (read only).
    fmetadata = list((get_processed_path(data_path) / prefix).glob("*_metadata.txt"))
    if not fmetadata:
        fmetadata = list((get_raw_path(data_path) / prefix).glob("*_metadata.txt"))
    assert (
        len(fmetadata) > 0
    ), f"No image metadata files found for {data_path} / {prefix}"
    fmetadata = fmetadata[0]
    with open(fmetadata) as json_file:
        metadata = json.load(json_file)
    return metadata


def get_mouse_path(data_path):
    """Return the mouse-level processed dir (parent of the chamber dir).

    Used by mouse-wide artifacts that are shared across chambers, e.g.
    ``section_position.csv`` and the shared soma cluster-means file.
    """
    return get_processed_path(data_path).parent


def load_section_position(data_path):
    """Load the section position information

    This is the same for all chambers and is contained in the parent folder of
    `data_path`

    Args:
        data_path (str): Relative path to dataset

    Returns:
        pandas.DataFrame: Slice position info

    """
    mouse_path = get_mouse_path(data_path)
    csv_path = mouse_path / "section_position.csv"
    if not csv_path.exists():
        # look in processed
        mouse_path = get_mouse_path(data_path)
        csv_path = mouse_path / "section_position.csv"
    slice_info = pd.read_csv(csv_path, index_col=None)
    return slice_info


def tile_in_rois(tile_coors, roi_dims):
    """Whether a tile is inside the grid of one of the ROIs.

    Args:
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos)
        roi_dims (numpy.ndarray): Output of `get_roi_dimensions`, (roi, max x, max y)

    Returns:
        bool: True if the ROI exists and the tile is within its grid.
    """
    roi, x, y = tile_coors
    in_grid = (roi_dims[:, 0] == roi) & (roi_dims[:, 1] >= x) & (roi_dims[:, 2] >= y)
    return bool(x >= 0 and y >= 0 and np.any(in_grid))


def _reference_tile_info(data_path, prefix, suffix):
    """Shape and dtype of a projected tile of an acquisition, read from its header.

    Returns:
        tuple: (shape, dtype) as returned by `load_stack`, (None, None) if there is no
            projected tile.
    """
    for root in (get_processed_path(data_path), get_raw_path(data_path)):
        for path in (root / prefix).glob(f"{prefix}_MMStack_*_{suffix}.tif"):
            with TiffFile(path) as tif:
                page = tif.pages[0]
                return (*page.shape, len(tif.pages)), page.dtype
    return None, None


def _load_missing_tile_fallback(data_path, prefix, suffix, tile_coors, tile_path):
    """Return a stand-in for a tile that was never acquired.

    Non-rectangular ROIs skip empty/no-tissue tiles during acquisition, so the grid of
    tiles has holes. A tile is considered never acquired if it is inside the grid of
    an imaged ROI and has no raw data. It is filled with the black level of each
    channel (`ops["black_level"]`, 0 if not defined), i.e. what the camera records
    without light, so that it is 0 after black level subtraction.

    Tiles outside the imaged ROIs, or tiles with raw data but no projection (failed
    projection), raise FileNotFoundError.

    Args:
        data_path (str): Relative path to dataset.
        prefix (str): Acquisition prefix, e.g. "barcode_round_1_1".
        suffix (str): Projection suffix being loaded, e.g. "max".
        tile_coors (tuple): Tile coordinates (roi, xpos, ypos).
        tile_path (Path): Path of the missing tile (used for error/log messages).

    Returns:
        numpy.ndarray: Stand-in tile, with the shape and dtype of real tiles.
    """
    if not tile_in_rois(tile_coors, get_roi_dimensions(data_path, prefix)):
        raise FileNotFoundError(
            f"{tile_path} not found, tile {tile_coors} is outside the ROIs of {prefix}"
        )
    if raw_tile_exists(data_path, prefix, tile_coors):
        raise FileNotFoundError(
            f"{tile_path} not found but raw data exists. Did the projection fail?"
        )
    shape, dtype = _reference_tile_info(data_path, prefix, suffix)
    if shape is None:
        raise FileNotFoundError(
            f"{tile_path} not found, and no {suffix} projection of {prefix} exists"
        )

    black_level = np.atleast_1d(load_ops(data_path, warn_missing=False)["black_level"])
    if np.ptp(black_level) == 0:
        # the same for all channels (e.g. assumed 0), whatever the number of channels
        black_level = black_level[:1]
    if black_level.size != shape[-1] and black_level.size != 1:
        raise ValueError(
            f"Black level has {black_level.size} values but {prefix} tiles have "
            + f"{shape[-1]} channels"
        )
    print(f"[missing-tile] {tile_path.name} not acquired -> black level {black_level}")
    tile = np.empty(shape, dtype=dtype)
    tile[...] = np.round(black_level).astype(dtype)
    return tile


def load_tile_by_coors(
    data_path,
    tile_coors=(1, 0, 0),
    suffix="max",
    prefix="genes_round_1_1",
    correct_illumination=False,
):
    """Load processed tile images

    Projections are looked up with `get_projection_path`: in processed first, then in
    raw (for acquisitions projected on the microscope), under either tile naming
    convention. Tiles that were never acquired (holes in non-rectangular ROIs) are
    filled with the black level, see `_load_missing_tile_fallback`.

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        suffix (str, optional): File name suffix. Defaults to "fstack".
        prefix (str, optional): Full folder name prefix, including round number.
            Defaults to "genes_round_1_1"
        correct_illumination (bool, optional): Whether to correct for illumination
            Defaults to False.

    Returns:
        numpy.ndarray: X x Y x channels stack. uint16 if not corrected, float otherwise.

    """
    if suffix != "max-median":
        tile_path = get_projection_path(data_path, prefix, tile_coors, suffix)
        if tile_path is not None:
            stack = load_stack(tile_path)
        else:
            # report the canonical processed name in errors and logs
            canonical = tile_stems(prefix, tile_coors)[0]
            tile_path = (
                get_processed_path(data_path) / prefix / f"{canonical}_{suffix}.tif"
            )
            stack = _load_missing_tile_fallback(
                data_path, prefix, suffix, tile_coors, tile_path
            )

        if correct_illumination:
            ops = load_ops(data_path)
            stack = stack.astype(float)
            correction_image = load_correction_image(data_path, suffix, prefix)
            # special case for missing data, where we don't want to do anything
            no_data = np.logical_not(np.any(stack, axis=(0, 1, 2)))
            if stack.ndim == 4:
                stack = (
                    stack - ops["black_level"][np.newaxis, np.newaxis, :, np.newaxis]
                ) / correction_image[:, :, :, np.newaxis]
                # blank rounds with no data
                stack[..., no_data] *= 0
            elif no_data:
                # do nothing, the stack is empty
                pass
            else:
                stack = (
                    stack - ops["black_level"][np.newaxis, np.newaxis, :]
                ) / correction_image
    else:
        stack = load_tile_by_coors(
            data_path, tile_coors, "max", prefix, correct_illumination
        )
        # by definition max is above median, won't underflow or clip
        stack -= load_tile_by_coors(
            data_path, tile_coors, "median", prefix, correct_illumination
        )
    return stack


def load_correction_image(data_path, projection, prefix, corr_prefix=None):
    """Load the image for illumination correction

    By default, find the appropriate correction image from the ops file. This can be
    overridden by providing a corr_prefix.

    Args:
        data_path (str): Relative path to dataset.
        projection (str): Projection to use, one of `max`, `median`.
        prefix (str): Prefix to correct, this is the image you want to correct, not
            the one you use for the correction.
        corr_prefix (str, optional): Prefix of the image to use for correction. If
            provided, this is used instead of the prefix. Defaults to None.

    Returns:
        numpy.ndarray: X x Y x channels stack.

    """
    ops = load_ops(data_path)
    if corr_prefix is None:
        if f"{prefix}_average_for_correction" in ops:
            # see if it has been specified in the ops file
            corr_prefix = ops[f"{prefix}_average_for_correction"]
        elif ("genes" in prefix) or ("barcode" in prefix):
            # if not and it's a round acquisition, use the "all_rounds" average
            corr_prefix = prefix.split("_round")[0] + "_round"
        else:
            # otherwise, just use the single prefix
            corr_prefix = prefix

    fname = f"{corr_prefix}_{projection}_average.tif"
    stack = load_stack(get_processed_path(data_path) / "averages" / fname)
    return stack


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


def get_tile_ome(fname, fmetadata=None, use_indexmap=True):
    """
    Load OME TIFF tile.

    Args:
        fname (str): path to OME TIFF
        fmetadata (str, optional): path to OME metadata file. Required if use_indexmap
            is False or None. Defaults to None.
        use_indexmap (bool, optional): Whether to use the indexmap from micromanager
            metadata. If True, the metadata file is not required. Defaults to True.

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
        expected_pages = nz * nch
        for ip, page in enumerate(stack.pages[:expected_pages]):
            page_array = page.asarray()
            if page_array.shape != (ypix, xpix):
                print(f"Skipping page {ip} due to shape mismatch: {page_array.shape}")
                continue
            im[:, :, channels[ip], zs[ip]] = page_array
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
    if prefix is None:
        ops = load_ops(data_path)
        prefix = ops["reference_prefix"]
    roi_dims_file = processed_path / f"{prefix}_roi_dims.npy"
    if roi_dims_file.exists():
        return np.load(roi_dims_file)

    # file does not exist, let's find roi dims from filenames and create the file.
    # Raw data is the reference, it may contain `.ome.tif` stacks or, for acquisitions
    # projected on the microscope, projections. If raw data has been archived, use the
    # projections in processed. Both tile naming conventions are accepted.
    raw_dir = get_raw_path(data_path) / prefix
    tiles = _list_tiles(raw_dir, prefix)
    if not tiles:
        tiles = _list_tiles(processed_path / prefix, prefix)
        if not tiles:
            raise FileNotFoundError(
                f"Cannot get roi dimension for {data_path} using {prefix}. No tile "
                + f"found in {raw_dir} or {processed_path / prefix}"
            )
        warnings.warn(
            f"No raw tiles found in {raw_dir}. Using projected data in processed."
        )
    tile_coors = np.array([t[:3] for t in tiles], dtype=int)

    rois = np.unique(tile_coors[:, 0])
    roi_list = np.empty((len(rois), 3), dtype=int)
    for iroi, roi in enumerate(rois):
        roi_list[iroi, :] = [
            roi,
            np.max(tile_coors[tile_coors[:, 0] == roi, 1]),
            np.max(tile_coors[tile_coors[:, 0] == roi, 2]),
        ]
    if save:
        # pre-projected acquisitions may not have a processed folder yet
        roi_dims_file.parent.mkdir(parents=True, exist_ok=True)
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


def get_channel_round_transforms(
    data_path, prefix, tile_coors=None, shifts_type="best", load_file=True
):
    """Load the channel and round shifts for a given tile and sequencing acquisition.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Prefix of the sequencing round.
        tile_coors (tuple, optional): Coordinates of the tile to process. Required if
            `shifts_type` is not `reference`. Defaults to None.
        corrected_shifts (str, optional): Which shift to use. One of `reference`,
            `single_tile`, `ransac`, or `best`. Defaults to `best`.
        load_file (bool, optional): Whether to load the shifts from file or just return
            the file name. Defaults to True.


    Returns:
        np.ndarray | Path: Array of channel and round shifts if load_file, else the path

    """

    tforms_path = get_processed_path(data_path) / "reg" / prefix
    old_path = get_processed_path(data_path) / "reg"
    if tile_coors is not None:
        tile_name = f"{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}"
    elif shifts_type != "reference":
        raise ValueError("tile_coors must be provided for non-reference shifts")

    if shifts_type == "reference":
        tforms_fname = f"ref_tile_tforms_{prefix}.npz"
    elif shifts_type == "single_tile":
        tforms_fname = f"tforms_{prefix}_{tile_name}.npz"
    elif (shifts_type == "ransac") or (shifts_type == "corrected"):
        tforms_fname = f"tforms_corrected_{prefix}_{tile_name}.npz"
    elif shifts_type == "best":
        tforms_fname = f"tforms_best_{prefix}_{tile_name}.npz"
    else:
        raise ValueError(f"unknown shift correction method: {shifts_type}")
    filename = tforms_path / tforms_fname
    if not filename.exists():
        filename = old_path / tforms_fname
    if not load_file:
        return filename
    tforms = np.load(filename, allow_pickle=True)
    return tforms


def load_sequencing_rounds(
    data_path,
    tile_coors=(1, 0, 0),
    nrounds=7,
    suffix="max",
    prefix="genes_round",
    specific_rounds=None,
    correct_illumination=False,
):
    """Load processed tile images across rounds

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        nrounds (int, optional): Number of rounds to load. Used only if
            `specific_rounds` is None. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to 'fstack'.
        prefix (str, optional): the folder name prefix, before round number. Defaults
            to "genes_round"
        specific_round (list, optional): if not None, specify which rounds must be
            loaded and ignores `nrounds`. Defaults to None
        correct_illumination (bool, optional): Whether to correct for illumination
            Defaults to False.

    Returns:
        numpy.ndarray: X x Y x channels x rounds stack.

    """
    if specific_rounds is None:
        specific_rounds = np.arange(nrounds) + 1

    ims = []
    for iround in specific_rounds:
        dirname = f"{prefix}_{iround}_1"
        ims.append(
            load_tile_by_coors(
                data_path,
                tile_coors=tile_coors,
                suffix=suffix,
                prefix=dirname,
                correct_illumination=correct_illumination,
            )
        )
    return np.stack(ims, axis=3)


def get_shifts_to_ref(data_path, prefix, roi, tilex, tiley):
    """Get the shifts to reference coordinates for a given tile

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of the tile to register
        roi (int): ROI ID
        tilex (int): X coordinate of the tile
        tiley (int): Y coordinate of the tile

    Returns:
        np.NpzFile: The transformation parameter to reference coordinates

    """
    ops = load_ops(data_path)
    if ops["corrected_shifts2ref"] == "single_tile":
        corrected_shifts = ""
    elif ops["corrected_shifts2ref"] == "ransac":
        corrected_shifts = "_corrected"
    elif ops["corrected_shifts2ref"] == "best":
        corrected_shifts = "_best"
    elif ops["corrected_shifts2ref"] == "stitched":
        corrected_shifts = "_stitched"
    else:
        raise ValueError(f"Corrected shifts {ops['corrected_shifts2ref']} not recognised")
    processed_path = get_processed_path(data_path)
    tform_path = processed_path / "reg" / f"tforms{corrected_shifts}_to_ref_{prefix}_{roi}_{tilex}_{tiley}.npz"
    if not tform_path.exists():
        tform_path = processed_path / "reg" / f"to_ref_{prefix}" / f"tforms{corrected_shifts}_to_ref_{prefix}_{roi}_{tilex}_{tiley}.npz"
    
    if not tform_path.exists():
        raise FileNotFoundError(
            f"Could not find tforms to ref for {prefix} tile {roi}_{tilex}_{tiley}"
        )
    
    tform2ref = np.load(tform_path)
    return tform2ref