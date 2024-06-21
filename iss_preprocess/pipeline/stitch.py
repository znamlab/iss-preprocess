import warnings
from pathlib import Path
from skimage.transform import AffineTransform, warp
import numpy as np
import pandas as pd
from image_tools.registration import phase_correlation as mpc
from image_tools.similarity_transforms import make_transform, transform_image
from scipy.ndimage import median_filter, binary_erosion
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation
from skimage import transform
from znamutils import slurm_it

import iss_preprocess as iss

from .. import vis
from ..image.correction import apply_illumination_correction
from ..io import get_roi_dimensions, load_ops, load_stack, load_tile_by_coors
from ..reg import (
    estimate_rotation_translation,
    estimate_scale_rotation_translation,
)
from . import pipeline
from .register import align_spots


def load_tile_ref_coors(data_path, tile_coors, prefix, filter_r=True, projection=None):
    """Load one single tile in the reference coordinates

    This load a tile of `prefix` with channels/rounds registered

    Args:
        data_path (str): Relative path to data
        tile_coordinates (tuple): (Roi, tileX, tileY) tuple
        prefix (str): Acquisition to load. If `genes_round` or `barcode_round` will load
            all the rounds.
        filter_r (bool, optional): Apply filter on rounds data? Parameters will be read
            from `ops`. Default to True
        projection (str, optional): Projection to load. If None, will use the one in
            `ops`. Default to None

    Returns:
        np.array: A (X x Y x Nchannels x Nrounds) registered stack
        np.array: A (X x Y) boolean array of bad pixels that fall outside image after
            registration

    """
    if "_masks" in prefix:
        # we have a mask, the load is different
        stack = iss.io.load_mask_by_coors(data_path, prefix, tile_coors, projection)
        prefix = prefix.replace("_masks", "")
        # make 3D to match the other load
        bad_pixels = np.zeros(stack.shape[:2], dtype=bool)
        stack = stack[:, :, np.newaxis]
        interpolation = 0
    else:
        stack, bad_pixels = pipeline.load_and_register_tile(
            data_path, tile_coors, prefix, filter_r=filter_r, projection=projection
        )
        interpolation = 1
    ops = load_ops(data_path)
    ref_prefix = ops["reference_prefix"]
    if prefix.startswith(ref_prefix):
        # No need to register to ref
        return stack, bad_pixels
    # we have data with channels/rounds registered
    # Now find how much the acquisition stitching is shifting the data compared to
    # reference
    # TODO: we are warping the image twice - in `load_and_register_tile` and here
    # if we ever use this function for downstream analyses (e.g. detecting spots)
    # we should make sure to warp once
    stack, bad_pixels = warp_stack_to_ref(
        stack=stack,
        data_path=data_path,
        prefix=prefix,
        tile_coors=tile_coors,
        bad_pixels=bad_pixels,
        interpolation=interpolation,
    )
    return stack, bad_pixels


def warp_stack_to_ref(
    stack, data_path, prefix, tile_coors, interpolation=1, bad_pixels=None
):
    """Warp a stack to the reference coordinates

    Args:
        stack (np.array): A (X x Y x Nchannels x Nrounds) stack
        data_path (str): Relative path to data
        prefix (str): Acquisition to use to find registration parameters
        tile_coors (tuple): (Roi, tileX, tileY) tuple
        interpolation (int, optional): Interpolation order. Defaults to 1.
        bad_pixels (np.array, optional): A (X x Y) boolean array of bad pixels that fall
            outside image after registration. If None, will not apply any mask. Defaults
            to None.

    Returns:
        np.array: A (X x Y x Nchannels x Nrounds) registered stack
        np.array: A (X x Y) boolean array of bad pixels that fall outside image after
            registration

    """
    ops = load_ops(data_path)
    reg2ref = get_tform_to_ref(data_path, prefix, tile_coors)

    if ops["align_method"] == "affine":
        tform = reg2ref["matrix_between_channels"][0]
    else:
        tform = make_transform(
            s=reg2ref["scales"][0][0],  # same reg for all round and channels
            angle=reg2ref["angles"][0][0],
            shift=reg2ref["shifts"][0],
            shape=stack.shape[:2],
        )
    if (bad_pixels is not None) and np.any(bad_pixels):
        stack[bad_pixels] = np.nan

    if stack.ndim == 2:
        # we have just an image, add an axis for channel and one for round
        stack = stack[:, :, np.newaxis, np.newaxis]
    elif stack.ndim == 3:
        # we have an image with multiple channels, add an axis for round
        stack = stack[:, :, :, np.newaxis]

    for ir in range(stack.shape[3]):
        for ic in range(stack.shape[2]):
            stack[:, :, ic, ir] = warp(
                stack[:, :, ic, ir],
                AffineTransform(matrix=tform).inverse,
                preserve_range=True,
                cval=np.nan,
                order=interpolation,
            )

    bad_pixels = np.any(np.isnan(stack), axis=(2, 3))
    stack[bad_pixels] = 0
    return stack, bad_pixels


def get_tform_to_ref(data_path, prefix, tile_coors, corrected_shifts=None):
    """Load the transformation to reference for a tile

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix
        tile_coors (tuple): (roi, tileX, tileY) tuple
        corrected_shifts (str, optional): Method used to correct shifts to reference.
            If None, will use the one in `ops`. Defaults to None.

    Returns:
        np.array: A dictionary with the transformation parameters

    """
    roi, tilex, tiley = tile_coors
    if corrected_shifts is None:
        ops = load_ops(data_path)
        corrected_shifts = ops["corrected_shifts2ref"]

    valid_shifts = ["single_tile", "ransac", "best"]
    assert corrected_shifts in valid_shifts, (
        f"unknown shifts2ref correction method, must be one of {valid_shifts}",
    )

    # now find registration to ref
    if (("genes" in prefix) or ("barcode" in prefix)) and (
        not prefix.endswith("round")
    ):
        reg_prefix = "_".join(prefix.split("_")[:-2])
    else:
        reg_prefix = prefix

    if corrected_shifts == "single_tile":
        correction_fname = "tforms_to_ref"
    elif corrected_shifts == "ransac":
        correction_fname = "tforms_corrected_to_ref"
    elif corrected_shifts == "best":
        correction_fname = "tforms_best_to_ref"
    reg2ref = np.load(
        iss.io.get_processed_path(data_path)
        / "reg"
        / f"{correction_fname}_{reg_prefix}_{roi}_{tilex}_{tiley}.npz"
    )
    return reg2ref


@slurm_it(conda_env="iss-preprocess")
def register_within_acquisition(
    data_path,
    prefix,
    ref_roi=None,
    ref_ch=0,
    suffix="max",
    correct_illumination=False,
    reload=True,
    save_plot=False,
    dimension_prefix="genes_round_1_1",
):
    """Estimate shifts between all adjacent tiles of an roi

    Saves the median shifts in `"reg" / f"{prefix}_shifts.npz"`.

    Args:
        data_path (str): path to image stacks.
        prefix (str, optional): Full name of the acquisition folder.
        ref_roi (int, optional): ROI to use for registration. If `None` use
            `ops['ref_tile'][0]`. Defaults to None.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        suffix (str, optional): File name suffix. Defaults to 'proj'.
        correct_illumination (bool, optional): Remove black levels and correct illumination
            before registration if True, return raw data otherwise. Default to False
        reload (bool, optional): If target file already exists, reload instead of
            recomputing. Defaults to True
        save_plot (bool, optional): If True save diagnostic plot. Defaults to False
        dimension_prefix (str, optional): Prefix to use to find ROI dimension. Used
            only if the acquisition is an overview. Defaults to 'genes_round_1_1'

    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
    save_fname = iss.io.get_processed_path(data_path) / "reg" / f"{prefix}_shifts.npz"

    if reload and save_fname.exists():
        return np.load(save_fname)
    ops = load_ops(data_path)
    if ref_roi is None:
        ref_roi = ops["ref_tile"][0]
    ndim = get_roi_dimensions(data_path, dimension_prefix)

    # roi_dims is read from file name (0-based), the actual number of tile needs +1
    ntiles = ndim[ndim[:, 0] == ref_roi][0][1:] + 1
    output = np.zeros((ntiles[0], ntiles[1], 4)) + np.nan

    # skip the first x position in case tile direction is right to left
    if ops["x_tile_direction"] == "right_to_left":
        rangex = range(1, ntiles[0])
    else:
        rangex = range(ntiles[0] - 1)
    # skip the first y position in case tile direction is top to bottom
    if ops["y_tile_direction"] == "top_to_bottom":
        rangey = range(1, ntiles[1])
    else:
        rangey = range(ntiles[1] - 1)

    for tilex in rangex:
        for tiley in rangey:
            shift_right, shift_down, tile_shape = register_adjacent_tiles(
                data_path,
                ref_coors=(ref_roi, tilex, tiley),
                ref_ch=ref_ch,
                suffix=suffix,
                prefix=prefix,
                correct_illumination=correct_illumination,
            )
            output[tilex, tiley] = np.hstack([shift_right, shift_down])
    shifts = np.nanmedian(output, axis=(0, 1))

    if save_plot:
        vis.diagnostics.adjacent_tiles_registration(
            data_path, prefix, saved_shifts=shifts, bytile_shifts=output
        )

    save_fname.parent.mkdir(exist_ok=True)
    np.savez(
        save_fname, shift_right=shifts[:2], shift_down=shifts[2:], tile_shape=tile_shape
    )
    return shifts[:2], shifts[2:], tile_shape


def register_adjacent_tiles(
    data_path,
    ref_coors=None,
    ref_ch=0,
    suffix="max",
    prefix="genes_round_1_1",
    correct_illumination=False,
):
    """Estimate shift between adjacent imaging tiles using phase correlation.

    Shifts are typically very similar between different tiles, using shifts
    estimated using a reference tile for the whole acquisition works well.

    Args:
        data_path (str): path to image stacks.
        ref_coors (tuple, optional): coordinates of the reference tile to use for
            registration. Must not be along the bottom or right edge of image. If `None`
            use `ops['ref_tile']`. Defaults to None.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        suffix (str, optional): File name suffix. Defaults to 'proj'.
        prefix (str, optional): Full name of the acquisition folder
        correct_illumination (bool, optional): Remove black levels and correct illumination
            before registration if True, return raw data otherwise. Default to False


    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
    ops = load_ops(data_path)
    if ref_coors is None:
        ref_coors = ops["ref_tile"]
    tile_ref = load_tile_by_coors(
        data_path, tile_coors=ref_coors, suffix=suffix, prefix=prefix
    )
    down_offset = 1 if ops["y_tile_direction"] == "bottom_to_top" else -1
    down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + down_offset)
    tile_down = load_tile_by_coors(
        data_path, tile_coors=down_coors, suffix=suffix, prefix=prefix
    )
    right_offset = 1 if ops["x_tile_direction"] == "left_to_right" else -1
    right_coors = (ref_coors[0], ref_coors[1] + right_offset, ref_coors[2])
    tile_right = load_tile_by_coors(
        data_path, tile_coors=right_coors, suffix=suffix, prefix=prefix
    )
    if correct_illumination:
        tile_ref = apply_illumination_correction(data_path, tile_ref, prefix)
        tile_down = apply_illumination_correction(data_path, tile_down, prefix)
        tile_right = apply_illumination_correction(data_path, tile_right, prefix)

    if ops["reg_median_filter"]:
        msize = ops["reg_median_filter"]
        assert isinstance(msize, int), "reg_median_filter must be an integer"
        tile_ref = median_filter(tile_ref, footprint=disk(msize), axes=(0, 1))
        tile_down = median_filter(tile_down, footprint=disk(msize), axes=(0, 1))
        tile_right = median_filter(tile_right, footprint=disk(msize), axes=(0, 1))

    ypix = tile_ref.shape[0]
    xpix = tile_ref.shape[1]
    reg_pix_x = int(xpix * ops["reg_fraction"])
    reg_pix_y = int(ypix * ops["reg_fraction"])

    shift_right, _, _ = phase_cross_correlation(
        tile_ref[:, -reg_pix_x:, ref_ch],
        tile_right[:, :reg_pix_x, ref_ch],
        upsample_factor=5,
    )
    if any(np.abs(shift_right) >= reg_pix_x * 0.1):
        warnings.warn(
            f"Shift to right tile ({right_coors}) is large: {shift_right}"
            f"({shift_right/reg_pix_x*100}% of overlap). Check that everything is fine."
        )
    shift_right += [0, xpix - reg_pix_x]
    if ops["x_tile_direction"] != "left_to_right":
        shift_right = -shift_right
    shift_down, _, _ = phase_cross_correlation(
        tile_ref[:reg_pix_y, :, ref_ch],
        tile_down[-reg_pix_y:, :, ref_ch],
        upsample_factor=5,
    )
    if any(np.abs(shift_down) >= reg_pix_y * 0.1):
        warnings.warn(
            f"Shift to down tile ({down_coors}) is large: {shift_down}"
            f"({shift_down/reg_pix_y*100}% of overlap). Check that everything is fine."
        )
    shift_down -= [ypix - reg_pix_y, 0]
    if ops["y_tile_direction"] != "bottom_to_top":
        shift_down = -shift_down

    return shift_right, shift_down, (ypix, xpix)


def get_tile_corners(data_path, prefix, roi):
    """Find the corners of all tiles for a roi

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix. For round-based acquisition, round 1 will be
            used
        roi (int): Roi ID

    Returns:
        numpy.ndarray: `tile_corners`, ntiles[0] x ntiles[1] x 2 x 4 matrix of tile
            corners coordinates. Corners are in this order:
            [(origin), (0, 1), (1, 1), (1, 0)]

    """
    roi_dims = get_roi_dimensions(data_path)
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    if not prefix.endswith("_1"):
        # we should have "barcode_round" or "genes_round"
        # always use round 1
        prefix = f"{prefix}_1"
    shifts = np.load(
        iss.io.get_processed_path(data_path) / "reg" / f"{prefix}_shifts.npz"
    )
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles
    )
    corners = np.stack(
        [
            tile_origins + np.array(c_pos) * shifts["tile_shape"]
            for c_pos in ([0, 0], [0, 1], [1, 1], [1, 0])
        ],
        axis=3,
    )
    return corners


def calculate_tile_positions(shift_right, shift_down, tile_shape, ntiles):
    """Calculate position of each tile based on the provided shifts.

    Args:
        shift_right (numpy.array): X and Y shifts between different columns
        shift_down (numpy.array): X and Y shifts between different rows
        tile_shape (numpy.array): shape of each tile
        ntiles (numpy.array): number of tile rows and columns

    Returns:
        numpy.ndarray: `tile_origins`, ntiles[0] x ntiles[1] x 2 matrix of tile origin
            coordinates
        numpy.ndarray: `tile_centers`, ntiles[0] x ntiles[1] x 2 matrix of tile center
            coordinates

    """
    yy, xx = np.meshgrid(np.arange(ntiles[1]), np.arange(ntiles[0]))

    tile_origins = (
        xx[:, :, np.newaxis] * shift_right + yy[:, :, np.newaxis] * shift_down
    )
    tile_origins -= np.min(tile_origins, axis=(0, 1))[np.newaxis, np.newaxis, :]

    center_offset = np.array([tile_shape[0] / 2, tile_shape[1] / 2])
    tile_centers = tile_origins + center_offset[np.newaxis, np.newaxis, :]

    return tile_origins, tile_centers


def stitch_tiles(
    data_path,
    prefix,
    roi=1,
    suffix="max",
    ich=0,
    correct_illumination=False,
    shifts_prefix=None,
    register_channels=True,
    allow_quick_estimate=False,
):
    """Load and stitch tile images using saved tile shifts.

    This will load the tile shifts saved by `register_within_acquisition`

    Args:
        data_path (str): path to image stacks.
        prefix (str): prefix specifying which images to load, e.g. 'round_01_1'
        roi (int, optional): id of ROI to load. Defaults to 1.
        suffix (str, optional): filename suffix. Defaults to 'fstack'.
        ich (int, optional): index of the channel to stitch. Defaults to 0.
        correct_illumination (bool, optional): Remove black levels and correct
            illumination if True, return raw data otherwise. Default to False
        shifts_prefix (str, optional): prefix to use to load tile shifts. If not
            provided, use `prefix`. Defaults to None.
        register_channels (bool, optional): If True, register channels before stitching.
            Defaults to True.
        allow_quick_estimate (bool, optional): If True, will estimate shifts from a
            single tile if shifts.npz is not found. Defaults to False.

    Returns:
        numpy.ndarray: stitched image.

    """
    processed_path = iss.io.get_processed_path(data_path)
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    if not shifts_prefix:
        shifts_prefix = prefix
    shift_file = processed_path / "reg" / f"{shifts_prefix}_shifts.npz"
    if shift_file.exists():
        shifts = np.load(shift_file)
    elif allow_quick_estimate:
        warnings.warn("Cannot load shifts.npz, will estimate from a single tile")
        ops = load_ops(data_path)
        try:
            metadata = iss.io.load_metadata(data_path)
            ref_roi = list(metadata["ROI"].keys())[0]
        except FileNotFoundError:
            ref_roi = roi_dims[0, 0]
            warnings.warn(f"Metadata file not found, using ROI {ref_roi} as reference.")
        ops_fname = processed_path / "ops.yml"
        if not ops_fname.exists():
            # Change ref tile to a central position where tissue will be
            ops.update(
                {
                    "ref_tile": [
                        ref_roi,
                        round(roi_dims[0, 1] / 2),
                        round(roi_dims[0, 2] / 2),
                    ],
                    "ref_ch": 0,
                }
            )
        shifts = {}
        (
            shifts["shift_right"],
            shifts["shift_down"],
            shifts["tile_shape"],
        ) = register_adjacent_tiles(
            data_path,
            ref_coors=ops["ref_tile"],
            ref_ch=ops["ref_ch"],
            suffix="max",
            prefix=prefix,
        )
    else:
        raise FileNotFoundError(f"Cannot find {shift_file}")

    tile_shape = shifts["tile_shape"]
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles=ntiles
    )
    tile_origins = tile_origins.astype(int)
    max_origin = np.max(tile_origins, axis=(0, 1))
    stitched_stack = np.zeros(max_origin + tile_shape)
    if register_channels:

        def load_func(data_path, tile_coors, prefix):
            stack, _ = iss.pipeline.load_and_register_tile(
                data_path,
                tile_coors,
                prefix=prefix,
                filter_r=False,
                projection=suffix,
                correct_illumination=correct_illumination,
            )
            return stack[:, :, ich, 0]

    else:
        if correct_illumination:
            ops = load_ops(data_path)
            average_image_fname = processed_path / "averages" / f"{prefix}_average.tif"
            average_image = load_stack(average_image_fname)[:, :, ich].astype(float)

        def load_func(data_path, tile_coors, prefix):
            stack = load_tile_by_coors(
                data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
            )[:, :, ich]
            if correct_illumination:
                stack = (stack.astype(float) - ops["black_level"][ich]) / average_image
            return stack

    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            stack = load_func(data_path, (roi, ix, iy), prefix=prefix)
            stitched_stack[
                tile_origins[ix, iy, 0] : tile_origins[ix, iy, 0] + tile_shape[0],
                tile_origins[ix, iy, 1] : tile_origins[ix, iy, 1] + tile_shape[1],
            ] = stack
    return stitched_stack


def stitch_registered(
    data_path, prefix, roi, channels=0, ref_prefix=None, filter_r=False, projection=None
):
    """Load registered stack and stitch them

    The output is in the reference coordinate.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition to stitch
        roi (int): Roi ID
        channels (list or int, optional): Channel id(s). Defaults to 0.
        ref_prefix (str, optional): Prefix of reference acquisition to load shifts. If
            None, load from ops. Defaults to None.
        filter_r (bool, optional): Filter image before stitching? Defaults to False.
        projection (str, optional): Projection to load. If None, will use the one in
            `ops`. Default to None

    Returns:
        np.array: stitched stack

    """
    ops = load_ops(data_path)
    if isinstance(channels, int):
        channels = [channels]
    elif channels is None:
        channels = np.arange(len(ops["camera_order"]))
    else:
        channels = list(channels)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]

    processed_path = iss.io.get_processed_path(data_path)
    if ref_prefix == "genes_round":
        ref_prefix = f"{ref_prefix}_{ops['ref_round']}_1"

    roi_dims = get_roi_dimensions(data_path)
    shifts = np.load(processed_path / "reg" / f"{ref_prefix}_shifts.npz")
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    tile_shape = shifts["tile_shape"]
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles=ntiles
    )
    tile_origins = tile_origins.astype(int)
    max_origin = np.max(tile_origins, axis=(0, 1))
    stitched_stack = np.zeros((*(max_origin + tile_shape), len(channels)))
    if "mask" in prefix:
        # we will increament the mask IDs while we go through
        n_mask_id = 0
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            stack, bad_pixels = load_tile_ref_coors(
                data_path=data_path,
                tile_coors=(roi, ix, iy),
                prefix=prefix,
                filter_r=filter_r,
                projection=projection,
            )
            stack = stack[:, :, channels, 0]  # unique round
            # do not copy 0s over data from previous tile
            if "mask" in prefix:
                bad_pixels = (stack[..., 0] == 0) | bad_pixels
                stack[~bad_pixels, 0] += n_mask_id
                n_mask_id = max(n_mask_id, np.max(stack[..., 0]))
            valid = np.logical_not(bad_pixels)
            stitched_stack[
                tile_origins[ix, iy, 0] : tile_origins[ix, iy, 0] + tile_shape[0],
                tile_origins[ix, iy, 1] : tile_origins[ix, iy, 1] + tile_shape[1],
                :,
            ][valid] = stack[valid]
    return stitched_stack


def merge_roi_spots(
    data_path, prefix, tile_origins, tile_centers, iroi=1, keep_all_spots=False
):
    """Load and combine spot locations across all tiles for an ROI.

    To avoid duplicate spots from tile overlap, we determine which tile center
    each spot is closest to. We then only keep the spots that are closest to
    the center of the tile they were detected on. The tile_centers do not need to be the
    center of the reference tile. For acquisition with a significant shift, it might be
    better to use the center of the acquisition tile registered to the reference.
    See merge_and_align_spots for an example.

    Args:
        data_path (str): path to pickle files containing spot locations for each tile.
        prefix (str): prefix of the spots to load and register (e.g. barcode_round)
        tile_origins (numpy.arry): origin of each tile
        tile_centers (numpy array): center of each tile for ROI duplication detection.
        iroi (int, optional): ID of ROI to load. Defaults to 1.
        keep_all_spots (bool, optional): If True, keep all spots. Otherwise, keep only
            spots which are closer to the tile_centers. Defaults to False.



    Returns:
        pandas.DataFrame: table containing spot locations across all tiles.

    """
    roi_dims = get_roi_dimensions(data_path)
    all_spots = []
    ntiles = roi_dims[roi_dims[:, 0] == iroi, 1:][0] + 1

    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            try:
                spots = align_spots(data_path, tile_coors=(iroi, ix, iy), prefix=prefix)
                spots["x"] = spots["x"] + tile_origins[ix, iy, 1]
                spots["y"] = spots["y"] + tile_origins[ix, iy, 0]

                if not keep_all_spots:
                    # calculate distance to tile centers
                    spot_dist = (
                        spots["x"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 1]
                    ) ** 2 + (
                        spots["y"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 0]
                    ) ** 2
                    home_tile_dist = (spot_dist[:, ix, iy]).copy()
                    spot_dist[:, ix, iy] = np.inf
                    min_spot_dist = np.min(spot_dist, axis=(1, 2))
                    keep_spots = home_tile_dist < min_spot_dist
                else:
                    keep_spots = np.ones(spots.shape[0], dtype=bool)
                all_spots.append(spots[keep_spots])
            except FileNotFoundError:
                print(f"could not load roi {iroi}, tile {ix}, {iy}")

    spots = pd.concat(all_spots, ignore_index=True)
    return spots


def stitch_and_register(
    data_path,
    target_prefix,
    reference_prefix=None,
    roi=1,
    downsample=3,
    ref_ch=0,
    target_ch=0,
    estimate_scale=False,
    estimate_rotation=True,
    target_projection=None,
    use_masked_correlation=False,
    debug=False,
):
    """Stitch target and reference stacks and align target to reference

    To speed up registration, images are downsampled before estimating registration
    parameters. These parameters are then applied to the full scale image.

    The reference stack always use the "projection" from ops as suffix. The target uses
    the same by default but that can be specified with `target_suffix`

    This does not use ops['max_shift_rounds'].

    Args:
        data_path (str): Relative path to data.
        reference_prefix (str): Acquisition prefix to register the stitched image to.
            Typically, "genes_round_1_1".
        target_prefix (str): Acquisition prefix to register.
        roi (int, optional): ROI ID to register (as specified in MicroManager).
            Defaults to 1.
        downsample (int, optional): Downsample factor for estimating registration
            parameter. Defaults to 5.
        ref_ch (int, optional): Channel of the reference image used for registration.
            Defaults to 0.
        target_ch (int, optional): Channel of the target image used for registration.
            Defaults to 0.
        estimate_scale (bool, optional): Whether to estimate scaling between target
            and reference images. Defaults to False.
        estimate_rotation (bool, optional): Whether to estimate rotation between target
            and reference images. Defaults to True.
        target_suffix (str, optional): Suffix to use for target stack. If None, will use
            the value from ops. Defaults to None.
        use_masked_correlation (bool, optional): Use masked correlation for registration.
            Defaults to False.
        debug (bool, optional): If True, return full xcorr. Defaults to False.

    Returns:
        numpy.ndarray: Stitched target image after registration.
        numpy.ndarray: Stitched reference image.
        float: Estimate rotation angle.
        tuple: Estimated X and Y shifts.
        float: Estimated scaling factor.
        dict: Debug information if `debug` is True.
    """
    warnings.warn(
        "stitching is now done on registered tiles", DeprecationWarning, stacklevel=2
    )
    ops = load_ops(data_path)

    if target_projection is None:
        target_projection = ops[f"{target_prefix.split('_')[0].lower()}_projection"]
    if reference_prefix is None:
        reference_prefix = ops["reference_prefix"]

    ref_projection = ops[f"{reference_prefix.split('_')[0].lower()}_projection"]
    if isinstance(target_ch, int):
        target_ch = [target_ch]
    stitched_stack_target = None
    for ch in target_ch:
        stitched = stitch_tiles(
            data_path,
            target_prefix,
            suffix=target_projection,
            roi=roi,
            ich=ch,
            shifts_prefix=reference_prefix,
            correct_illumination=True,
        ).astype(
            np.single
        )  # to save memory
        if stitched_stack_target is None:
            stitched_stack_target = stitched
        else:
            stitched_stack_target += stitched
    stitched_stack_target /= len(target_ch)

    if isinstance(ref_ch, int):
        ref_ch = [ref_ch]
    stitched_stack_reference = None
    for ch in ref_ch:
        stitched = stitch_tiles(
            data_path,
            prefix=reference_prefix,
            suffix=ref_projection,
            roi=roi,
            ich=ch,
            shifts_prefix=reference_prefix,
            correct_illumination=True,
        ).astype(np.single)
        if stitched_stack_reference is None:
            stitched_stack_reference = stitched
        else:
            stitched_stack_reference += stitched
    stitched_stack_reference /= len(ref_ch)

    # If they have different shapes, crop to the smallest size
    if stitched_stack_target.shape != stitched_stack_reference.shape:
        warnings.warn("Stitched stacks have different shapes. Padding to match.")
        stacks_shape = np.vstack(
            (stitched_stack_target.shape, stitched_stack_reference.shape)
        )
        fshape = np.min(stacks_shape, axis=0)
        stitched_stack_target = stitched_stack_target[: fshape[0], : fshape[1]]
        stitched_stack_reference = stitched_stack_reference[: fshape[0], : fshape[1]]
    else:
        fshape = stitched_stack_target.shape

    def prep_stack(stack, downsample):
        if stack.dtype != bool:
            ma = np.nanpercentile(stack, 99)
            stack = np.clip(stack, 0, ma)
            stack = stack / ma
        # downsample
        new_size = np.array(stack.shape) // downsample
        stack = transform.resize(stack, new_size)
        return stack

    # setup common args for registration
    kwargs = dict(
        angle_range=1.0,
        niter=3,
        nangles=11,
        upsample=False,
        debug=debug,
        max_shift=ops["max_shift2ref"] // downsample,
        min_shift=0,
        reference=prep_stack(stitched_stack_reference, downsample),
        target=prep_stack(stitched_stack_target, downsample),
    )
    if use_masked_correlation:
        kwargs["target_mask"] = prep_stack(stitched_stack_target != 0, downsample)
        kwargs["reference_mask"] = prep_stack(stitched_stack_reference != 0, downsample)

    if estimate_scale and estimate_rotation:
        out = estimate_scale_rotation_translation(
            scale_range=0.01,
            **kwargs,
        )
        if debug:
            angle, shift, scale, debug_dict = out
        else:
            angle, shift, scale = out
    elif estimate_rotation:
        out = estimate_rotation_translation(
            **kwargs,
        )
        if debug:
            angle, shift, debug_dict = out
        else:
            angle, shift = out
        scale = 1
    else:
        shift, _, _, _ = mpc.phase_correlation(kwargs["reference"], kwargs["target"])
        scale = 1
        angle = 0
    shift *= downsample

    stitched_stack_target = transform_image(
        stitched_stack_target, scale=scale, angle=angle, shift=shift
    )

    fname = f"{target_prefix}_roi{roi}_tform_to_ref.npz"
    print(f"Saving {fname} in the reg folder")
    np.savez(
        iss.io.get_processed_path(data_path) / "reg" / fname,
        angle=angle,
        shift=shift,
        scale=scale,
        stitched_stack_shape=fshape,
    )
    output = [stitched_stack_target, stitched_stack_reference, angle, shift, scale]
    if debug:
        output.append(debug_dict)
    return tuple(output)


@slurm_it(conda_env="iss-preprocess")
def merge_and_align_spots(
    data_path,
    roi,
    spots_prefix="barcode_round",
    ref_prefix=None,
    keep_all_spots=False,
):
    """Combine spots across tiles and align to reference coordinates for a single ROI.

    For each tile, spots will be registered to the reference coordinates using
    `iss.pipeline.register.align_spots`. The spots will then be merged together using
    `merge_roi_spots`. To avoid duplicate spots, we define a set of tile centers and
    keep only the spots that are closest to the center of the tile they were detected on.


    Args:
        data_path (str): Relative path to data.
        roi (int): ROI ID to process (as specified in MicroManager).
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        ref_prefix (str, optional): Acquisition prefix of the reference acquistion
            to transform spot coordinates to. Defaults to "genes_round_1_1".
        keep_all_spots (bool, optional): If True, keep all spots. Otherwise, keep only
            spots which are closer to the tile_centers. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing all spots in reference coordinates.

    """
    ops = load_ops(data_path)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]
    processed_path = iss.io.get_processed_path(data_path)

    # find tile origin, final shape, and shifts in reference coordinates
    ref_corners = get_tile_corners(data_path, prefix=ref_prefix, roi=roi)
    ref_centers = np.mean(ref_corners, axis=3)
    ref_origins = ref_corners[..., 0]

    # always use the center of the reference tile for spot merging
    # we might have to change that
    trans_centers = ref_centers
    spots = merge_roi_spots(
        data_path,
        prefix=spots_prefix,
        tile_centers=trans_centers,
        tile_origins=ref_origins,
        iroi=roi,
        keep_all_spots=keep_all_spots,
    )

    spots.to_pickle(processed_path / f"{spots_prefix}_spots_{roi}.pkl")
    return spots


def merge_and_align_spots_all_rois(
    data_path,
    spots_prefix="barcode_round",
    ref_prefix="genes_round_1_1",
    keep_all_spots=False,
    dependency=None,
):
    """Start batch jobs to combine spots across tiles and align to reference coordinates
    for all ROIs.

     Args:
        data_path (str): Relative path to data.
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        ref_prefix (str, optional): Acquisition prefix to use as a reference for
            registration. Defaults to "genes_round_1_1".

    """
    ops = load_ops(data_path)
    roi_dims = get_roi_dimensions(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, 0]:
        slurm_folder = Path.home() / "slurm_logs" / data_path / "align_spots"
        slurm_folder.mkdir(exist_ok=True, parents=True)
        merge_and_align_spots(
            data_path,
            roi,
            spots_prefix=spots_prefix,
            ref_prefix=ref_prefix,
            keep_all_spots=keep_all_spots,
            use_slurm=True,
            slurm_folder=slurm_folder,
            scripts_name=f"iss_align_spots_{roi}",
            job_dependency=dependency,
        )
