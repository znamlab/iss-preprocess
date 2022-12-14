import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..io import load_stack
from .pipeline import load_processed_tile


def register_adjacent_tiles(
    data_path,
    ref_coors=(1, 0, 0),
    reg_fraction=0.1,
    ref_ch=0,
    ref_round=0,
    nrounds=7,
    suffix="fstack",
    prefix="genes_round",
):
    """Estimate shift between adjacent imaging tiles using phase correlation.

    Args:
        data_path (str): path to image stacks.
        ref_coors (tuple, optional): coordinates of the reference tile to use for
            registration. Must not be along the bottom or right edge of image. Defaults to (1,0,0).
        reg_fraction (float, optional): overlap fraction used for registration. Defaults to 0.1.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        ref_round (int, optional): reference round used for registration. Defaults to 0.
        nrounds (int, optional): Number of rounds to load. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to 'proj'.
        prefix (str, optional): the folder name prefix, before round number. Defaults to "round"

    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
    tile_ref = load_processed_tile(
        data_path, ref_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )
    down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + 1)
    tile_down = load_processed_tile(
        data_path, down_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )
    right_coors = (ref_coors[0], ref_coors[1] + 1, ref_coors[2])
    tile_right = load_processed_tile(
        data_path, right_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )

    ypix = tile_ref.shape[0]
    xpix = tile_ref.shape[1]
    reg_pix_x = int(xpix * reg_fraction)
    reg_pix_y = int(ypix * reg_fraction)

    shift_right = phase_cross_correlation(
        tile_ref[:, -reg_pix_x:, ref_ch, ref_round],
        tile_right[:, :reg_pix_x, ref_ch, ref_round],
        upsample_factor=5,
    )[0] + [0, xpix - reg_pix_x]

    shift_down = phase_cross_correlation(
        tile_ref[:reg_pix_y, :, ref_ch, ref_round],
        tile_down[-reg_pix_y:, :, ref_ch, ref_round],
        upsample_factor=5,
    )[0] - [ypix - reg_pix_y, 0]

    return shift_right, shift_down, (ypix, xpix)


def calculate_tile_positions(shift_right, shift_down, tile_shape, ntiles):
    """Calculate position of each tile based on the provided shifts.

    Args:
        shift_right (numpy.array): X and Y shifts between different columns
        shift_down (numpy.array): X and Y shifts between different rows
        tile_shape (numpy.array): shape of each tile
        ntiles (numpy.array): number of tile rows and columns

    Returns:
        numpy.ndarray: `tile_origins`, ntiles[0] x ntiles[1] x 2 matrix of tile origin coordinates
        numpy.ndarray: `tile_centers`, ntiles[0] x ntiles[1] x 2 matrix of tile center coordinates

    """
    tile_centers = np.empty((ntiles[0], ntiles[1], 2))
    tile_origins = np.empty((ntiles[0], ntiles[1], 2))

    center_offset = np.array([tile_shape[0] / 2, tile_shape[1] / 2])
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            tile_origins[ix, iy, :] = iy * shift_down + ix * shift_right
    tile_origins = (
        tile_origins - np.min(tile_origins, axis=(0, 1))[np.newaxis, np.newaxis, :]
    )
    tile_centers = tile_origins + center_offset[np.newaxis, np.newaxis, :]
    return tile_origins, tile_centers


def stitch_tiles(
    data_path,
    prefix,
    shift_right,
    shift_down,
    roi=1,
    suffix="fstack",
    ich=0,
    black_level=0,
    correction_image=None,
):
    """Load and stitch tile images using provided tile shifts.

    Args:
        data_path (str): path to image stacks.
        prefix (str): prefix specifying which images to load, e.g. 'round_01_1'
        shift_right (tuple): x and y shift between tiles along a row
        shift_down (tuple): x and y shift between tiles along a column
        roi (int, optional): id of ROI to load. Defaults to 1.
        suffix (str, optional): filename suffix. Defaults to 'proj'.
        ich (int, optional): index of the channel to stitch. Defaults to 0.
        black_level (int, optional): the black level for thischannel. Defaults to 0.
        correction_image (np.array, optional): Image for illumination correction for 
            channel `ich`. Defaults to None (no correction)

    Returns:
        numpy.ndarray: stitched image.

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    # load first tile to get shape
    stack = load_stack(
        processed_path
        / data_path
        / prefix
        / f"{prefix}_MMStack_{roi}-Pos{str(0).zfill(3)}_{str(0).zfill(3)}_{suffix}.tif"
    )
    tile_shape = stack.shape[:2]

    tile_origins, tile_centers = calculate_tile_positions(
        shift_right, shift_down, tile_shape, ntiles
    )
    tile_origins = tile_origins.astype(int)
    max_origin = np.max(tile_origins, axis=(0, 1))
    stitched_stack = np.zeros(max_origin + tile_shape)
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            fname = f"{prefix}_MMStack_{roi}-Pos{str(ix).zfill(3)}_{str(iy).zfill(3)}_{suffix}.tif"
            stack = load_stack(processed_path / data_path / prefix / fname)[:, :, ich]
            # do correction as float and convert back, cliping to avoid overflow
            dtype = stack.dtype
            dtype_info = np.iinfo(dtype)
            stack = np.array(stack, dtype=float) - float(black_level)
            if correction_image is not None:
                stack /= correction_image
            stack = np.clip(stack, dtype_info.min, dtype_info.max)
            stack = np.array(stack, dtype=dtype)
                
            stitched_stack[
                tile_origins[ix, iy, 0] : tile_origins[ix, iy, 0] + tile_shape[0],
                tile_origins[ix, iy, 1] : tile_origins[ix, iy, 1] + tile_shape[1],
            ] = stack
    return stitched_stack


def merge_roi_spots(
    data_path, shift_right, shift_down, tile_shape, iroi=1, prefix="genes_round"
):
    """Load and combine spot locations across all tiles for an ROI.

    Args:
        data_path (str): path to pickle files containing spot locations for each tile.
        shift_right (numpy.array): X and Y shifts between different columns
        shift_down (numpy.array): X and Y shifts between different rows
        tile_shape (numpy.array): shape of each tile
        iroi (int, optional): ID of ROI to load. Defaults to 1.

    Returns:
        pandas.DataFrame: table containing spot locations across all tiles.
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    all_spots = []
    ntiles = roi_dims[roi_dims[:, 0] == iroi, 1:][0] + 1
    tile_origins, tile_centers = calculate_tile_positions(
        shift_right, shift_down, tile_shape, ntiles
    )

    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            spots = pd.read_pickle(
                processed_path
                / data_path
                / "spots"
                / f"{prefix}_spots_{iroi}_{ix}_{iy}.pkl"
            )
            spots["x"] = spots["x"] + tile_origins[ix, iy, 1]
            spots["y"] = spots["y"] + tile_origins[ix, iy, 0]

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
            all_spots.append(spots[keep_spots])

    spots = pd.concat(all_spots, ignore_index=True)
    return spots
