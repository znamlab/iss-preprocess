from os import system
import numpy as np
import pandas as pd
from skimage.registration import phase_cross_correlation
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..io import load_tile_by_coors, load_stack
from ..reg import (
    estimate_rotation_translation,
    estimate_scale_rotation_translation,
    transform_image,
    make_transform,
)


def register_adjacent_tiles(
    data_path,
    ref_coors=(1, 0, 0),
    reg_fraction=0.1,
    ref_ch=0,
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
    tile_ref = load_tile_by_coors(
        data_path, tile_coors=ref_coors, suffix=suffix, prefix=prefix
    )
    down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + 1)
    tile_down = load_tile_by_coors(
        data_path, tile_coors=down_coors, suffix=suffix, prefix=prefix
    )
    right_coors = (ref_coors[0], ref_coors[1] + 1, ref_coors[2])
    tile_right = load_tile_by_coors(
        data_path, tile_coors=right_coors, suffix=suffix, prefix=prefix
    )
    ypix = tile_ref.shape[0]
    xpix = tile_ref.shape[1]
    reg_pix_x = int(xpix * reg_fraction)
    reg_pix_y = int(ypix * reg_fraction)

    shift_right = phase_cross_correlation(
        tile_ref[:, -reg_pix_x:, ref_ch],
        tile_right[:, :reg_pix_x, ref_ch],
        upsample_factor=5,
    )[0] + [0, xpix - reg_pix_x]

    shift_down = phase_cross_correlation(
        tile_ref[:reg_pix_y, :, ref_ch],
        tile_down[-reg_pix_y:, :, ref_ch],
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
    correct_illumination=False,
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
        correct_illumination (bool, optional): Remove black levels and correct 
            illumination if True, return raw data otherwise. Default to False

    Returns:
        numpy.ndarray: stitched image.

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    # load first tile to get shape
    stack = load_tile_by_coors(
        data_path, tile_coors=(roi, 0, 0), suffix=suffix, prefix=prefix
    )
    tile_shape = stack.shape[:2]

    tile_origins, tile_centers = calculate_tile_positions(
        shift_right, shift_down, tile_shape, ntiles
    )
    tile_origins = tile_origins.astype(int)
    max_origin = np.max(tile_origins, axis=(0, 1))
    stitched_stack = np.zeros(max_origin + tile_shape)
    if correct_illumination:
        ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
        average_image_fname = (
            processed_path / data_path / "averages" / f"{prefix}_average.tif"
        )
        average_image = load_stack(average_image_fname)[:, :, ich].astype(float)
        average_image = average_image / np.max(average_image, axis=(0, 1))
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            stack = load_tile_by_coors(
                data_path, tile_coors=(roi, ix, iy), suffix=suffix, prefix=prefix
            )[:, :, ich]
            if correct_illumination:
                stack = (stack.astype(float) - ops["black_level"][ich]) / average_image
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
            try:
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
            except FileNotFoundError:
                print(f"coult not load roi {iroi}, tile {ix}, {iy}")

    spots = pd.concat(all_spots, ignore_index=True)
    return spots


def stitch_and_register(
    data_path,
    reference_prefix,
    target_prefix,
    roi=1,
    downsample=5,
    ref_ch=0,
    target_ch=0,
    estimate_scale=False,
):
    """Stitch target and reference stacks and align target to reference

    Args:
        data_path (_type_): _description_
        reference_prefix (_type_): _description_
        target_prefix (_type_): _description_
        roi (int, optional): _description_. Defaults to 1.
        downsample (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    # TODO: should we use the same `shift_right` and `shift_down` for target
    # and reference images?
    shift_right, shift_down, tile_shape = register_adjacent_tiles(
        data_path, ref_coors=ops["ref_tile"], prefix=reference_prefix
    )
    stitched_stack_target = stitch_tiles(
        data_path,
        target_prefix,
        shift_right,
        shift_down,
        suffix=ops["projection"],
        roi=roi,
        ich=target_ch,
        correct_illumination=True,
    ).astype(np.single)  # to save memory
    stitched_stack_reference = stitch_tiles(
        data_path,
        reference_prefix,
        shift_right,
        shift_down,
        suffix=ops["projection"],
        roi=roi,
        ich=ref_ch,
        correct_illumination=True,
    ).astype(np.single)
    if estimate_scale:
        scale, angle, shift = estimate_scale_rotation_translation(
            stitched_stack_reference[::downsample, ::downsample],
            stitched_stack_target[::downsample, ::downsample],
            niter=3,
            nangles=11,
            verbose=True,
            scale_range=0.01,
            angle_range=1.0,
            upsample=False,
        )
    else:
        angle, shift = estimate_rotation_translation(
            stitched_stack_reference[::downsample, ::downsample],
            stitched_stack_target[::downsample, ::downsample],
            angle_range=1.0,
            niter=3,
            nangles=11,
            upsample=None,
        )
        scale = 1

    stitched_stack_target = transform_image(
        stitched_stack_target, scale=scale, angle=angle, shift=shift * downsample
    )
    return stitched_stack_target, stitched_stack_reference, angle, shift * downsample


def merge_and_align_spots(
    data_path,
    roi,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()

    ref_prefix = f'genes_round_{ops["ref_round"]+1}_1'
    stitched_stack_barcodes, _, angle, shift = stitch_and_register(
        data_path, ref_prefix, reg_prefix, roi=roi, downsample=5
    )
    spots_tform = make_transform(1, angle, shift, stitched_stack_barcodes.shape)
    shift_right, shift_down, tile_shape = register_adjacent_tiles(
        data_path, ref_coors=ops["ref_tile"], prefix=ref_prefix
    )
    spots = merge_roi_spots(
        data_path,
        shift_right,
        shift_down,
        tile_shape,
        iroi=roi,
        prefix=spots_prefix,
    )
    transformed_coors = spots_tform @ np.stack(
        [spots["x"], spots["y"], np.ones(len(spots))]
    )
    spots["x"] = [x for x in transformed_coors[0, :]]
    spots["y"] = [y for y in transformed_coors[1, :]]
    spots.to_pickle(processed_path / data_path / f"{spots_prefix}_spots_{roi}.pkl")
    np.savez(
        processed_path / data_path / f"{spots_prefix}_spots_tform_{roi}.npz",
        angle=angle,
        shift=shift,
        tform=spots_tform,
    )


def merge_and_align_spots_all_rois(
    data_path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    script_path = str(Path(__file__).parent.parent.parent / f"align_spots.sh")
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, 0]:
        args = f"--export=DATAPATH={data_path},ROI={roi},"
        args += f"SPOTS_PREFIX={spots_prefix},REG_PREFIX={reg_prefix}"
        args += f" --output={Path.home()}/slurm_logs/iss_align_spots_%j.out"
        command = f"sbatch {args} {script_path}"
        print(command)
        system(command)
