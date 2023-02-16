from os import system
import numpy as np
import pandas as pd
import subprocess, shlex
from skimage.registration import phase_cross_correlation
from flexiznam.config import PARAMETERS
from pathlib import Path
from . import pipeline
from ..io import (
    load_tile_by_coors,
    load_stack,
    load_ops,
    get_roi_dimensions,
)
from ..reg import (
    estimate_rotation_translation,
    estimate_scale_rotation_translation,
    transform_image,
    make_transform,
)


def load_tile_ref_coors(data_path, tile_coors, prefix, filter_r=True):
    """Load one single tile

    This load a tile of `prefix` with channels/rounds registered if `coordinate_frame`
    is "local". If `coordinate_frame` is "global" also register to reference acquisition

    Args:
        data_path (str): Relative path to data
        tile_coordinates (tuple): (Roi, tileX, tileY) tuple
        prefix (str): Acquisition to load. If `genes_round` or `barcode_round` will load
            all the rounds.
        filter_r (bool, optional): Apply filter on rounds data? Parameters will be read
            from `ops`. Default to True

    Returns:
        np.array: A (X x Y x Nchannels x Nrounds) registered stack
    """
    stack = pipeline.load_and_register_tile(
        data_path, tile_coors, prefix, filter_r=filter_r
    )

    if prefix == "genes_round_1_1":
        # No need to register to ref
        return stack

    # we have data with channels/rounds registered
    # Now find how much the acquisition stitching is shifting the data compared to
    # reference
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi, tilex, tiley = tile_coors
    ref_corners = get_tile_corners(data_path, prefix="genes_round_1_1", roi=roi)
    acq_corners = get_tile_corners(data_path, prefix=prefix, roi=roi)

    shift = acq_corners[tilex, tiley] - ref_corners[tilex, tiley]
    # shift should be the same for the 4 corners
    assert np.allclose(shift, shift[:, 0, np.newaxis])
    shift = shift[:, 0]

    # now find registration to ref
    reg2ref = np.load(
        processed_path / data_path / "reg" / f"{prefix}_roi{roi}_tform_to_ref.npz"
    )

    # apply the same registration to all channels and rounds
    for ir in range(stack.shape[3]):
        for ic in range(stack.shape[2]):
            stack[:, :, ic, ir] = transform_image(
                stack[:, :, ic, ir],
                scale=reg2ref["scale"],
                angle=reg2ref["angle"],
                shift=reg2ref["shift"] + shift,  # add the stitching shifts
            )
            raise ValueError("THIS IS WRONG: shift should depend on tile position")
    return stack


def register_adjacent_tiles(
    data_path,
    ref_coors=None,
    reg_fraction=0.1,
    ref_ch=0,
    suffix="fstack",
    prefix="genes_round",
):
    """Estimate shift between adjacent imaging tiles using phase correlation.

    Shifts are typically very similar between different tiles, using shifts
    estimated using a reference tile for the whole acquisition works well.

    Args:
        data_path (str): path to image stacks.
        ref_coors (tuple, optional): coordinates of the reference tile to use for
            registration. Must not be along the bottom or right edge of image. If `None`
            use `ops['ref_tile']`. Defaults to None.
        reg_fraction (float, optional): overlap fraction used for registration.
            Defaults to 0.1.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        ref_round (int, optional): reference round used for registration. Defaults to 0.
        nrounds (int, optional): Number of rounds to load. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to 'proj'.
        prefix (str, optional): Full name of the acquisition folder

    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])

    if ref_coors is None:
        ops = load_ops(data_path)
        ref_coors = ops["ref_tile"]

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

    np.savez(
        processed_path / data_path / "reg" / f"{prefix}_shifts.npz",
        shift_right=shift_right,
        shift_down=shift_down,
        tile_shape=tile_ref[:2],
    )

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
    if "round" in prefix:
        # always use round 1
        prefix = f"{prefix.split('_')[0]}_round_1_1"
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    shifts = np.load(processed_path / data_path / "reg" / f"{prefix}_shifts.npz")
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
    suffix="fstack",
    ich=0,
    correct_illumination=False,
    shifts_prefix=None,
):
    """Load and stitch tile images using provided tile shifts.

    This will load the tile shifts saved by `register_within_acquisition`

    Args:
        data_path (str): path to image stacks.
        prefix (str): prefix specifying which images to load, e.g. 'round_01_1'
        roi (int, optional): id of ROI to load. Defaults to 1.
        suffix (str, optional): filename suffix. Defaults to 'proj'.
        ich (int, optional): index of the channel to stitch. Defaults to 0.
        correct_illumination (bool, optional): Remove black levels and correct
            illumination if True, return raw data otherwise. Default to False
        shifts_prefix (str, optional): prefix to use to load tile shifts. If not
            provided, use `prefix`. Defaults to None.

    Returns:
        numpy.ndarray: stitched image.

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    if not shifts_prefix:
        shifts_prefix = prefix
    shifts = np.load(processed_path / data_path / "reg" / f"{shifts_prefix}_shifts.npz")
    tile_shape = shifts["tile_shape"]
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles=ntiles
    )
    max_origin = np.max(tile_origins, axis=(0, 1)).astype(int)
    stitched_stack = np.zeros(max_origin + tile_shape)
    if correct_illumination:
        ops = load_ops(data_path)
        average_image_fname = (
            processed_path / data_path / "averages" / f"{prefix}_average.tif"
        )
        average_image = load_stack(average_image_fname)[:, :, ich].astype(float)
        # TODO: use the illumination corerction function?
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

    To avoid duplicate spots from tile overlap, we determine which tile center
    each spot is closest to. We then only keep the spots that are closest to
    the center of the tile they were detected on.

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
    roi_dims = get_roi_dimensions(data_path)
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

    To speed up registration, images are downsampled before estimating registration
    parameters. These parameters are then applied to the full scale image.

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

    Returns:
        numpy.ndarray: Stitched target image after registration.
        numpy.ndarray: Stitched reference image.
        float: Estimate rotation angle.
        tuple: Estimated X and Y shifts.
    """
    ops = load_ops(data_path)

    stitched_stack_target = stitch_tiles(
        data_path,
        target_prefix,
        suffix=ops["projection"],
        roi=roi,
        ich=target_ch,
        correct_illumination=True,
    ).astype(
        np.single
    )  # to save memory
    stitched_stack_reference = stitch_tiles(
        data_path,
        reference_prefix,
        suffix=ops["projection"],
        roi=roi,
        ich=ref_ch,
        correct_illumination=True,
    ).astype(np.single)

    # If they have different shapes, 0 pad the smallest, keeping origin at (0, 0)
    if stitched_stack_target.shape != stitched_stack_reference.shape:
        stacks_shape = np.vstack(
            (stitched_stack_target.shape, stitched_stack_reference.shape)
        )
        final_shape = np.max(stacks_shape, axis=0)
        padding = final_shape[np.newaxis, :] - stacks_shape
        if padding.max() > 20:
            raise IOError("Large shape difference. Check that everything is fine.")
        if np.sum(padding[0, :]):
            pad_target = [[int(p / 2), int(p / 2) + (p % 2)] for p in padding[0]]
            # if uneven, need to add one after
            stitched_stack_target = np.pad(stitched_stack_target, pad_target)
        if np.sum(padding[1, :]):
            pad_ref = [[int(p / 2), int(p / 2) + (p % 2)] for p in padding[1]]
            stitched_stack_reference = np.pad(stitched_stack_reference, pad_ref)

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
    shift *= downsample

    stitched_stack_target = transform_image(
        stitched_stack_target, scale=scale, angle=angle, shift=shift
    )

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    fname = f"{target_prefix}_roi{roi}_tform_to_ref.npz"
    print(f"Saving {fname} in the reg folder")
    np.savez(
        processed_path / data_path / "reg" / fname,
        angle=angle,
        shift=shift,
        scale=scale,
        stitched_stack_shape=stacks_shape[0],
    )

    return (
        stitched_stack_target,
        stitched_stack_reference,
        angle,
        shift,
        scale,
    )


def merge_and_align_spots(
    data_path,
    roi,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
):
    """Combine spots across tiles and align to reference coordinates for a single ROI.

    We first generate a DataFrame containing all spots in global coordinates
    of the acquisition they were detected in using `merge_roi_spots`. We then
    transform their coordinates into coordinates of the reference genes round
    using the transformation estimated by `stitch_and_register`.

    Args:
        data_path (str): Relative path to data.
        roi (int): ROI ID to process (as specified in MicroManager).
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        reg_prefix (str, optional): Acquisition prefix of the image files to use to
            estimate the tranformation to reference image. Defaults to "barcode_round_1_1".
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    reg_path = processed_path / data_path / "reg"
    tile_corners = get_tile_corners(data_path, prefix=spots_prefix, roi=roi)
    stitched_shape = np.max(tile_corners, axis=(0, 1, 3)).astype(int)

    # find within acq shifts and merge roi spots
    shifts = np.load(reg_path / f"{reg_prefix}_shifts.npz")
    spots = merge_roi_spots(
        data_path,
        shifts["shift_right"],
        shifts["shift_down"],
        shifts["tile_shape"],
        iroi=roi,
        prefix=spots_prefix,
    )

    # get transform to global coordinate and apply to merged spots
    tform2ref = np.load(reg_path / f"{reg_prefix}_tform_to_ref.npz")
    spots_tform = make_transform(
        tform2ref["scale"], tform2ref["angle"], tform2ref["shift"], stitched_shape
    )
    transformed_coors = spots_tform @ np.stack(
        [spots["x"], spots["y"], np.ones(len(spots))]
    )
    spots["x"] = [x for x in transformed_coors[0, :]]
    spots["y"] = [y for y in transformed_coors[1, :]]
    spots.to_pickle(processed_path / data_path / f"{spots_prefix}_spots_{roi}.pkl")


def merge_and_align_spots_all_rois(
    data_path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
    ref_prefix="genes_round_1_1",
):
    """Start batch jobs to combine spots across tiles and align to reference coordinates
    for all ROIs.

     Args:
        data_path (str): Relative path to data.
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        reg_prefix (str, optional): Acquisition prefix of the image files to use to
            estimate the tranformation to reference image. Defaults to "barcode_round_1_1".
        ref_prefix (str, optional): Acquisition prefix to use as a reference for
            registration. Defaults to "genes_round_1_1".
    """
    ops = load_ops(data_path)
    roi_dims = get_roi_dimensions(data_path)
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "align_spots.sh"
    )
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, 0]:
        args = f"--export=DATAPATH={data_path},ROI={roi},"
        args += f"SPOTS_PREFIX={spots_prefix},REG_PREFIX={reg_prefix},REF_PREFIX={ref_prefix}"
        args += f" --output={Path.home()}/slurm_logs/iss_align_spots_%j.out"
        command = f"sbatch {args} {script_path}"
        print(command)
        system(command)
