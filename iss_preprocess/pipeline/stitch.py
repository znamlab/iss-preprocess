from os import system
import numpy as np
import pandas as pd
import subprocess, shlex
from skimage.registration import phase_cross_correlation
from flexiznam.config import PARAMETERS
from pathlib import Path
from . import pipeline
from ..image.correction import apply_illumination_correction
from ..io import (
    load_tile_by_coors,
    load_stack,
    load_ops,
    get_roi_dimensions,
    load_metadata,
)
from .sequencing import load_and_register_tile
from .hybridisation import load_and_register_hyb_tile
from ..reg import (
    estimate_rotation_translation,
    estimate_scale_rotation_translation,
    transform_image,
    make_transform,
)


def register_acquisitions(data_path, which, prefix, by_tiles=False):
    """Start bash job to register all ROIs

    Args:
        data_path (str): Relative path to data
        which (str): "within" or "across" for acquisition registration and registration
            to reference acquisiton respectively
        by_tiles (bool, optional): Register across using single tiles instead of
            stitched image. Defaults to False.
    """

    ops = load_ops(data_path)

    if which.lower() == "within":
        script_name = "register_within_acquisition"
        rois_to_do = [None]
    elif which.lower() == "across":
        script_name = "register_across_acquisitions"
        rois_to_do = ops["use_rois"]
    else:
        raise IOError("`which` must be 'within' or 'across'")

    export_args = dict(PREFIX=prefix)
    if by_tiles:
        arguments = ",".join([f"{k}={v}" for k, v in export_args.items()])
        pipeline.batch_process_tiles(
            data_path, script_name, additional_args="," + arguments
        )
    else:
        export_args["DATAPATH"] = data_path
        script_path = str(
            Path(__file__).parent.parent.parent / "scripts" / f"{script_name}.sh"
        )
        for roi in rois_to_do:
            if roi is not None:  # within. we register only one tile in one roi
                export_args["ROI"] = roi
            args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
            args = (
                args
                + f" --output={Path.home()}/slurm_logs/iss_reg_{which}_%j.out"
                + f" --error={Path.home()}/slurm_logs/iss_reg_{which}_%j.err"
            )
            command = f"sbatch {args} {script_path}"
            print(command)
            subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )


def load_tile(
    data_path, tile_coordinates, prefix, coordinate_frame="global", filter_r=True
):
    """Load one single tile

    This load a tile of `prefix` with channels/rounds registered if `coordinate_frame`
    is "local". If `coordinate_frame` is "global" also register to reference acquisition

    Args:
        data_path (str): Relative path to data
        tile_coordinates (tuple): (Roi, tileX, tileY) tuple
        prefix (str): Acquisition to load. If `genes_round` or `barcode_round` will load
            all the rounds.
        coordinate_frame (str, optional): Either "local" or "global". Defaults to
            "global".
        filter_r (bool, optional): Apply filter on rounds data? Parameters will be read
            from `ops`. Default to True

    Returns:
        np.array: A (X x Y x Nchannels x Nrounds) registered stack
    """
    assert coordinate_frame in ("local", "global")

    ops = load_ops(data_path)
    metadata = load_metadata(data_path)

    if filter_r:
        filter_r = ops["filter_r"]
    if prefix.startswith("genes_round") or prefix.startswith("barcode_round"):
        parts = prefix.split("_")
        if len(parts) > 2:
            acq_type = "_".join(parts[:2])
            rounds = np.array([int(parts[2])])
        else:
            acq_type = prefix
            rounds = np.arange(ops[f"{acq_type}s"]) + 1

        stack, bad_pixel = load_and_register_tile(
            data_path,
            tile_coors=tile_coordinates,
            suffix=ops["projection"],
            prefix=acq_type,
            filter_r=filter_r,
            correct_channels=True,
            correct_illumination=True,
            corrected_shifts=True,
            specific_rounds=rounds,
        )
        # the transforms for all rounds are the same and saved with round 1
        prefix = acq_type + "_1_1"

    elif prefix in metadata["hybridisation"]:
        stack, bad_pixel = load_and_register_hyb_tile(
            data_path,
            tile_coors=tile_coordinates,
            prefix=prefix,
            suffix=ops["hybridisation_projection"],
            filter_r=filter_r,
            correct_illumination=True,
            correct_channels=True,
        )
        stack = np.array(stack, ndmin=4)
    else:
        stack = load_tile_by_coors(
            data_path,
            tile_coors=tile_coordinates,
            suffix=ops["projection"],
            prefix=prefix,
        )
        bad_pixel = np.zeros(stack.shape, dtype=bool)
        stack = apply_illumination_correction(data_path, stack, prefix)

    stack[bad_pixel] = 0

    # ensure we have 4d to match acquisitions with rounds
    if stack.ndim == 3:
        stack = stack[..., np.newaxis]

    if coordinate_frame == "local" or prefix == "genes_round_1_1":
        # No need to register to ref
        return stack

    # we have data with channels/rounds registered
    # Now find how much the acquisition stitching is shifting the data compared to
    # reference
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi, tilex, tiley = tile_coordinates
    ref_corners = get_tiles_corners(data_path, prefix="genes_round_1_1", roi=roi)
    acq_corners = get_tiles_corners(data_path, prefix=prefix, roi=roi)

    shift = acq_corners[tilex, tiley] - ref_corners[tilex, tiley]
    # shift should be the same for the 4 corners
    assert np.allclose(shift, shift[:, 0, np.newaxis])
    shift = shift[:, 0]

    # now find registration to ref
    reg2ref = np.load(
        processed_path / data_path / "reg" / f"{prefix}_roi{roi}_shifts_to_global.npz"
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
    return stack


def register_within_acquisition(data_path, prefix):
    """Save registration of a single acquisition

    This is for stitching and does not register across channel.
    This saves "{prefix}_shifts.npz" which contains the information need to stitch tiles
    together in the acquisition coordinates

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisiton prefix
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    shift_right, shift_down, tile_shape = register_adjacent_tiles(
        data_path, prefix=prefix
    )
    np.savez(
        processed_path / data_path / "reg" / f"{prefix}_shifts.npz",
        shift_right=shift_right,
        shift_down=shift_down,
        tile_shape=tile_shape,
    )


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
        prefix (str, optional): the folder name prefix, before round number.
            Defaults to "round"

    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
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

    return shift_right, shift_down, (ypix, xpix)


def get_tiles_corners(data_path, prefix, roi):
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

    Returns:
        numpy.ndarray: stitched image.

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1

    shifts = np.load(processed_path / data_path / "reg" / f"{prefix}_shifts.npz")
    tile_shape = shifts["tile_shape"]
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles=ntiles
    )
    max_origin = np.max(tile_origins, axis=(0, 1))
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


def register_across_acquisitions(
    data_path,
    prefix,
    roi,
    ref_ch=0,
    target_ch=0,
    reference_prefix="genes_round_1_1",
    estimate_scale=False,
    tilex=None,
    tiley=None,
):
    """Register an acquisition to the reference acquisition

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition to register
        roi (int): ROI to register
        ref_ch (int, optional): Channel from reference to use. Defaults to 0.
        target_ch (int, optional): Channel from target to use. Defaults to 0.
        reference_prefix (str, optional): Reference acquisition. Defaults to
            "genes_round_1_1".
        estimate_scale (bool, optional): Estimate scale if True. Only shift and angle
            otherwise. Defaults to True
        tilex (int, optional): X of tile to register. If not provided will use the whole
            stitched acquisition. Defaults to None.
        tiley (int, optional): Y of tile to register. If not provided will use the whole
            stitched acquisition. Defaults to None.
    """
    input_args = locals()
    for k, v in input_args.items():
        if not k.startswith("_"):
            print(f"{k} = {v}")
    print("", flush=True)

    if tilex is None:
        assert tiley is None
        print("Stitch and register", flush=True)
        _, _, angle, shift, scale = stitch_and_register(
            data_path,
            reference_prefix=reference_prefix,
            target_prefix=prefix,
            roi=roi,
            downsample=5,
            ref_ch=ref_ch,
            target_ch=target_ch,
            estimate_scale=estimate_scale,
        )
    else:
        assert tiley is not None
        print("Register single tile", flush=True)
        angle, shift, scale = register_single_tile(
            data_path,
            reference_prefix=reference_prefix,
            target_prefix=prefix,
            tile_coordinates=(roi, tilex, tiley),
            ref_ch=ref_ch,
            target_ch=target_ch,
            estimate_scale=estimate_scale,
        )

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    tilecoor = "" if tilex is None else f"{tilex}_{tiley}_"
    fname = f"{prefix}_roi{roi}_{tilecoor}shifts_to_global.npz"
    print(f"Saving {fname} in the reg folder")
    np.savez(
        processed_path / data_path / "reg" / fname,
        angle=angle,
        shift=shift,
        scale=scale,
    )


def register_single_tile(
    data_path,
    target_prefix,
    tile_coordinates,
    reference_prefix="genes_round_1_1",
    ref_ch=0,
    target_ch=0,
    estimate_scale=False,
):
    """Register a single tile to the corresponding reference acquisition tile

    Args:
        data_path (str): Relative path to data
        target_prefix (str): Acquisition to register
        tile_coordinates (tuple): (ROI, tileX, tileY) coordinates of tile)
        reference_prefix (str, optional): Reference acquisition. Defaults to
            "genes_round_1_1"
        ref_ch (int, optional): Reference channel. Defaults to 0.
        target_ch (int, optional): Target channel. Defaults to 0.
        estimate_scale (bool, optional): Estimate scale if True. Defaults to False.

    Returns:
        angle, shift, scale: Transform parameters
    """
    ref_tile = load_tile(
        data_path, tile_coordinates, reference_prefix, coordinate_frame="local"
    )
    target_tile = load_tile(
        data_path, tile_coordinates, target_prefix, coordinate_frame="local"
    )
    # TODO binarise somehow and use all channels instead of one
    if estimate_scale:
        scale, angle, shift = estimate_scale_rotation_translation(
            ref_tile[:, :, ref_ch, 0],
            target_tile[:, :, target_ch, 0],
            niter=3,
            nangles=11,
            verbose=True,
            scale_range=0.01,
            angle_range=1.0,
            upsample=False,
        )
    else:
        angle, shift = estimate_rotation_translation(
            ref_tile[:, :, ref_ch, 0],
            target_tile[:, :, target_ch, 0],
            angle_range=1.0,
            niter=3,
            nangles=11,
            upsample=None,
        )
        scale = 1
    return angle, shift, scale


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
        if np.sum(padding[0, :]):
            stitched_stack_target = np.pad(
                stitched_stack_target, [(0, p) for p in padding[0]]
            )
        if np.sum(padding[1, :]):
            stitched_stack_reference = np.pad(
                stitched_stack_reference, [(0, p) for p in padding[1]]
            )

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
    return (
        stitched_stack_target,
        stitched_stack_reference,
        angle,
        shift * downsample,
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
    ops = load_ops(data_path)

    ref_prefix = f'genes_round_{ops["ref_round"]+1}_1'
    stitched_stack_barcodes, _, angle, shift, scale = stitch_and_register(
        data_path, ref_prefix, reg_prefix, roi=roi, downsample=5
    )
    spots_tform = make_transform(scale, angle, shift, stitched_stack_barcodes.shape)
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
    """Start batch jobs to combine spots across tiles and align to reference coordinates
    for all ROIs.

     Args:
        data_path (str): Relative path to data.
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        reg_prefix (str, optional): Acquisition prefix of the image files to use to
            estimate the tranformation to reference image. Defaults to "barcode_round_1_1".
    """
    ops = load_ops(data_path)
    roi_dims = get_roi_dimensions(data_path)
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "align_spots.sh"
    )
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, 0]:
        args = f"--export=DATAPATH={data_path},ROI={roi},"
        args += f"SPOTS_PREFIX={spots_prefix},REG_PREFIX={reg_prefix}"
        args += f" --output={Path.home()}/slurm_logs/iss_align_spots_%j.out"
        command = f"sbatch {args} {script_path}"
        print(command)
        system(command)
