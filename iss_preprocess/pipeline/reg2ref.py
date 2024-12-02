import numpy as np
from image_tools.similarity_transforms import make_transform
from scipy.ndimage import median_filter
from skimage.morphology import disk
from znamutils import slurm_it

from ..io import get_processed_path, get_roi_dimensions, load_ops
from ..reg import estimate_rotation_translation
from ..vis.diagnostics import check_reg2ref_using_stitched
from .core import batch_process_tiles
from .register import load_and_register_tile
from .stitch import get_tile_corners, register_within_acquisition, stitch_and_register


def register_all_tiles_to_ref(data_path, reg_prefix, use_masked_correlation):
    """Register all tiles to the reference tile

    Args:
        data_path (str): Relative path to data
        reg_prefix (str): Prefix to register, "barcode_round" for instance
        use_masked_correlation (bool): Use masked correlation to register

    Returns:
        list: Job IDs for batch processing

    """
    print("Batch processing all tiles", flush=True)

    roi_dims = get_roi_dimensions(data_path)
    additional_args = (
        f",REG_PREFIX={reg_prefix},"
        + f"USE_MASK={'true' if use_masked_correlation else 'false'}"
    )
    job_ids = batch_process_tiles(
        data_path,
        "register_tile_to_ref",
        additional_args=additional_args,
        roi_dims=roi_dims,
    )
    return job_ids


def register_tile_to_ref(
    data_path,
    tile_coors,
    reg_prefix,
    ref_prefix=None,
    binarise_quantile=None,
    ref_tile_coors=None,
    reg_channels=None,
    ref_channels=None,
    use_masked_correlation=False,
):
    """Register a single tile to the corresponding reference tile

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinates
        reg_prefix (str): Prefix to register, "barcode_round" for instance
        ref_prefix (str, optional): Reference prefix, if None will read from ops.
            Defaults to None.
        binarise_quantile (float, optional): Quantile to binarise images before
            registration. If None will read from ops, Defaults to None.
        ref_tile_coors (tuple, optional): Tile coordinates of the reference tile.
            Usually not needed as it is assumed to be the same as the tile to register.
            Defaults to None.
        reg_channels (list, optional): Channels to use for registration. If None
            will read from ops. Defaults to None
        ref_channels (list, optional): Channels to use for registration. If None will
            read from ops. Defaults to None
        use_masked_correlation (bool, optional): Use masked correlation to register.
            Defaults to False.

    Returns:
        angle (float): Rotation angle
        shifts (np.array): X and Y shifts

    """
    ops = load_ops(data_path)
    # if None, get ref_prefix, ref_channels, binarise_quantile and reg_channels from ops
    if (ref_prefix is None) or (ref_prefix == "None"):
        ref_prefix = ops["reference_prefix"]
    if ref_prefix == reg_prefix:
        raise ValueError("Reference and register prefixes are the same")
    spref = reg_prefix.split("_")[0]  # short prefix
    if ref_channels is None:
        ref_channels = ops["reg2ref_reference_channels"]
        ref_channels = ops.get(f"reg2ref_reference_channels_for_{spref}", ref_channels)

    if binarise_quantile is None:
        binarise_quantile = ops.get(f"{spref}_binarise_quantile", 0.7)
    if reg_channels is None:
        # use either the same as ref or what is in the ops
        reg_channels = ops.get(f"reg2ref_{spref}_channels", ref_channels)
        # if there is something defined for this acquisition, use it instead
        reg_channels = ops.get(f"reg2ref_{reg_prefix}_channels", reg_channels)

    print(f"Registering {reg_prefix} to {ref_prefix}", flush=True)
    if use_masked_correlation:
        print("Using masked correlation", flush=True)
    if ref_tile_coors is None:
        ref_tile_coors = tile_coors
    else:
        print(f"Register to {ref_tile_coors}", flush=True)

    print("Parameters: ")
    print(f"    reg_channels: {reg_channels}")
    print(f"    ref_channels: {ref_channels}")
    print(f"    binarise_quantile: {binarise_quantile}", flush=True)

    # For registration, we don't want to 0 bad pixels. If one round is bad, we will
    # average across others so we don't care, if a channel is bad, we should have
    # signal in the other, if we don't, it's already 0.
    ref_all_channels, ref_bad_pixels = load_and_register_tile(
        data_path=data_path,
        tile_coors=ref_tile_coors,
        prefix=ref_prefix,
        filter_r=False,
        zero_bad_pixels=False,
    )
    reg_all_channels, reg_bad_pixels = load_and_register_tile(
        data_path=data_path,
        tile_coors=tile_coors,
        prefix=reg_prefix,
        filter_r=False,
        zero_bad_pixels=False,
    )

    if ref_channels is not None:
        if isinstance(ref_channels, int):
            ref_channels = [ref_channels]
        ref_all_channels = ref_all_channels[:, :, ref_channels]
    ref = np.nanmean(ref_all_channels, axis=(2, 3))
    ref = np.nan_to_num(ref)

    if reg_channels is not None:
        if isinstance(reg_channels, int):
            reg_channels = [reg_channels]
        reg_all_channels = reg_all_channels[:, :, reg_channels]
    reg = np.nanmean(reg_all_channels, axis=(2, 3))
    reg = np.nan_to_num(reg)

    if ops["reg_median_filter"]:
        ref = median_filter(ref, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))
        reg = median_filter(reg, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))

    if binarise_quantile is not None:
        reg = reg > np.quantile(reg, binarise_quantile)
        ref = ref > np.quantile(ref, binarise_quantile)

    angle, shift = estimate_rotation_translation(
        ref,
        reg,
        angle_range=1.0,
        niter=3,
        nangles=15,
        max_shift=ops["rounds_max_shift"],
        reference_mask=~ref_bad_pixels if use_masked_correlation else None,
        target_mask=~reg_bad_pixels if use_masked_correlation else None,
    )
    print(f"Angle: {angle}, Shifts: {shift}")
    # make it into affine matrix
    tforms = make_transform(s=1, angle=angle, shift=shift, shape=reg.shape[:2])
    processed_path = get_processed_path(data_path)
    r, x, y = tile_coors
    target = processed_path / "reg" / f"tforms_to_ref_{reg_prefix}_{r}_{x}_{y}.npz"
    # reshape tforms to be like the multichannels tforms
    np.savez(target, matrix_between_channels=tforms.reshape((1, 3, 3)))
    print(f"Saved tforms to {target}", flush=True)
    return tforms


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
    if ops["corrected_shifts"] == "single_tile":
        corrected_shifts = ""
    elif ops["corrected_shifts"] == "ransac":
        corrected_shifts = "_corrected"
    elif ops["corrected_shifts"] == "best":
        corrected_shifts = "_best"
    else:
        raise ValueError(f"Corrected shifts {ops['corrected_shifts']} not recognised")
    processed_path = get_processed_path(data_path)
    tform2ref = np.load(
        processed_path
        / "reg"
        / f"tforms{corrected_shifts}_to_ref_{prefix}_{roi}_{tilex}_{tiley}.npz"
    )
    return tform2ref


@slurm_it(conda_env="iss-preprocess", print_job_id=True, slurm_options=dict(mem="72G"))
def register_to_ref_using_stitched_registration(
    data_path,
    roi,
    reg_prefix,
    ref_prefix=None,
    ref_channels=None,
    reg_channels=None,
    estimate_rotation=True,
    target_suffix=None,
    use_masked_correlation=False,
    downsample=5,
    save_plot=True,
):
    """Register all tiles to the reference using the stitched registration

    This will stitch both the reference and target tiles using the reference shifts,
    then register the stitched target to the stitched reference to get the best
    similarity transform.
    Then the transformation is applied to each tile and saved instead of the one
    generated by "register_tile_to_ref".

    Args:
        data_path (str): Relative path to data
        roi (int): ROI to register
        target_prefix (str): Prefix of the target tile
        ref_prefix (str, optional): Prefix of the reference tile. If None, reads
            from ops. Defaults to None.
        ref_channels (list, optional): Channels to use for registration. If None
            will read from ops. Defaults to None.
        reg_channels (list, optional): Channels to use for registration. If None
            will read from ops. Defaults to None.
        estimate_rotation (bool, optional): Estimate rotation. Defaults to True.
        target_suffix (str, optional): Suffix of the target tile. Defaults to None.
        use_masked_correlation (bool, optional): Use masked correlation. Defaults to
            False.
        downsample (int, optional): Downsample factor. Defaults to 3.
        save_plot (bool, optional): Save a diagnostic plot. Defaults to True.

    Returns:
        None

    """
    ops = load_ops(data_path)
    if (ref_prefix is None) or (ref_prefix == "None"):
        ref_prefix = ops["reference_prefix"]
    if ref_prefix == reg_prefix:
        raise ValueError("Reference and register prefixes are the same")
    if ref_channels is None:
        ref_channels = ops["reg2ref_reference_channels"]
    spref = reg_prefix.split("_")[0]  # short prefix
    if reg_channels is None:
        # use either the same as ref or what is in the ops
        reg_channels = ops.get(f"reg2ref_{spref}_channels", ref_channels)
        # if there is something defined for this acquisition, use it instead
        reg_channels = ops.get(f"reg2ref_{reg_prefix}_channels", reg_channels)

    # get the transformation from the stitched image to the reference
    print(f"Registering {reg_prefix} to {ref_prefix} for ROI {roi}")
    print(f"    mask: {use_masked_correlation}")
    print(f"    ref_channels: {ref_channels}")
    print(f"    reg_channels: {reg_channels}")
    print(f"    estimate_rotation: {estimate_rotation}")
    print(f"    downsample: {downsample}")
    print(f"    save_plot: {save_plot}")

    # first register within if needed
    register_within_acquisition(
        data_path,
        prefix=ref_prefix,
        roi=roi,
        reload=True,
        save_plot=True,
        use_slurm=False,
    )
    register_within_acquisition(
        data_path,
        prefix=reg_prefix,
        roi=roi,
        reload=True,
        save_plot=True,
        use_slurm=False,
    )

    (
        stitched_stack_target,
        stitched_stack_reference,
        angle,
        shift,
        scale,
    ) = stitch_and_register(
        data_path,
        reference_prefix=ref_prefix,
        target_prefix=reg_prefix,
        roi=roi,
        downsample=downsample,
        ref_ch=ref_channels,
        target_ch=reg_channels,
        estimate_scale=False,  # never estimate scale
        estimate_rotation=estimate_rotation,
        target_projection=target_suffix,
        use_masked_correlation=use_masked_correlation,
        debug=False,
    )
    print(f"Angle: {angle}, Shifts: {shift}, Scale: {scale}")
    # transform the center of each tile
    tform2ref = make_transform(
        scale,
        angle,
        shift,
        stitched_stack_target.shape[:2],
    )
    reg_corners = get_tile_corners(data_path, reg_prefix, roi)
    tile_shape = reg_corners[0, 0, :, 2] - reg_corners[0, 0, :, 0]
    ref_centers = np.mean(reg_corners, axis=3)
    trans_centers = np.pad(ref_centers, ((0, 0), (0, 0), (0, 1)), constant_values=1)
    trans_centers = (
        tform2ref[np.newaxis, np.newaxis, ...] @ trans_centers[..., np.newaxis]
    )
    trans_centers = trans_centers[..., :-1, 0]

    # make tile by tile transformation from that
    for tilex in range(trans_centers.shape[0]):
        for tiley in range(trans_centers.shape[1]):
            shift_tile = trans_centers[tilex, tiley] - ref_centers[tilex, tiley]
            # this is a col/row shift, flip to x/y
            shift_tile = shift_tile[::-1]
            tforms = make_transform(1, angle, shift_tile, tile_shape)
            processed_path = get_processed_path(data_path)

            target = (
                processed_path
                / "reg"
                / f"tforms_to_ref_{reg_prefix}_{roi}_{tilex}_{tiley}.npz"
            )
            # reshape tforms to be like the multichannels tforms
            np.savez(target, matrix_between_channels=tforms.reshape((1, 3, 3)))

    if save_plot:
        save_folder = get_processed_path(data_path) / "figures" / "registration"
        save_folder /= f"{reg_prefix}_to_{ref_prefix}"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_path = save_folder / f"{reg_prefix}_to_{ref_prefix}_roi_{roi}.png"
        check_reg2ref_using_stitched(
            save_path,
            stitched_stack_reference,
            stitched_stack_target,
            ref_centers,
            trans_centers,
        )

    print("Done")
