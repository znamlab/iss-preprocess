from pathlib import Path
from warnings import warn

import numpy as np
from image_tools.similarity_transforms import make_transform
from scipy.ndimage import median_filter
from skimage.morphology import disk
from skimage.transform import SimilarityTransform
from sklearn.linear_model import RANSACRegressor
from znamutils import slurm_it

from ..diagnostics.diag_register import check_tile_shifts
from ..image.correction import apply_illumination_correction
from ..io import (
    get_channel_round_transforms,
    get_processed_path,
    get_raw_filename,
    get_raw_path,
    get_roi_dimensions,
    get_tile_ome,
    load_metadata,
    load_ops,
    load_tile_by_coors,
)
from ..reg import (
    apply_corrections,
    estimate_affine_for_tile,
    estimate_shifts_and_angles_for_tile,
    estimate_shifts_for_tile,
    register_channels_and_rounds,
)
from ..vis.diagnostics import plot_registration_correlograms
from .hybridisation import get_channel_shifts, load_and_register_hyb_tile
from .sequencing import load_and_register_sequencing_tile, load_sequencing_rounds


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="64G"))
def run_register_reference_tile(data_path, prefix="genes_round", diag=False):
    """Subfunction to run the registration of the reference tile

    This function actually perform the computation. It performs the registration of the
    the reference tile specified inn the ops. This include shifts and rotations between
    rounds and shifts, rotations, and scaling between channels.

    Shifts are estimated using phase correlation. Rotation and scaling are estimated
    using iterative grid search.

    Results are saved in a npz file in the processed directory in:
    data_path / 'reg' / `prefix` / 'ref_tile_tforms_`prefix`_round.npz'

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Directory prefix to register.
            Defaults to "genes_round".
        diag (bool, optional): Whether to save diagnostic plots.

    """
    if diag:
        diag_plot_dir = get_processed_path(data_path) / "figures" / "ref_tile"
        diag_plot_dir.mkdir(parents=True, exist_ok=True)
    ops = load_ops(data_path)
    nrounds = ops[prefix + "s"]
    projection = ops[f"{prefix.split('_')[0].lower()}_projection"]
    stack = load_sequencing_rounds(
        data_path, ops["ref_tile"], prefix=prefix, suffix=projection, nrounds=nrounds
    )
    if ops["align_method"] == "affine":
        affine_by_block = True
    elif ops["align_method"] == "similarity":
        affine_by_block = False
    else:
        raise ValueError(f"Align method {ops['align_method']} not recognised")

    out = register_channels_and_rounds(
        stack,
        ref_ch=ops["ref_ch"],
        ref_round=ops["ref_round"],
        median_filter=ops["reg_median_filter"],
        max_shift=ops["rounds_max_shift"],
        min_shift=ops["rounds_min_shift"],
        debug=diag,
        use_masked_correlation=ops["use_masked_correlation"],
        affine_by_block=affine_by_block,
    )
    if diag:
        (
            angles_within_channels,
            shifts_within_channels,
            matrix_between_channels,
            debug_dict,
        ) = out
        plot_registration_correlograms(
            data_path,
            prefix,
            "register_reference_tile",
            debug_dict,
        )
    else:
        (
            angles_within_channels,
            shifts_within_channels,
            matrix_between_channels,
        ) = out

    save_path = get_channel_round_transforms(
        data_path, prefix, tile_coors=None, shifts_type="reference", load_file=False
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        angles_within_channels=angles_within_channels,
        shifts_within_channels=shifts_within_channels,
        matrix_between_channels=matrix_between_channels,
        allow_pickle=True,
    )
    print(f"Saved tforms to {save_path}")


def register_fluorescent_tile(
    data_path,
    tile_coors,
    prefix,
    reference_prefix=None,
    debug=False,
    save_output=True,
):
    """Estimate channel registration parameters for a single round acquisition

    The stack will be binarised if ops[f"{prefix_start}_binarise_quantile"] is not None.
    The scale and initial parameters will be loaded from the reference prefix and
    optimised using either a similarity transform or an affine transform, depending
    on ops["align_method"].

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of tile to register, in (ROI, X, Y) format.
        prefix (str): Directory prefix to register. Defaults to
        reference_prefix (str, optional): Prefix to load scale or initial matrix from.
            Defaults to None.
        debug (bool, optional): Return debug information. Defaults to False.
        save_output (bool, optional): Save output to disk. Defaults to True.

    Returns:

        dict: Debug information if debug is True, None otherwise.
    """

    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    ops_prefix = prefix.split("_")[0].lower()
    projection = ops[f"{ops_prefix}_projection"]
    projection = ops.get(f"{ops_prefix}_reg_projection", projection)
    print("Projection used for registration:", projection)
    if reference_prefix is not None:
        tforms_path = processed_path / f"tforms_{reference_prefix}.npz"
        reference_tforms = np.load(tforms_path, allow_pickle=True)
    else:
        reference_tforms = None

    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=projection, prefix=prefix
    )
    correct_illumination = ops.get(f"{ops_prefix}_reg_correct_illumination", False)
    if correct_illumination:
        print("Correcting illumination")
        stack = apply_illumination_correction(data_path, stack, prefix)
    else:
        print("Not correcting illumination")

    # median filter if needed
    median_filter_size = ops["reg_median_filter"]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        stack = median_filter(stack, footprint=disk(median_filter_size), axes=(0, 1))

    binarise_quantile = ops[prefix.split("_")[0].lower() + "_binarise_quantile"]
    ref_ch = ops["ref_ch"]
    ref_ch = ops.get(f"{prefix.split('_')[0].lower()}_ref_ch", ref_ch)

    channel_grouping = ops.get(f"{ops_prefix}_reg_channel_grouping", None)
    if channel_grouping is None:
        print("Registering all channels together")
        out = _reg_chans(
            ops,
            ops_prefix,
            stack,
            reference_prefix,
            binarise_quantile,
            reference_tforms,
            ref_ch,
            debug,
        )
    else:
        print(f"Registering channels by pairs: {channel_grouping}")
        out = register_channels_by_pairs(
            channel_grouping,
            ops,
            ops_prefix,
            stack,
            reference_prefix,
            binarise_quantile,
            reference_tforms,
            debug,
        )
    if debug:
        to_save, db_info = out
    else:
        to_save = out
    if save_output:
        save_dir = processed_path / "reg" / prefix
        save_dir.mkdir(parents=True, exist_ok=True)
        target = f"tforms_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
        np.savez(
            save_dir / target,
            allow_pickle=True,
            **to_save,
        )
        print(f"Saved tforms to {save_dir / target}")

    if debug:
        return to_save, db_info
    return to_save


def register_channels_by_pairs(
    channel_grouping,
    ops,
    ops_prefix,
    stack,
    reference_prefix,
    binarise_quantile,
    reference_tforms,
    debug=False,
):
    """Register channels for a single tile iteratively by group of channels

    channel_grouping must be a list of list (of list ....). The inner most levels will
    be registered together, using the first channel of the list as reference. Then the
    upper level will be registered together.
    For instance, if channel_grouping = [[0, 1], [2, 3]], channels 0 and 1 will be
    registered together (ref=0), then channels 2 and 3 will be registered together
    (ref=2), and finally the two groups will be registered together (ref=0).

    Args:
        channel_grouping (list): List of list of channels to register together.
        ops (dict): Experiment metadata.
        ops_prefix (str): Prefix to use for ops, e.g. "genes".
        stack (np.array): Image stack to register.
        reference_prefix (str): Prefix to load scale or initial matrix from.
        binarise_quantile (float): Quantile to binarise images before registration.
        reference_tforms (dict): Reference transformation parameters.
        debug (bool): Return debug information.

    Returns:
        dict: Transformation parameters.
        dict: Debug information, only if debug is True
    """
    assert stack.ndim == 3, "Stack must be XxYxNch"
    nch = stack.shape[2]
    initial_tforms = [[] for i in range(nch)]
    if debug:
        db_info = {"first_round": {}}
    for group in channel_grouping:
        assert all(isinstance(ch, int) for ch in group), "Only integers are allowed"
        tform = _reg_chans(
            ops,
            ops_prefix,
            stack[..., group],
            reference_prefix,
            binarise_quantile,
            reference_tforms,
            ref_ch=0,  # always 0 as channels are reordered
            debug=False,
        )
        if debug:
            db_info["first_round"][tuple(group)] = tform
        if "matrix_between_channels" in tform.keys():
            tform_matrix = tform["matrix_between_channels"]
        else:
            angles = (tform["angles_between_channels"],)
            shifts = (tform["shifts_between_channels"],)
            scales = (tform["scales_between_channels"],)
            tform_matrix = []
            for sc, sh, an in zip(scales, shifts, angles):
                tform_matrix.append(
                    make_transform(s=sc, angle=an, shift=sh, shape=stack.shape[:2])
                )
        for i, ch in enumerate(group):
            initial_tforms[ch] = tform_matrix[i]
    # now we need to merge the tforms
    second_round = [g[0] for g in channel_grouping]
    tform = _reg_chans(
        ops,
        ops_prefix,
        stack[..., second_round],
        reference_prefix,
        binarise_quantile,
        reference_tforms,
        ref_ch=0,
        debug=False,
    )
    if debug:
        db_info["second_round"] = tform
    if "matrix_between_channels" in tform.keys():
        tform_matrix = tform["matrix_between_channels"]
    else:
        angles = (tform["angles_between_channels"],)
        shifts = (tform["shifts_between_channels"],)
        scales = (tform["scales_between_channels"],)
        tform_matrix = []
        for sc, sh, an in zip(scales, shifts, angles):
            tform_matrix.append(
                make_transform(s=sc, angle=an, shift=sh, shape=stack.shape[:2])
            )

    # multiply the tforms to get the final one
    output = initial_tforms.copy()
    for igpg, gpg in enumerate(channel_grouping):
        if not igpg:
            # reference group
            continue
        for ch in gpg:
            # the first round must run successfully but the second one can fail,
            # we should still be mostly fine
            if np.any(np.isnan(tform_matrix[igpg])):
                warn(f"Failed to register group {gpg} to reference")
                tform_matrix[igpg] = np.eye(3)
            output[ch] = output[ch] @ tform_matrix[igpg]
    # convert back to expected output format
    if "matrix_between_channels" in tform.keys():
        output = dict(matrix_between_channels=output)
    else:
        angles, shifts, scales = [], [], []
        for ch, tf in enumerate(output):
            sim = SimilarityTransform(matrix=tf)
            angles.append(sim.rotation)
            shifts.append(sim.translation)
            scales.append(sim.scale)
        output = dict(
            angles_between_channels=angles,
            shifts_between_channels=shifts,
            scales_between_channels=scales,
        )
    if debug:
        return output, db_info
    return output


def _reg_chans(
    ops,
    ops_prefix,
    stack,
    reference_prefix,
    binarise_quantile,
    reference_tforms,
    ref_ch,
    debug,
):
    """Register channels for a single tile

    Inner function running the relevant phase correlation on channels of one stack

    Args:
        ops (dict): Experiment metadata.
        ops_prefix (str): Prefix to use for ops, e.g. "genes".
        stack (np.array): Image stack to register.
        reference_prefix (str): Prefix to load scale or initial matrix from.
        binarise_quantile (float): Quantile to binarise images before registration.
        reference_tforms (dict): Reference transformation parameters.
        ref_ch (int): Reference channel.
        debug (bool): Return debug information.

    Returns:
        dict: Transformation parameters.
        dict: Debug information, only if debug is True
    """

    if ops["align_method"] == "similarity":
        if reference_prefix is None:
            raise ValueError(
                "Reference prefix must be provided for similarity transform"
            )
        # binarise if needed
        nch = stack.shape[2]
        if binarise_quantile is not None:
            for ich in range(nch):
                ref_thresh = np.quantile(stack[:, :, ich], binarise_quantile)
                stack[:, :, ich] = stack[:, :, ich] > ref_thresh

        out = estimate_shifts_and_angles_for_tile(
            stack,
            scales=reference_tforms["scales_between_channels"],
            ref_ch=ref_ch,
            max_shift=ops["rounds_max_shift"],
            debug=debug,
        )
        if debug:
            angles, shifts, db_info = out
        else:
            angles, shifts = out
        to_save = dict(
            angles=angles,
            shifts=shifts,
            scales=reference_tforms["scales_between_channels"],
        )
    elif ops["align_method"] == "affine":
        block_size = ops.get(f"{ops_prefix}_reg_block_size", 256)
        overlap = ops.get(f"{ops_prefix}_reg_block_overlap", 0.5)
        correlation_threshold = ops.get(f"{ops_prefix}_correlation_threshold", None)
        max_residual = ops.get(f"{ops_prefix}_max_residual", 2)
        print("Registration parameters:")
        print(f"    block size {block_size}\n    overlap {overlap}")
        print(f"    correlation threshold {correlation_threshold}")
        print(f"    binarise quantile {binarise_quantile}")
        print(f"    max residual {max_residual}")
        print(f"    ref channel {ref_ch}")
        print(f"    max shift {ops['rounds_max_shift']}")
        if reference_prefix is None:
            tform_matrix = None
        else:
            tform_matrix = reference_tforms["matrix_between_channels"]
        matrix = estimate_affine_for_tile(
            stack,
            tform_matrix=tform_matrix,
            ref_ch=ref_ch,
            max_shift=ops["rounds_max_shift"],
            max_residual=max_residual,
            debug=debug,
            block_size=block_size,
            overlap=overlap,
            correlation_threshold=correlation_threshold,
            binarise_quantile=binarise_quantile,
        )
        if debug:
            matrix, db_info = matrix
        to_save = dict(matrix_between_channels=matrix)
    else:
        raise ValueError(f"Align method {ops['align_method']} not recognised")
    if debug:
        return to_save, db_info
    return to_save


def estimate_shifts_by_coors(
    data_path, tile_coors=(0, 0, 0), prefix="genes_round", suffix="max"
):
    """Estimate shifts across channels and sequencing rounds using provided reference
    rotation angles and scale factors.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to register, in (ROI, X, Y)
            format. Defaults to (0, 0, 0).
        prefix (str, optional): Directory prefix to register. Defaults to "genes_round".
        suffix (str, optional): Filename suffix specifying which z-projection to use.
            Defaults to "fstack".

    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)

    median_filter_size = ops["reg_median_filter"]
    nrounds = ops[prefix + "s"]
    tforms_path = processed_path / "reg" / prefix / f"ref_tile_tforms_{prefix}.npz"
    stack = load_sequencing_rounds(
        data_path, tile_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )
    reference_tforms = np.load(tforms_path, allow_pickle=True)
    (_, shifts_within_channels, matrix_between_channels) = estimate_shifts_for_tile(
        stack,
        reference_tforms["angles_within_channels"],
        reference_tforms["matrix_between_channels"],
        ref_ch=ops["ref_ch"],
        ref_round=ops["ref_round"],
        max_shift=ops["rounds_max_shift"],
        min_shift=ops["rounds_min_shift"],
        median_filter_size=median_filter_size,
    )
    save_dir = processed_path / "reg" / prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    target = f"tforms_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
    np.savez(
        save_dir / target,
        angles_within_channels=reference_tforms["angles_within_channels"],
        shifts_within_channels=shifts_within_channels,
        matrix_between_channels=matrix_between_channels,
        allow_pickle=True,
    )
    print(f"Saved tforms to {save_dir / target}")


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="64G"))
def run_correct_shifts(data_path, prefix):
    """Use robust regression to correct shifts across tiles within an ROI
    for all ROIs.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "genes_round".

    """
    roi_dims = get_roi_dimensions(data_path)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi_dim in roi_dims[use_rois, :]:
        correct_shifts_roi(
            data_path,
            roi_dim,
            prefix=prefix,
            max_shift=ops["ransac_max_shift"],
            min_tiles=ops["ransac_min_tiles"],
        )
        filter_ransac_shifts(
            data_path, prefix, roi_dim, max_residuals=ops["ransac_residual_threshold"]
        )
    check_tile_shifts(data_path, prefix)


def correct_shifts_roi(
    data_path, roi_dims, prefix="genes_round", max_shift=500, min_tiles=0
):
    """Use robust regression to correct shifts across tiles for a single ROI.

    RANSAC regression is applied to shifts within and across channels using
    tile X and Y position as predictors.

    Args:
        data_path (str): Relative path to data.
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles,
            Ytiles) format.
        prefix (str, optional): Directory prefix to use. Defaults to "genes_round".
        max_shift (int, optional): Maximum shift to include tiles in RANSAC regression.
            Tiles with larger absolute shifts will not be included in the fit but will
            still have their corrected shifts estimated. Defaults to 500.
        min_tiles (int, optional): Minimum number of tiles to use for RANSAC regression,
            otherwise median is used.

    """
    processed_path = get_processed_path(data_path)
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1

    shifts_within_channels = []
    shifts_between_channels = []
    for iy in range(ny):
        for ix in range(nx):
            tforms = np.load(
                processed_path / "reg" / prefix / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
            )
            shifts_within_channels.append(tforms["shifts_within_channels"])
            matrix_between_channels = tforms["matrix_between_channels"]
            shifts = [m[:2, 2] for m in matrix_between_channels]
            shifts_between_channels.append(shifts)
    shifts_within_channels = np.stack(shifts_within_channels, axis=3)
    shifts_between_channels = np.stack(shifts_between_channels, axis=2)

    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_within_channels_corrected = np.zeros(shifts_within_channels.shape)
    shifts_between_channels_corrected = np.zeros(shifts_between_channels.shape)
    # TODO: maybe make `training` in the loop above?
    training = np.stack([ys.flatten(), xs.flatten(), np.ones(nx * ny)], axis=1)
    ntiles = nx * ny
    if ntiles < min_tiles:
        shifts_within_channels_corrected = np.tile(
            np.median(shifts_within_channels, axis=3)[:, :, :, np.newaxis],
            (1, 1, 1, ntiles),
        )
        shifts_between_channels_corrected = np.tile(
            np.median(shifts_between_channels, axis=2)[:, :, np.newaxis], (1, 1, ntiles)
        )
    else:
        for ich in range(shifts_within_channels.shape[0]):
            for iround in range(shifts_within_channels.shape[1]):
                median_shift = np.median(
                    shifts_within_channels[ich, iround, :, :], axis=1
                )[:, np.newaxis]
                inliers = np.all(
                    np.abs(shifts_within_channels[ich, iround, :, :] - median_shift)
                    < max_shift,
                    axis=0,
                )
                for idim in range(2):
                    reg = RANSACRegressor(random_state=0).fit(
                        training[inliers, :],
                        shifts_within_channels[ich, iround, idim, inliers],
                    )
                    shifts_within_channels_corrected[ich, iround, idim, :] = (
                        reg.predict(training)
                    )
            median_shift = np.median(shifts_between_channels[ich, :, :], axis=1)[
                :, np.newaxis
            ]
            inliers = np.all(
                np.abs(shifts_between_channels[ich, :, :] - median_shift) < max_shift,
                axis=0,
            )
            for idim in range(2):
                reg = RANSACRegressor(random_state=0).fit(
                    training[inliers, :], shifts_between_channels[ich, idim, inliers]
                )
                shifts_between_channels_corrected[ich, idim, :] = reg.predict(training)

    save_dir = processed_path / "reg" / prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    itile = 0
    matrix = matrix_between_channels.copy()
    for iy in range(ny):
        for ix in range(nx):
            matrix[:, :2, 2] = shifts_between_channels_corrected[:, :, itile]
            target = f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
            np.savez(
                save_dir / target,
                angles_within_channels=tforms["angles_within_channels"],
                shifts_within_channels=shifts_within_channels_corrected[:, :, :, itile],
                matrix_between_channels=matrix,
                allow_pickle=True,
            )
            print(f"Saved tforms to {save_dir / target}")
            # TODO: perhaps save in a cleaner way
            itile += 1


def filter_ransac_shifts(data_path, prefix, roi_dims, max_residuals=10):
    """Filter shifts to use RANSAC shifts only if the initial shifts are off

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "genes_round".
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles,
            Ytiles)
        max_residuals (int, optional): Threshold on residuals above which the RANSAC
            shifts are used. Defaults to 10.
    """
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    for iy in range(ny):
        for ix in range(nx):
            tforms_init = get_channel_round_transforms(
                data_path, prefix, (roi, ix, iy), shifts_type="reference"
            )
            tforms_corrected = get_channel_round_transforms(
                data_path, prefix, (roi, ix, iy), shifts_type="corrected"
            )
            tforms_best = {key: tforms_init[key] for key in tforms_init.keys()}

            # first for within, it's easy shifts are saved separately
            if "shifts_within_channels" not in tforms_init.keys():
                # it must be a non-round acquisition, let's skip
                pass
            else:
                shifts_init = tforms_init["shifts_within_channels"]
                shifts_corrected = tforms_corrected["shifts_within_channels"]
                residuals = np.abs(shifts_init - shifts_corrected)
                shifts_best = np.array(shifts_init, copy=True)
                shifts_best[residuals > max_residuals] = shifts_corrected[
                    residuals > max_residuals
                ]
                tforms_best["shifts_within_channels"] = shifts_best

            # then for between, we need to update the matrix
            matrix_init = tforms_init["matrix_between_channels"]
            # replace the NaN with inf to replace them
            matrix_init[np.isnan(matrix_init)] = np.inf
            matrix_corrected = tforms_corrected["matrix_between_channels"]
            residuals = np.abs(matrix_init[:, :2, 2] - matrix_corrected[:, :2, 2])
            matrix_best = matrix_init.copy()
            good = np.all(residuals < max_residuals, axis=1)
            matrix_best[~good] = matrix_corrected[~good]
            tforms_best["matrix_between_channels"] = matrix_best

            tforms_best.update({"allow_pickle": True})
            target = get_channel_round_transforms(
                data_path, prefix, (roi, ix, iy), shifts_type="best", load_file=False
            )
            np.savez(target, **tforms_best)


@slurm_it(conda_env="iss-preprocess")
def correct_hyb_shifts(data_path, prefix=None):
    """Use robust regression across tiles to correct shifts and angles
    for hybridisation rounds. Either processes a specific hybridisation
    round or all rounds.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "hybridisation_1_1". If None,
            processes all hybridisation acquisitions.
    """
    metadata = load_metadata(data_path)
    if prefix:
        if isinstance(prefix, str):
            prefix = [prefix]
    else:
        prefix = metadata["hybridisation"].keys()
    for hyb_round in prefix:
        roi_dims = get_roi_dimensions(data_path, hyb_round)
        ops = load_ops(data_path)
        if "use_rois" not in ops.keys():
            ops["use_rois"] = roi_dims[:, 0]
        use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])

        use_median = ops.get(
            f"{hyb_round.split('_')[0]}_use_median_channel_registration", False
        )
        use_median = ops.get(f"{hyb_round}_use_median_channel_registration", use_median)
        if use_median:
            print("Using median channel registration for all ROIs")
            merge_shifts(data_path, prefix=hyb_round, n_chans=4)
        else:
            for roi in roi_dims[use_rois, :]:
                print(f"correcting shifts for ROI {roi}, {hyb_round} from {data_path}")
                try:
                    correct_shifts_single_round_roi(
                        data_path, roi, prefix=hyb_round, fit_angle=False, n_chans=4
                    )
                except ValueError:
                    txt = f"!!! Could not correct shifts for ROI {roi}, {hyb_round}"
                    # We both warn and print to have the message in out and err
                    warn(txt)
                    print(txt)
                    continue

        for roi in roi_dims[use_rois, :]:
            filter_ransac_shifts(
                data_path,
                prefix=hyb_round,
                roi_dims=roi,
                max_residuals=ops["ransac_residual_threshold"],
            )


@slurm_it(conda_env="iss-preprocess")
def correct_shifts_to_ref(data_path, prefix, max_shift=None, fit_angle=False):
    """Use robust regression across tiles to correct shifts to reference acquisition

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "genes_round".
        fit_angle (bool, optional): Fit the angle with robust regression if True,
            otherwise takes the median. Defaults to False

    """
    roi_dims = get_roi_dimensions(data_path)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    if max_shift is None:
        max_shift = ops["max_shift2ref"]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    prefix_to_reg = f"to_ref_{prefix}"
    for roi_dim in roi_dims[use_rois, :]:
        print(f"correcting shifts for ROI {roi_dim}, {prefix_to_reg} from {data_path}")
        try:
            correct_shifts_single_round_roi(
                data_path,
                roi_dim,
                prefix=prefix_to_reg,
                fit_angle=fit_angle,
                max_shift=max_shift,
                n_chans=1,  # we register one channel since they are all aligned already
            )
        except ValueError:
            txt = f"!!! Could not correct shifts for ROI {roi_dim[0]}, {prefix_to_reg}"
            # We both warn and print to make sure the message is seen in out and err
            warn(txt)
            print(txt)
            continue
        filter_ransac_shifts(data_path, prefix_to_reg, roi_dim, max_residuals=10)


def merge_shifts(data_path, prefix, n_chans=4):
    """Merge shifts for all ROI/tiles into a single shift median shift

    Useful if some of the registration failed and we want to use the same shift for all
    tiles

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "hybridisation_1_1".
        n_chans (int, optional): Number of channels to merge. Defaults to 4.

    """
    ops = load_ops(data_path)
    roi_dims = get_roi_dimensions(data_path, prefix)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    align_method = ops["align_method"]

    shifts = []
    angles = []
    for roi, nx, ny in roi_dims[use_rois, ...]:
        nx += 1
        ny += 1
        shift, angle, scales = _load_shift_roi(
            data_path, prefix, roi, nx, ny, align_method, n_chans
        )
        shifts.append(shift)
        angles.append(angle)

    shifts = np.concatenate(shifts, axis=2)
    angles = np.concatenate(angles, axis=1)
    bad = np.sum(np.any(np.isnan(shifts), axis=(0, 1)))
    if bad > 0:
        print(f"{bad}/{shifts.shape[2]} tiles have failed fits")
    shifts_corrected = np.nanmedian(shifts, axis=2)
    angles_corrected = np.nanmedian(angles, axis=1)
    print(f"Median shifts: \n{np.round(shifts_corrected,2)}")
    print(f"Median angles/affine: \n{np.round(angles_corrected, 3)}")

    if align_method == "affine":
        matrix_corrected = np.zeros((shifts.shape[0], 3, 3))
        for ich in range(shifts.shape[0]):
            matrix_corrected[ich, :2, 2] = shifts_corrected[ich]
            matrix_corrected[ich, :2, :2] = angles_corrected[ich]
            matrix_corrected[ich, 2, 2] = 1

    else:
        # i need to check the shape of that if we ever need to use it
        raise NotImplementedError("Merging shifts for non-affine not implemented yet")

    # save all the corrected to the same median value
    processed_path = get_processed_path(data_path)
    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    for roi, nx, ny in roi_dims[use_rois, ...]:
        nx += 1
        ny += 1
        itile = 0
        for iy in range(ny):
            for ix in range(nx):
                if align_method == "affine":
                    to_save = dict(matrix_between_channels=matrix_corrected)
                else:
                    to_save = dict(
                        shifts=shifts_corrected[:, :, itile],
                        angles=angles_corrected[:, itile],
                        scales=scales,
                    )
                np.savez(
                    save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                    allow_pickle=True,
                    **to_save,
                )
                itile += 1
    print("Merged shifts for all tiles")


def _load_shift_roi(data_path, prefix, roi, nx, ny, align_method, n_chans=None):
    processed_path = get_processed_path(data_path)
    shifts = []
    angles = []
    scales = []
    for iy in range(ny):
        for ix in range(nx):
            fname = processed_path / "reg" / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
            if not fname.exists():
                print(f"No tforms for tile {roi} {ix} {iy}")
                shifts.append(np.array([[np.nan, np.nan]]))
                if n_chans is None:
                    raise ValueError("n_chans must be provided if tforms are missing")
                if align_method == "affine":
                    angles.append(np.zeros((n_chans, 2, 2)) + np.nan)
                else:
                    angles.append(np.array(np.nan, ndmin=2))

                continue
            try:
                tforms = np.load(
                    processed_path / "reg" / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                if align_method == "affine":
                    shifts.append(tforms["matrix_between_channels"][:, :2, 2])
                    angles.append(tforms["matrix_between_channels"][:, :2, :2])
                else:
                    scales.append(tforms["scales"])
                    shifts.append(tforms["shifts"])
                    angles.append(tforms["angles"])
            except ValueError:
                print(f"couldn't load tile {roi} {ix} {iy}")
                if n_chans is None:
                    raise ValueError("n_chans must be provided if tforms are missing")
                shifts.append(np.array([[np.nan, np.nan]]))
                if align_method == "affine":
                    angles.append(np.zeros((n_chans, 2, 2)) + np.nan)
                else:
                    angles.append(np.array(np.nan, ndmin=2))

    shifts = np.stack(shifts, axis=2)
    angles = np.stack(angles, axis=1)
    return shifts, angles, scales


def correct_shifts_single_round_roi(
    data_path,
    roi_dims,
    prefix="hybridisation_1_1",
    max_shift=500,
    fit_angle=True,
    align_method=None,
    n_chans=None,
):
    """Use robust regression across tiles to correct shifts and angles
    for a single hybridisation round and ROI.

    Args:
        data_path (str): Relative path to data.
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles,
            Ytiles) format.
        prefix (str, optional): Prefix of the round to be processed.
            Defaults to "hybridisation_1_1".
        max_shift (int, optional): Maximum shift to include tiles in RANSAC regression.
            Tiles with larger absolute shifts will not be included in the fit but will
            still have their corrected shifts estimated. Defaults to 500.
        fit_angle (bool, optional): Fit the angle with robust regression if True,
            otherwise takes the median. Defaults to True
        align_method (str, optional): Method to use for alignment. If None, will be
            read from ops. Defaults to None.

    Returns:
        None
    """

    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    if align_method is None:
        align_method = ops["align_method"]

    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    shifts, angles, scales = _load_shift_roi(
        data_path, prefix, roi, nx, ny, align_method, n_chans
    )
    if align_method != "affine":
        assert len(scales), "scales must be exist for non-affine alignment"
        # they should all be the same
        assert np.all(scales == scales[0]), "scales must be the same for all tiles"
        scales = scales[0]
    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_corrected = np.zeros(shifts.shape)
    angles_corrected = np.zeros(angles.shape)

    training = np.stack([ys.flatten(), xs.flatten(), np.ones(nx * ny)], axis=1)

    for ich in range(shifts.shape[0]):
        for idim in range(2):
            inliers = np.all(np.abs(shifts[ich, :, :]) < max_shift, axis=0)
            reg = RANSACRegressor(random_state=0).fit(
                training[inliers, :], shifts[ich, idim, inliers]
            )
            shifts_corrected[ich, idim, :] = reg.predict(training)
        if fit_angle:
            if ops["align_method"] == "affine":
                raise ValueError("Angle correction not implemented for affine")
            reg = RANSACRegressor(random_state=0).fit(training, angles[ich, :])
            angles_corrected[ich, :] = reg.predict(training)
        else:
            angles_corrected[ich, :] = np.nanmedian(angles[ich, :], axis=0)

    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    itile = 0
    for iy in range(ny):
        for ix in range(nx):
            if align_method == "affine":
                matrix_corrected = np.zeros((shifts.shape[0], 3, 3))
                for ich in range(shifts.shape[0]):
                    matrix_corrected[ich, :2, 2] = shifts_corrected[ich, :, itile]
                    matrix_corrected[ich, :2, :2] = angles_corrected[ich, itile]
                    matrix_corrected[ich, 2, 2] = 1
                to_save = dict(matrix_between_channels=matrix_corrected)
            else:
                to_save = dict(
                    shifts=shifts_corrected[:, :, itile],
                    angles=angles_corrected[:, itile],
                    scales=scales,
                )
            np.savez(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                allow_pickle=True,
                **to_save,
            )
            itile += 1


def load_and_register_tile(
    data_path,
    tile_coors,
    prefix,
    filter_r=True,
    projection=None,
    zero_bad_pixels=False,
    correct_illumination=True,
):
    """Load one single tile

    Load a tile of `prefix` with channels/rounds registered, apply illumination
    correction and filtering.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (Roi, tileX, tileY) tuple
        prefix (str): Acquisition to load. If `genes_round` or `barcode_round` will load
            all the rounds.
        filter_r (bool, optional): Apply filter on rounds data? Parameters will be read
            from `ops`. Default to True
        projection (str, optional): Projection to use. If None, will read from `ops`.
            Defaults to None
        zero_bad_pixels (bool, optional): Set bad pixels to zero. Defaults to False
        correct_illumination (bool, optional): Apply illumination correction. Defaults
            to True

    Returns:
        numpy.ndarray: A (X x Y x Nchannels x Nrounds) registered stack
        numpy.ndarray: X x Y boolean mask of bad pixels where data is missing after
            registration

    """
    ops = load_ops(data_path)
    if projection is None:
        projection = ops[f"{prefix.split('_')[0].lower()}_projection"]
    if filter_r and isinstance(filter_r, bool):
        filter_r = ops["filter_r"]
    if prefix.startswith("genes_round") or prefix.startswith("barcode_round"):
        parts = prefix.split("_")
        if len(parts) > 2:
            acq_type = "_".join(parts[:2])
            rounds = np.array([int(parts[2])])
        else:
            acq_type = prefix
            rounds = np.arange(ops[f"{acq_type}s"]) + 1

        stack, bad_pixels = load_and_register_sequencing_tile(
            data_path,
            tile_coors=tile_coors,
            suffix=projection,
            prefix=acq_type,
            filter_r=filter_r,
            correct_channels=True,
            correct_illumination=correct_illumination,
            corrected_shifts=ops["corrected_shifts"],
            specific_rounds=rounds,
        )
        # the transforms for all rounds are the same and saved with round 1
        prefix = acq_type + "_1_1"
    else:
        stack, bad_pixels = load_and_register_hyb_tile(
            data_path,
            tile_coors=tile_coors,
            prefix=prefix,
            suffix=projection,
            filter_r=filter_r,
            correct_illumination=correct_illumination,
            correct_channels=False,
            corrected_shifts=ops["corrected_shifts"],
        )

    # ensure we have 4d to match acquisitions with rounds
    if stack.ndim == 3:
        stack = stack[..., np.newaxis]

    if zero_bad_pixels:
        stack[bad_pixels] = 0

    return stack, bad_pixels


def load_and_register_raw_stack(data_path, prefix, tile_coors, corrected_shifts=None):
    """Load a raw stack and apply illumination correction and channel registration.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition to load.
        tile_coors (tuple): (Roi, tileX, tileY) tuple
        corrected_shifts (str, optional): Shift correction method. Defaults to None.

    Returns:
        numpy.ndarray: A (X x Y x Nchannels) registered stack

    """

    if corrected_shifts is None:
        ops = load_ops(data_path)
        corrected_shifts = ops["corrected_shifts"]
    valid_shifts = ["reference", "single_tile", "ransac", "best"]
    assert corrected_shifts in valid_shifts, (
        f"unknown shift correction method, must be one of {valid_shifts}",
    )
    tforms = get_channel_shifts(data_path, prefix, tile_coors, corrected_shifts)
    fname = get_raw_filename(data_path, prefix, tile_coors)
    tile_path = str(Path(data_path) / prefix / fname)

    fmetadata = get_raw_path(tile_path + "_metadata.txt")
    if fmetadata.exists():
        stack = get_tile_ome(
            get_raw_path(tile_path + ".ome.tif"),
            fmetadata,
        )
    else:
        stack = get_tile_ome(
            get_raw_path(tile_path + ".ome.tif"),
            None,
            use_indexmap=True,
        )
    stack = apply_illumination_correction(data_path, stack, prefix)
    c_stack = np.zeros_like(stack)
    for z in np.arange(stack.shape[-1]):
        c_stack[..., z] = apply_corrections(
            stack[..., z], matrix=tforms["matrix_between_channels"], cval=np.nan
        )

    return c_stack
