import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from skimage.morphology import disk
from sklearn.linear_model import RANSACRegressor
from znamutils import slurm_it

import iss_preprocess as iss

from ..io import get_roi_dimensions, load_metadata, load_ops, load_tile_by_coors
from ..reg import (
    estimate_rotation_translation,
    estimate_affine_for_tile,
    estimate_shifts_and_angles_for_tile,
    estimate_shifts_for_tile,
    make_transform,
    register_channels_and_rounds,
)
from .sequencing import load_sequencing_rounds


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="64G"))
def register_reference_tile(data_path, prefix="genes_round", diag=False):
    """Estimate round and channel registration parameters for
    the specified tile, include shifts and rotations between rounds
    and shifts, rotations, and scaling between channels.

    Shifts are estimated using phase correlation. Rotation and
    scaling are estimated using iterative grid search.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Directory prefix to register.
            Defaults to "genes_round".
        diag (bool, optional): Whether to save diagnostic plots.

    """
    if diag:
        diag_plot_dir = iss.io.get_processed_path(data_path) / "figures" / "ref_tile"
        diag_plot_dir.mkdir(parents=True, exist_ok=True)
    ops = load_ops(data_path)
    nrounds = ops[prefix + "s"]
    projection = ops[f"{prefix.split('_')[0].lower()}_projection"]
    stack = load_sequencing_rounds(
        data_path, ops["ref_tile"], prefix=prefix, suffix=projection, nrounds=nrounds
    )

    out = register_channels_and_rounds(
        stack,
        ref_ch=ops["ref_ch"],
        ref_round=ops["ref_round"],
        median_filter=ops["reg_median_filter"],
        max_shift=ops["rounds_max_shift"],
        min_shift=ops["rounds_min_shift"],
        debug=diag,
        use_masked_correlation=ops["use_masked_correlation"],
        affine_by_block=ops["align_method"],
    )
    if diag:
        (
            angles_within_channels,
            shifts_within_channels,
            matrix_between_channels,
            debug_dict,
        ) = out
        iss.vis.diagnostics.plot_registration_correlograms(
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

    save_path = iss.io.get_processed_path(data_path) / f"tforms_{prefix}.npz"
    np.savez(
        save_path,
        angles_within_channels=angles_within_channels,
        shifts_within_channels=shifts_within_channels,
        matrix_between_channels=matrix_between_channels,
        allow_pickle=True,
    )


def register_fluorescent_tile(
    data_path,
    tile_coors,
    prefix,
    reference_prefix=None,
    debug=False,
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

    Returns:
        dict: Debug information if debug is True, None otherwise.
    """

    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    projection = ops[f"{prefix.split('_')[0].lower()}_projection"]
    if reference_prefix is not None:
        tforms_path = processed_path / f"tforms_{reference_prefix}.npz"
        reference_tforms = np.load(tforms_path, allow_pickle=True)

    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=projection, prefix=prefix
    )

    # median filter if needed
    median_filter_size = ops["reg_median_filter"]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        stack = median_filter(stack, footprint=disk(median_filter_size), axes=(0, 1))

    binarise_quantile = ops[prefix.split("_")[0].lower() + "_binarise_quantile"]
    match ops["align_method"]:
        case "similarity":
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
                ref_ch=ops["ref_ch"],
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
        case "affine":
            ops_prefix = prefix.split("_")[0].lower()
            block_size = ops.get(f"{ops_prefix}_reg_block_size", 256)
            overlap = ops.get(f"{ops_prefix}_reg_block_overlap", 0.5)
            correlation_threshold = ops.get(f"{ops_prefix}_correlation_threshold", None)
            print("Registration parameters:")
            print(f"    block size {block_size}\n    overlap {overlap}")
            print(f"    correlation threshold {correlation_threshold}")
            print(f"    binarise quantile {binarise_quantile}")
            if reference_prefix is None:
                tform_matrix = None
            else:
                tform_matrix = reference_tforms["matrix_between_channels"]
            matrix = estimate_affine_for_tile(
                stack,
                tform_matrix=tform_matrix,
                ref_ch=ops["ref_ch"],
                max_shift=ops["rounds_max_shift"],
                debug=debug,
                block_size=block_size,
                overlap=overlap,
                correlation_threshold=correlation_threshold,
                binarise_quantile=binarise_quantile,
            )
            if debug:
                matrix, db_info = matrix
            to_save = dict(matrix_between_channels=matrix)
        case _:
            raise ValueError(f"Align method {ops['align_method']} not recognised")

    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir
        / f"tforms_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz",
        allow_pickle=True,
        **to_save,
    )

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
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)

    median_filter_size = ops["reg_median_filter"]
    nrounds = ops[prefix + "s"]
    tforms_path = processed_path / f"tforms_{prefix}.npz"
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
    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir
        / f"tforms_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz",
        angles_within_channels=reference_tforms["angles_within_channels"],
        shifts_within_channels=shifts_within_channels,
        matrix_between_channels=matrix_between_channels,
        allow_pickle=True,
    )


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="64G"))
def correct_shifts(data_path, prefix):
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
    iss.pipeline.check_tile_shifts(data_path, prefix)


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
    processed_path = iss.io.get_processed_path(data_path)
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1

    shifts_within_channels = []
    shifts_between_channels = []
    for iy in range(ny):
        for ix in range(nx):
            tforms = np.load(
                processed_path / "reg" / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
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
    # TODO: maybe make `trianing` in the loop above?
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
                    shifts_within_channels_corrected[
                        ich, iround, idim, :
                    ] = reg.predict(training)
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

    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    itile = 0
    matrix = matrix_between_channels.copy()
    for iy in range(ny):
        for ix in range(nx):
            matrix[:, :2, 2] = shifts_between_channels_corrected[:, :, itile]
            np.savez(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                angles_within_channels=tforms["angles_within_channels"],
                shifts_within_channels=shifts_within_channels_corrected[:, :, :, itile],
                matrix_between_channels=matrix,
                allow_pickle=True,
            )
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
    save_dir = iss.io.get_processed_path(data_path) / "reg"
    for iy in range(ny):
        for ix in range(nx):
            tforms_init = np.load(save_dir / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz")
            tforms_corrected = np.load(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
            )
            tforms_best = {key: tforms_init[key] for key in tforms_init.keys()}

            # first for within, it's easy shifts are saved separatly
            if "shifts_within_channels" not in tforms_init.keys():
                # must be a non-round acquisition
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
            matrix_corrected = tforms_corrected["matrix_between_channels"]
            residuals = np.abs(matrix_init[:, :2, 2] - matrix_corrected[:, :2, 2])
            matrix_best = matrix_init.copy()
            bad = residuals > max_residuals
            matrix_best[:, :2, 2][bad] = matrix_corrected[:, :2, 2][bad]

            tforms_best.update({"allow_pickle": True})
            np.savez(
                save_dir / f"tforms_best_{prefix}_{roi}_{ix}_{iy}.npz", **tforms_best
            )


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
    roi_dims = get_roi_dimensions(data_path)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    metadata = load_metadata(data_path)
    if prefix:
        if isinstance(prefix, str):
            prefix = [prefix]
    else:
        prefix = metadata["hybridisation"].keys()
    for hyb_round in prefix:
        for roi in roi_dims[use_rois, :]:
            print(f"correcting shifts for ROI {roi}, {hyb_round} from {data_path}")
            correct_shifts_single_round_roi(
                data_path, roi, prefix=hyb_round, fit_angle=False, n_chans=4
            )
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
        correct_shifts_single_round_roi(
            data_path,
            roi_dim,
            prefix=prefix_to_reg,
            fit_angle=fit_angle,
            max_shift=max_shift,
            n_chans=1,  # we register one channel since they are all aligned already
        )
        filter_ransac_shifts(data_path, prefix_to_reg, roi_dim, max_residuals=10)


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

    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    if align_method is None:
        align_method = ops["align_method"]

    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    shifts = []
    angles = []
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
                    scales=tforms["scales"],
                )
            np.savez(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                allow_pickle=True,
                **to_save,
            )
            itile += 1


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
    job_ids = iss.pipeline.batch_process_tiles(
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
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinats
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
    # if None, read ref_prefix, ref_channels, binarise_quantile and reg_channels from ops
    if (ref_prefix is None) or (ref_prefix == "None"):
        ref_prefix = ops["reference_prefix"]
    if ref_prefix == reg_prefix:
        raise ValueError("Reference and register prefixes are the same")
    if ref_channels is None:
        ref_channels = ops["reg2ref_reference_channels"]
    spref = reg_prefix.split("_")[0].lower()  # short prefix
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
    ref_all_channels, ref_bad_pixels = iss.pipeline.load_and_register_tile(
        data_path=data_path,
        tile_coors=ref_tile_coors,
        prefix=ref_prefix,
        filter_r=False,
        zero_bad_pixels=False,
    )
    reg_all_channels, reg_bad_pixels = iss.pipeline.load_and_register_tile(
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
    processed_path = iss.io.get_processed_path(data_path)
    r, x, y = tile_coors
    target = processed_path / "reg" / f"tforms_to_ref_{reg_prefix}_{r}_{x}_{y}.npz"
    # reshape tforms to be like the multichannels tforms
    np.savez(target, matrix_between_channels=tforms.reshape((1, 3, 3)))
    return tforms


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def register_to_ref_using_stitched_registration(
    data_path,
    reg_prefix,
    ref_prefix=None,
    rois=None,
    ref_channels=None,
    reg_channels=None,
    estimate_rotation=True,
    target_suffix=None,
    use_masked_correlation=False,
    downsample=3,
):
    """Register all tiles to the reference using the stitched registration

    This will stitch both the reference and target tiles using the reference shifts,
    then register the stitched target to the stitched reference to get the best
    similarity transform.
    Then the transformation is applied to each tile and saved instead of the one
    generated by "register_tile_to_ref".

    Args:
        data_path (str): Relative path to data
        target_prefix (str): Prefix of the target tile
        reference_prefix (str, optional): Prefix of the reference tile. If None, reads
            from ops. Defaults to None.
        roi (list, optional): ROI IDs. If None, process all rois. Defaults to None.
        ref_ch (int, optional): Reference channel. If None, reads from ops. Defaults to
            None.
        target_ch (int, optional): Target channel. If None, reads from ops. Defaults to
            None.
        estimate_rotation (bool, optional): Estimate rotation. Defaults to True.
        target_suffix (str, optional): Suffix of the target tile. Defaults to None.
        use_masked_correlation (bool, optional): Use masked correlation. Defaults to False.
        downsample (int, optional): Downsample factor. Defaults to 3.

    Returns:
        None

    """
    # format arguments
    if rois is None:
        roi_dims = get_roi_dimensions(data_path)
        rois = roi_dims[:, 0]
    elif isinstance(rois, int):
        rois = [rois]
    ops = load_ops(data_path)
    if (ref_prefix is None) or (ref_prefix == "None"):
        ref_prefix = ops["reference_prefix"]
    if ref_prefix == reg_prefix:
        raise ValueError("Reference and register prefixes are the same")
    if ref_channels is None:
        ref_channels = ops["reg2ref_reference_channels"]
    spref = reg_prefix.split("_")[0].lower()  # short prefix
    if reg_channels is None:
        # use either the same as ref or what is in the ops
        reg_channels = ops.get(f"reg2ref_{spref}_channels", ref_channels)
        # if there is something defined for this acquisition, use it instead
        reg_channels = ops.get(f"reg2ref_{reg_prefix}_channels", reg_channels)

    for roi in rois:
        # get the transformation from the stitched image to the reference
        print(f"Registering {reg_prefix} to {ref_prefix} for ROI {roi}")
        estimate_scale = False  # never estimate scale
        (
            stitched_stack_target,
            _,
            angle,
            shift,
            scale,
        ) = iss.pipeline.stitch.stitch_and_register(
            data_path,
            ref_prefix,
            reg_prefix,
            roi,
            downsample,
            ref_channels,
            reg_channels,
            estimate_scale,
            estimate_rotation,
            target_suffix,
            use_masked_correlation,
            debug=False,
        )

        # transform the center of each tile
        tform2ref = make_transform(
            scale,
            angle,
            shift,
            stitched_stack_target.shape[:2],
        )
        ref_corners = iss.pipeline.stitch.get_tile_corners(data_path, ref_prefix, roi)
        tile_shape = ref_corners[0, 0, :, 2] - ref_corners[0, 0, :, 0]
        ref_centers = np.mean(ref_corners, axis=3)
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
                processed_path = iss.io.get_processed_path(data_path)

                target = (
                    processed_path
                    / "reg"
                    / f"tforms_to_ref_{reg_prefix}_{roi}_{tilex}_{tiley}.npz"
                )
                # reshape tforms to be like the multichannels tforms
                np.savez(target, matrix_between_channels=tforms.reshape((1, 3, 3)))
    print("Done")


def align_spots(data_path, tile_coors, prefix, ref_prefix=None):
    """Use previously computed transformation matrices to align spots to reference
    coordinates.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinates
        prefix (str): Prefix of spots to load
        ref_prefix (str, optional): Prefix of the reference spots. If None, reads from
            ops. Defaults to None.

    Returns:
        pd.DataFrame: The spot dataframe with x and y registered to reference tile.

    """

    if ref_prefix is None:
        ops = load_ops(data_path)
        ref_prefix = ops["reference_prefix"]

    roi, tilex, tiley = tile_coors
    processed_path = iss.io.get_processed_path(data_path)
    spots = pd.read_pickle(
        processed_path / "spots" / f"{prefix}_spots_{roi}_{tilex}_{tiley}.pkl"
    )
    spots["tile"] = f"{roi}_{tilex}_{tiley}"
    if ref_prefix.startswith(prefix):
        # it is the ref, no need to register
        return spots

    tform2ref = get_shifts_to_ref(data_path, prefix, roi, tilex, tiley)

    if ops["align_method"] == "similarity":
        # always get tile shape for ref_prefix
        tile_shape = np.load(processed_path / "reg" / f"{ref_prefix}_shifts.npz")[
            "tile_shape"
        ]
        spots_tform = make_transform(
            tform2ref["scales"][0][0],
            tform2ref["angles"][0][0],
            tform2ref["shifts"][0],
            tile_shape,
        )
    else:
        spots_tform = tform2ref["matrix_between_channels"][0]
    transformed_coors = spots_tform @ np.stack(
        [spots["x"], spots["y"], np.ones(len(spots))]
    )
    spots["x"] = [x for x in transformed_coors[0, :]]
    spots["y"] = [y for y in transformed_coors[1, :]]
    return spots


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
    match ops["corrected_shifts"]:
        case "single_tile":
            corrected_shifts = ""
        case "ransac":
            corrected_shifts = "_corrected"
        case "best":
            corrected_shifts = "_best"
        case _:
            raise ValueError(
                f"Corrected shifts {ops['corrected_shifts']} not recognised"
            )
    processed_path = iss.io.get_processed_path(data_path)
    tform2ref = np.load(
        processed_path
        / "reg"
        / f"tforms{corrected_shifts}_to_ref_{prefix}_{roi}_{tilex}_{tiley}.npz"
    )
    return tform2ref
