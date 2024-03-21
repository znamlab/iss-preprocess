import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from skimage.morphology import disk
from scipy.ndimage import median_filter
from znamutils import slurm_it
import iss_preprocess as iss
from ..reg import (
    register_channels_and_rounds,
    estimate_shifts_for_tile,
    estimate_shifts_and_angles_for_tile,
    estimate_rotation_translation,
    make_transform,
)
from .sequencing import load_sequencing_rounds
from ..io import load_tile_by_coors, load_metadata, load_ops, get_roi_dimensions


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
    )
    if diag:
        (
            angles_within_channels,
            shifts_within_channels,
            scales_between_channels,
            angles_between_channels,
            shifts_between_channels,
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
            scales_between_channels,
            angles_between_channels,
            shifts_between_channels,
        ) = out

    save_path = iss.io.get_processed_path(data_path) / f"tforms_{prefix}.npz"
    np.savez(
        save_path,
        angles_within_channels=angles_within_channels,
        shifts_within_channels=shifts_within_channels,
        scales_between_channels=scales_between_channels,
        angles_between_channels=angles_between_channels,
        shifts_between_channels=shifts_between_channels,
        allow_pickle=True,
    )


def estimate_shifts_and_angles_by_coors(
    data_path,
    tile_coors=(0, 0, 0),
    prefix="hybridisation_1_1",
    suffix="max",
    reference_prefix="genes_round",
):
    """Estimate shifts and rotations angles for hybridisation images.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to register, in (ROI, X, Y)
            format. Defaults to (0, 0, 0).
        prefix (str, optional): Prefix of the hybridisation round. Defaults to "hybridisation_1_1".
        reference_prefix (str, optional): Prefix to use for loading precomputed
            scale factors between channels. Defaults to "barcode_round".
        suffix (str, optional): Filename suffix specifying which z-projection to use.
            Defaults to "max".
        reference_prefix (str, optional): Prefix of the reference round. Defaults to "barcode_round".

    """
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    tforms_path = processed_path / f"tforms_{reference_prefix}.npz"
    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
    )
    reference_tforms = np.load(tforms_path, allow_pickle=True)
    threshold_quantile = ops[prefix.split("_")[0].lower() + "_binarise_quantile"]
    angles, shifts = estimate_shifts_and_angles_for_tile(
        stack,
        reference_tforms["scales_between_channels"],
        ref_ch=ops["ref_ch"],
        max_shift=ops["rounds_max_shift"],
        binarise_quantile=threshold_quantile,
    )
    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_dir
        / f"tforms_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz",
        angles=angles,
        shifts=shifts,
        scales=reference_tforms["scales_between_channels"],
        allow_pickle=True,
    )


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
    (_, shifts_within_channels, shifts_between_channels) = estimate_shifts_for_tile(
        stack,
        reference_tforms["angles_within_channels"],
        reference_tforms["scales_between_channels"],
        reference_tforms["angles_between_channels"],
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
        scales_between_channels=reference_tforms["scales_between_channels"],
        angles_between_channels=reference_tforms["angles_between_channels"],
        shifts_between_channels=shifts_between_channels,
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
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles, Ytiles)
            format.
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
            shifts_between_channels.append(tforms["shifts_between_channels"])
    shifts_within_channels = np.stack(shifts_within_channels, axis=3)
    shifts_between_channels = np.stack(shifts_between_channels, axis=2)

    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_within_channels_corrected = np.zeros(shifts_within_channels.shape)
    shifts_between_channels_corrected = np.zeros(shifts_between_channels.shape)
    # TODO: maybe make X in the loop above?
    X = np.stack([ys.flatten(), xs.flatten(), np.ones(nx * ny)], axis=1)
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
                        X[inliers, :],
                        shifts_within_channels[ich, iround, idim, inliers],
                    )
                    shifts_within_channels_corrected[
                        ich, iround, idim, :
                    ] = reg.predict(X)
            median_shift = np.median(shifts_between_channels[ich, :, :], axis=1)[
                :, np.newaxis
            ]
            inliers = np.all(
                np.abs(shifts_between_channels[ich, :, :] - median_shift) < max_shift,
                axis=0,
            )
            for idim in range(2):
                reg = RANSACRegressor(random_state=0).fit(
                    X[inliers, :], shifts_between_channels[ich, idim, inliers]
                )
                shifts_between_channels_corrected[ich, idim, :] = reg.predict(X)

    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    itile = 0
    for iy in range(ny):
        for ix in range(nx):
            np.savez(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                angles_within_channels=tforms["angles_within_channels"],
                shifts_within_channels=shifts_within_channels_corrected[:, :, :, itile],
                scales_between_channels=tforms["scales_between_channels"],
                angles_between_channels=tforms["angles_between_channels"],
                shifts_between_channels=shifts_between_channels_corrected[:, :, itile],
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
            for which in ["within", "between"]:
                shifts_init = tforms_init[f"shifts_{which}_channels"]
                shifts_corrected = tforms_corrected[f"shifts_{which}_channels"]
                residuals = np.abs(shifts_init - shifts_corrected)
                shifts_best = np.array(shifts_init, copy=True)
                shifts_best[residuals > max_residuals] = shifts_corrected[
                    residuals > max_residuals
                ]
                tforms_best[f"shifts_{which}_channels"] = shifts_best
            tforms_best.update({"allow_pickle": True})
            np.savez(
                save_dir / f"tforms_best_{prefix}_{roi}_{ix}_{iy}.npz", **tforms_best
            )


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
        for roi in roi_dims[use_rois, :]:
            print(f"correcting shifts for ROI {roi}, {prefix} from {data_path}")
            correct_shifts_single_round_roi(data_path, roi, prefix=prefix)
    else:
        for hyb_round in metadata["hybridisation"].keys():
            for roi in roi_dims[use_rois, :]:
                print(f"correcting shifts for ROI {roi}, {hyb_round} from {data_path}")
                correct_shifts_single_round_roi(data_path, roi, prefix=hyb_round)


@slurm_it(conda_env="iss-preprocess")
def correct_shifts_to_ref(data_path, prefix, max_shift=500, fit_angle=False):
    """Use robust regression across tiles to correct shifts to reference acquisition

    Args:
        data_path (str): Relative path to data.
        prefix (str): Directory prefix to use, e.g. "genes_round".
        fit_angle (bool, optional): Fit the angle with robust regression if True,
            otherwise takes the median. Defaults to False

    """
    roi_dims = get_roi_dimensions(data_path, prefix)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    prefix_to_reg = f"to_ref_{prefix}"
    for roi_dim in roi_dims[use_rois, :]:
        print(f"correcting shifts for ROI {roi_dim}, {prefix_to_reg} from {data_path}")
        correct_shifts_single_round_roi(
            data_path, roi_dim, prefix=prefix_to_reg, fit_angle=fit_angle, max_shift=max_shift,
        )
        filter_ransac_shifts_to_ref(data_path, prefix, roi_dim, max_residuals=10)


def correct_shifts_single_round_roi(
    data_path, roi_dims, prefix="hybridisation_1_1", max_shift=500, fit_angle=True
):
    """Use robust regression across tiles to correct shifts and angles
    for a single hybridisation round and ROI.

    Args:
        data_path (str): Relative path to data.
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles, Ytiles)
            format.
        prefix (str, optional): Prefix of the round to be processed.
            Defaults to "hybridisation_1_1".
        max_shift (int, optional): Maximum shift to include tiles in RANSAC regression.
            Tiles with larger absolute shifts will not be included in the fit but will
            still have their corrected shifts estimated. Defaults to 500.
        fit_angle (bool, optional): Fit the angle with robust regression if True,
            otherwise takes the median. Defaults to True

    """
    processed_path = iss.io.get_processed_path(data_path)
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    shifts = []
    angles = []
    for iy in range(ny):
        for ix in range(nx):
            fname = processed_path / "reg" / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
            if not fname.exists():
                print(f"Not tforms for tile {roi} {ix} {iy}")
                shifts.append(np.array([[np.nan, np.nan]]))
                angles.append(np.array(np.nan, ndmin=2))
                continue
            try:
                tforms = np.load(
                    processed_path / "reg" / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                shifts.append(tforms["shifts"])
                angles.append(tforms["angles"])
            except ValueError:
                print(f"couldn't load tile {roi} {ix} {iy}")
                shifts.append(np.array([[np.nan, np.nan]]))
                angles.append(np.array(np.nan, ndmin=2))

    shifts = np.stack(shifts, axis=2)
    angles = np.stack(angles, axis=1)

    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_corrected = np.zeros(shifts.shape)
    angles_corrected = np.zeros(angles.shape)

    X = np.stack([ys.flatten(), xs.flatten(), np.ones(nx * ny)], axis=1)

    for ich in range(shifts.shape[0]):
        for idim in range(2):
            inliers = np.all(np.abs(shifts[ich, :, :]) < max_shift, axis=0)
            reg = RANSACRegressor(random_state=0).fit(
                X[inliers, :], shifts[ich, idim, inliers]
            )
            shifts_corrected[ich, idim, :] = reg.predict(X)
        if fit_angle:
            reg = RANSACRegressor(random_state=0).fit(X, angles[ich, :])
            angles_corrected[ich, :] = reg.predict(X)
        else:
            angles_corrected[ich, :] = np.nanmedian(angles[ich, :])

    save_dir = processed_path / "reg"
    save_dir.mkdir(parents=True, exist_ok=True)
    itile = 0
    for iy in range(ny):
        for ix in range(nx):
            np.savez(
                save_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz",
                angles=angles_corrected[:, itile],
                shifts=shifts_corrected[:, :, itile],
                scales=tforms["scales"],
                allow_pickle=True,
            )
            itile += 1


def register_tile_to_ref(
    data_path,
    tile_coors,
    reg_prefix,
    ref_prefix="genes_round",
    binarise_quantile=0.7,
    ref_tile_coors=None,
    reg_channels=None,
    ref_channels=None,
):
    """Register a single tile to the corresponding reference tile

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinats
        reg_prefix (str): Prefix to register, "barcode_round" for instance
        ref_prefix (str, optional): Reference prefix. Defaults to "genes_round".
        binarise_quantile (float, optional): Quantile to binarise images before
            registration. Defaults to 0.7.
        ref_tile_coors (tuple, optional): Tile coordinates of the reference tile.
            Usually not needed as it is assumed to be the same as the tile to register.
            Defaults to None.
        reg_channels (list, optional): Channels to use for registration. If None
            will use all channels. Defaults to None
        ref_channels (list, optional): Channels to use for registration. If None will
            use all channels. Defaults to None

    Returns:
        angle (float): Rotation angle
        shifts (np.array): X and Y shifts

    """
    # Check if we are registering DAPI overview images, if so skip tile to tile registration
    if reg_prefix.startswith("DAPI"):
        print(f"Converting DAPI overview image shift to tile shifts")
        processed_path = iss.io.get_processed_path(data_path)
        r, x, y = tile_coors
        image_shifts = np.load(processed_path / "reg" / f"{reg_prefix}_roi{r}_tform_to_ref.npz")
        shifts = image_shifts["shift"]
        angles = image_shifts["angle"]
        scales = image_shifts["scale"]
        print(f"Angle: {angles}, Shifts: {shifts}")
        save_dir = processed_path / "reg" / f"tforms_to_ref_{reg_prefix}_{r}_{x}_{y}.npz"
        print(f"Saving results to {save_dir}")
        np.savez(
            save_dir,
            angles=np.array([[angles]]),
            shifts=np.array([shifts]),
            scales=np.array([[scales]]),
        )
        return angles, shifts
    
    if ref_tile_coors is None:
        ref_tile_coors = tile_coors
    else:
        print(f"Register to {ref_tile_coors}", flush=True)
    ops = load_ops(data_path)
    ref_all_channels, _ = iss.pipeline.load_and_register_tile(
        data_path=data_path,
        tile_coors=ref_tile_coors,
        prefix=ref_prefix,
        filter_r=False,
    )
    reg_all_channels, _ = iss.pipeline.load_and_register_tile(
        data_path=data_path, tile_coors=tile_coors, prefix=reg_prefix, filter_r=False
    )

    if ref_channels is not None:
        if isinstance(ref_channels, int):
            ref_channels = [ref_channels]
        ref_all_channels = ref_all_channels[:, :, ref_channels]
    ref = np.nanmean(ref_all_channels, axis=(2, 3))

    if reg_channels is not None:
        if isinstance(reg_channels, int):
            reg_channels = [reg_channels]
        reg_all_channels = reg_all_channels[:, :, reg_channels]
    reg = np.nanmean(reg_all_channels, axis=(2, 3))

    if ops["reg_median_filter"]:
        ref = median_filter(ref, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))
        reg = median_filter(reg, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))

    if binarise_quantile is not None:
        reg = reg > np.quantile(reg, binarise_quantile)
        ref = ref > np.quantile(ref, binarise_quantile)

    angles, shifts = estimate_rotation_translation(
        ref,
        reg,
        angle_range=1.0,
        niter=3,
        nangles=15,
        max_shift=ops["rounds_max_shift"],
    )
    print(f"Angle: {angles}, Shifts: {shifts}")
    processed_path = iss.io.get_processed_path(data_path)
    r, x, y = tile_coors
    save_dir = processed_path / "reg" / f"tforms_to_ref_{reg_prefix}_{r}_{x}_{y}.npz"
    print(f"Saving results to {save_dir}")
    # save also scale and make sure that all have the proper shape to match
    # multi-channel registrations and reuse the ransac function
    np.savez(
        save_dir,
        angles=np.array([[angles]]),
        shifts=np.array([shifts]),
        scales=np.array([[1]]),
    )
    return angles, shifts


def filter_ransac_shifts_to_ref(data_path, prefix, roi_dims, max_residuals=10):
    """Filter shifts to use RANSAC shifts only if the initial shifts are off

    Args:
        data_path (str): Relative path to data
        prefix (str): Directory prefix to use, e.g. "genes_round"
        roi_dims (tuple): Dimensions of the ROI to be processed, in (ROI_ID, Xtiles, Ytiles)
        max_residuals (int, optional): Threshold on residuals above which the RANSAC shifts are used. Defaults to 10

    """
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    save_dir = iss.io.get_processed_path(data_path) / "reg"
    for iy in range(ny):
        for ix in range(nx):
            fname = save_dir / f"tforms_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
            if not fname.exists():
                print(f"Skipping {fname}")
                continue
            try:
                tforms_init = np.load(
                    save_dir / f"tforms_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )
            except ValueError:
                print(f"couldn't load {fname}, using corrected")
                tforms_init = np.load(
                    save_dir / f"tforms_corrected_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )
            tforms_corrected = np.load(
                save_dir / f"tforms_corrected_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
            )
            tforms_best = {key: tforms_init[key] for key in tforms_init.keys()}

            shifts_init = tforms_init[f"shifts"]
            shifts_corrected = tforms_corrected[f"shifts"]
            residuals = np.max(np.abs(shifts_init - shifts_corrected))
            shifts_best = np.array(shifts_init, copy=True)
            angles_best = np.array(tforms_corrected[f"angles"], copy=True)
            scales_best = np.array(tforms_corrected[f"scales"], copy=True)
            to_replace = residuals > max_residuals
            shifts_best[to_replace] = shifts_corrected[to_replace]
            angles_best[to_replace] = tforms_corrected[f"angles"][to_replace]
            scales_best[to_replace] = tforms_corrected[f"scales"][to_replace]

            tforms_best[f"shifts"] = shifts_best
            tforms_best[f"angles"] = angles_best
            tforms_best[f"scales"] = scales_best
            tforms_best.update({"allow_pickle": True})

            np.savez(
                save_dir / f"tforms_best_to_ref_{prefix}_{roi}_{ix}_{iy}.npz",
                **tforms_best,
            )


def align_spots(data_path, tile_coors, prefix, ref_prefix="genes_round_1_1"):
    """Use previously computed transformation matrices to align spots to reference
    coordinates.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinates
        prefix (str): Prefix of spots to load
        ref_prefix (str, optional): Prefix of the reference spots. Defaults to
            "genes_round_1_1".

    Returns:
        pd.DataFrame: The spot dataframe with x and y registered to reference tile.

    """
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
