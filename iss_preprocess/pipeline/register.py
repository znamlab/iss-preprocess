import numpy as np
from sklearn.linear_model import RANSACRegressor
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..reg import (
    register_channels_and_rounds,
    estimate_shifts_for_tile,
    estimate_shifts_and_angles_for_tile,
)
from .sequencing import load_sequencing_rounds
from ..io import load_tile_by_coors, load_metadata, load_ops


def register_reference_tile(data_path, prefix="genes_round"):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops_path = processed_path / data_path / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    nrounds = ops[prefix + "s"]
    stack = load_sequencing_rounds(
        data_path,
        ops["ref_tile"],
        prefix=prefix,
        suffix=ops["projection"],
        nrounds=nrounds,
    )
    (
        angles_within_channels,
        shifts_within_channels,
        scales_between_channels,
        angles_between_channels,
        shifts_between_channels,
    ) = register_channels_and_rounds(
        stack, ref_ch=ops["ref_ch"], ref_round=ops["ref_round"]
    )
    save_path = processed_path / data_path / f"tforms_{prefix}.npz"
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
    suffix="fstack",
    reference_prefix="barcode_round",
):
    """Estimate shifts and rotations angles for hybridisation images.

    Args:
        data_path (_type_): _description_
        tile_coors (tuple, optional): _description_. Defaults to (0, 0, 0).
        prefix (str, optional): _description_. Defaults to "hybridisation_1_1".
        reference_prefix (str, optional): _description_. Defaults to "barcode_round".
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops_path = processed_path / data_path / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    tforms_path = processed_path / data_path / f"tforms_{reference_prefix}.npz"
    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
    )
    reference_tforms = np.load(tforms_path, allow_pickle=True)
    angles, shifts = estimate_shifts_and_angles_for_tile(
        stack, reference_tforms["scales_between_channels"], ref_ch=ops["ref_ch"]
    )
    save_dir = processed_path / data_path / "reg"
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
    data_path, tile_coors=(0, 0, 0), prefix="round", suffix="fstack"
):
    """Estimate shifts across channels and sequencing rounds using provided reference
    rotation angles and scale factors.

    Args:
        data_path (_type_): _description_
        tile_coors (tuple, optional): _description_. Defaults to (0, 0, 0).
        prefix (str, optional): _description_. Defaults to "round".
        suffix (str, optional): _description_. Defaults to "fstack".
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops_path = processed_path / data_path / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    nrounds = ops[prefix + "s"]
    tforms_path = processed_path / data_path / f"tforms_{prefix}.npz"
    stack = load_sequencing_rounds(
        data_path, tile_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )
    reference_tforms = np.load(tforms_path, allow_pickle=True)
    (_, shifts_within_channels, shifts_between_channels,) = estimate_shifts_for_tile(
        stack,
        reference_tforms["angles_within_channels"],
        reference_tforms["scales_between_channels"],
        reference_tforms["angles_between_channels"],
        ref_ch=ops["ref_ch"],
        ref_round=0,
    )
    save_dir = processed_path / data_path / "reg"
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


def correct_shifts(data_path, prefix):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ops = load_ops(data_path)
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, :]:
        correct_shifts_roi(data_path, roi, prefix=prefix)


def correct_shifts_roi(data_path, roi_dims, prefix="genes_round", max_shift=500):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1

    shifts_within_channels = []
    shifts_between_channels = []
    for iy in range(ny):
        for ix in range(nx):
            tforms = np.load(
                processed_path
                / data_path
                / "reg"
                / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
            )
            shifts_within_channels.append(tforms["shifts_within_channels"])
            shifts_between_channels.append(tforms["shifts_between_channels"])
    shifts_within_channels = np.stack(shifts_within_channels, axis=3)
    shifts_between_channels = np.stack(shifts_between_channels, axis=2)

    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_within_channels_corrected = np.zeros(shifts_within_channels.shape)
    shifts_between_channels_corrected = np.zeros(shifts_between_channels.shape)
    # TODO: maybe make X in the loop above?
    X = np.stack(
        [
            ys.flatten(),
            xs.flatten(),
            np.ones(
                nx * ny,
            ),
        ],
        axis=1,
    )

    for ich in range(shifts_within_channels.shape[0]):
        for iround in range(shifts_within_channels.shape[1]):
            inliers = np.all(
                np.abs(shifts_within_channels[ich, iround, :, :]) < max_shift, axis=0
            )
            for idim in range(2):
                reg = RANSACRegressor(random_state=0).fit(
                    X[inliers, :], shifts_within_channels[ich, iround, idim, inliers]
                )
                shifts_within_channels_corrected[ich, iround, idim, :] = reg.predict(X)
        inliers = np.all(np.abs(shifts_between_channels[ich, :, :]) < max_shift, axis=0)
        for idim in range(2):
            reg = RANSACRegressor(random_state=0).fit(
                X[inliers, :], shifts_between_channels[ich, idim, inliers]
            )
            shifts_between_channels_corrected[ich, idim, :] = reg.predict(X)

    save_dir = processed_path / data_path / "reg"
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


def correct_hyb_shifts(data_path, prefix=None):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ops = load_ops(data_path)
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    metadata = load_metadata(data_path)
    if prefix:
        for roi in roi_dims[use_rois, :]:
            print(f"correcting shifts for ROI {roi}, {prefix} from {data_path}")
            correct_hyb_shifts_roi(data_path, roi, prefix=prefix)
    else:
        for hyb_round in metadata["hybridisation"].keys():
            for roi in roi_dims[use_rois, :]:
                print(f"correcting shifts for ROI {roi}, {hyb_round} from {data_path}")
                correct_hyb_shifts_roi(data_path, roi, prefix=hyb_round)


def correct_hyb_shifts_roi(
    data_path, roi_dims, prefix="hybridisation_1_1", max_shift=500
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])

    roi = roi_dims[0]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1
    shifts = []
    angles = []
    for iy in range(ny):
        for ix in range(nx):
            try:
                tforms = np.load(
                    processed_path
                    / data_path
                    / "reg"
                    / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz"
                )
            except:
                print(f"couldn't load tile {roi} {ix} {iy}")
            shifts.append(tforms["shifts"])
            angles.append(tforms["angles"])
    shifts = np.stack(shifts, axis=2)
    angles = np.stack(angles, axis=1)

    xs, ys = np.meshgrid(range(nx), range(ny))
    shifts_corrected = np.zeros(shifts.shape)
    angles_corrected = np.zeros(angles.shape)

    X = np.stack(
        [
            ys.flatten(),
            xs.flatten(),
            np.ones(
                nx * ny,
            ),
        ],
        axis=1,
    )

    for ich in range(shifts.shape[0]):
        for idim in range(2):
            inliers = np.all(np.abs(shifts[ich, :, :]) < max_shift, axis=0)
            reg = RANSACRegressor(random_state=0).fit(
                X[inliers, :], shifts[ich, idim, inliers]
            )
            shifts_corrected[ich, idim, :] = reg.predict(X)
        reg = RANSACRegressor(random_state=0).fit(X, angles[ich, :])
        angles_corrected[ich, :] = reg.predict(X)

    save_dir = processed_path / data_path / "reg"
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
