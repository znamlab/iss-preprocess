import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from image_tools.registration import phase_correlation as mpc
from image_tools.similarity_transforms import make_transform, transform_image
from scipy.ndimage import median_filter
from skimage.morphology import disk
from skimage.transform import AffineTransform, warp
from skimage import transform
from znamutils import slurm_it

import iss_preprocess as iss

from iss_preprocess import vis
from iss_preprocess.image.correction import apply_illumination_correction
from iss_preprocess.io import (
    get_roi_dimensions,
    load_ops,
    load_stack,
    load_tile_by_coors,
)
from iss_preprocess.reg import (
    estimate_rotation_translation,
    estimate_scale_rotation_translation,
)
from iss_preprocess import pipeline
from iss_preprocess.pipeline.register import align_spots, align_cell_dataframe


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


def register_all_rois_within(
    data_path,
    prefix=None,
    ref_ch=None,
    suffix="max-median",
    correct_illumination=True,
    reload=False,
    save_plot=True,
    dimension_prefix="genes_round_1_1",
    verbose=1,
    use_slurm=True,
    job_dependency=None,
    scripts_name=None,
    slurm_folder=None,
):
    """Register all tiles within each ROI

    Args:
        data_path (str): Relative path to data
        prefix (str, optional): Prefix of acquisition to register. If None, will use the
            one in `ops`. Defaults to None.
        ref_ch (int, optional): Reference channel to use for registration. If None, will
            use the one in `ops`. Defaults to None.
        suffix (str, optional): Suffix to use to load the images. Defaults to
            'max-median'.
        correct_illumination (bool, optional): Correct illumination before registration.
            Defaults to True.
        reload (bool, optional): Reload saved shifts if True. Defaults to False.
        save_plot (bool, optional): Save diagnostic plot. Defaults to True.
        dimension_prefix (str, optional): Prefix to use to find ROI dimension. Used
            only if the acquisition is an overview. Defaults to 'genes_round_1_1'.
        verbose (int, optional): Verbosity level. Defaults to 1.
        use_slurm (bool, optional): Use SLURM to parallelize the registration. Defaults
            to True.
        job_dependencies (list, optional): List of job dependencies. Defaults to None.
        script_names (str, optional):Script names for slurm jobs. Defaults to None.
        slurm_folder (str, optional): Folder to save SLURM logs. Defaults to None.

    Returns:
        list: List of outputs from `register_within_acquisition`
    """
    ops = load_ops(data_path)
    if prefix is None:
        prefix = ops["reference_prefix"]

    min_corrcoef = ops.get(f"{prefix}_min_corrcoef", 0.3)
    max_delta_shift = ops.get(f"{prefix}_max_delta_shift", 20)

    roi_dims = get_roi_dimensions(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    if use_slurm:
        if slurm_folder is None:
            slurm_folder = Path.home() / "slurm_logs" / data_path / "register_within"
        slurm_folder.mkdir(exist_ok=True, parents=True)
    else:
        slurm_folder = None

    outs = []
    for roi in roi_dims[use_rois, 0]:
        print(f"Registering ROI {roi}")
        scripts_name = f"register_within_{prefix}_{roi}"
        outs.append(
            register_within_acquisition(
                data_path,
                prefix=prefix,
                roi=roi,
                ref_ch=ref_ch,
                suffix=suffix,
                correct_illumination=correct_illumination,
                reload=reload,
                save_plot=save_plot,
                dimension_prefix=dimension_prefix,
                min_corrcoef=min_corrcoef,
                max_delta_shift=max_delta_shift,
                verbose=verbose,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=scripts_name,
                job_dependency=job_dependency,
            )
        )
    return outs


@slurm_it(
    conda_env="iss-preprocess", print_job_id=True, slurm_options={"time": "2:00:00"}
)
def register_within_acquisition(
    data_path,
    roi,
    prefix=None,
    ref_ch=None,
    suffix="max",
    correct_illumination=False,
    reload=True,
    save_plot=False,
    dimension_prefix="genes_round_1_1",
    min_corrcoef=0.6,
    max_delta_shift=20,
    verbose=2,
):
    """Estimate shifts between all adjacent tiles of an roi

    Saves shifts as `reg/f"{prefix}_within"/f"{prefix}_{roi}_shifts.npz"`

    Args:
        data_path (str): path to image stacks.
        roi (int): id of ROI to load.
        prefix (str, optional): Full name of the acquisition folder.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        suffix (str, optional): File name suffix. Defaults to 'proj'.
        correct_illumination (bool, optional): Remove black levels and correct illumination
            before registration if True, return raw data otherwise. Default to False
        reload (bool, optional): If target file already exists, reload instead of
            recomputing. Defaults to True
        save_plot (bool, optional): If True save diagnostic plot. Defaults to False
        dimension_prefix (str, optional): Prefix to use to find ROI dimension. Used
            only if the acquisition is an overview. Defaults to 'genes_round_1_1'
        min_corrcoef (float, optional): Minimum correlation coefficient to consider a
            shift as valid. Defaults to 0.6.
        max_delta_shift (int, optional): Maximum shift, relative to median of the row or
            column, to consider a shift as valid. Defaults to 20.
        verbose (int, optional): Verbosity level. Defaults to 2.

    Returns:
        dict: dictionary containing the shifts, tile shape and number of tiles

    """
    ops = load_ops(data_path)
    if prefix is None:
        prefix = ops["reference_prefix"]
    if ref_ch is None:
        ref_ch = ops["ref_ch"]

    verbose = int(verbose)
    save_fname = (
        iss.io.get_processed_path(data_path)
        / "reg"
        / f"{prefix}_within"
        / f"{prefix}_{roi}_shifts.npz"
    )
    if reload and save_fname.exists():
        print("Reloading saved shifts")
        print(f"Shifts at {save_fname}")
        return np.load(save_fname)

    ndim = get_roi_dimensions(data_path, dimension_prefix)
    # roi_dims is read from file name (0-based), the actual number of tile needs +1
    ntiles = ndim[ndim[:, 0] == roi][0][1:] + 1

    shifts = np.zeros((ntiles[0], ntiles[1], 4)) + np.nan
    xcorr_max = np.zeros((ntiles[0], ntiles[1], 2))
    with tqdm(
        total=np.prod(ntiles), desc="Registering tiles", disable=verbose < 1
    ) as pbar:
        pbar.set_postfix_str(f"ROI {roi}")
        for tilex in range(ntiles[0]):
            pbar.set_postfix_str(f"ROI {roi}, tile {tilex}")
            for tiley in range(ntiles[1]):
                pbar.set_postfix_str(f"ROI {roi}, tile {tilex}, {tiley}")
                pbar.update(1)
                reg_out = register_adjacent_tiles(
                    data_path,
                    ref_coors=(roi, tilex, tiley),
                    ref_ch=ref_ch,
                    suffix=suffix,
                    prefix=prefix,
                    correct_illumination=correct_illumination,
                    verbose=verbose > 1,
                    overlap_ratio=0.01,
                )

                shifts[tilex, tiley] = np.hstack(
                    [reg_out["shift_right"], reg_out["shift_down"]]
                )
                xcorr_max[tilex, tiley] = [reg_out["corr_right"], reg_out["corr_down"]]

        pbar.set_description(f"Correcting shifts")
        clean_down = shifts[..., 2:].copy()
        # Ignore shifts with low correlation
        bad_down = xcorr_max[..., 1] < min_corrcoef
        clean_down[bad_down, :] = np.nan
        # Find the median of remain shift along each row
        med_by_row = np.zeros((ntiles[1], 2)) + np.nan
        med_by_row[1:, :] = np.nanmedian(clean_down[:, 1:], axis=0)
        delta_shift = np.linalg.norm(shifts[..., 2:] - med_by_row[None, :, :], axis=2)
        # replace shifts that are either low corr or too far from median
        bad_down = bad_down | (delta_shift > max_delta_shift)
        clean_down[bad_down, :] = med_by_row[np.where(bad_down)[1]]

        # Same for right shifts
        clean_right = shifts[..., :2].copy()
        bad_right = xcorr_max[..., 0] < min_corrcoef
        clean_right[bad_right, :] = np.nan
        med_by_col = np.zeros((ntiles[0], 2)) + np.nan
        med_by_col[1:, :] = np.nanmedian(clean_right[1:], axis=1)
        delta_shift = np.linalg.norm(clean_right - med_by_col[:, None, :], axis=2)
        bad_right = bad_right | (delta_shift > max_delta_shift)
        clean_right[bad_right, :] = med_by_col[np.where(bad_right)[0]]

        pbar.set_description(f"Saving shifts")
        save_fname.parent.mkdir(exist_ok=True)
        output = dict(
            raw_shift_right=shifts[..., :2],
            raw_shift_down=shifts[..., 2:],
            shift_right=clean_right,
            shift_down=clean_down,
            xcorr_right=xcorr_max[..., 0],
            xcorr_down=xcorr_max[..., 1],
            tile_shape=reg_out["tile_shape"],
            ntiles=ntiles,
        )
        np.savez(
            save_fname,
            **output,
        )
        if save_plot:
            pbar.set_description(f"Saving plot")
            vis.diagnostics.adjacent_tiles_registration(
                data_path,
                prefix,
                roi=roi,
                shifts=np.dstack([clean_right, clean_down]),
                raw_shifts=shifts,
                xcorr_max=xcorr_max,
                min_corrcoef=min_corrcoef,
                max_delta_shift=max_delta_shift,
            )
    if verbose > 1:
        print(f"Saved shifts to {save_fname}")
    return output


def register_adjacent_tiles(
    data_path,
    ref_coors=None,
    ref_ch=0,
    suffix="max",
    prefix="genes_round_1_1",
    correct_illumination=False,
    overlap_ratio=0.01,
    verbose=True,
    debug=False,
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
        overlap_ratio (float, optional): Minimum overlap between masks to consider the
            correlation results. Defaults to 0.01.
        verbose (bool, optional): If True, print warnings when shifts are large.
            Defaults to True.
        debug (bool, optional): Return additional information for debugging. Defaults to
            False.


    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """

    ops = load_ops(data_path)
    if ref_coors is None:
        ref_coors = ops["ref_tile"]

    # small helper to prepare the stack
    def prep_stack(stack):
        if correct_illumination:
            stack = apply_illumination_correction(data_path, stack, prefix)
        if ops["reg_median_filter"]:
            msize = ops["reg_median_filter"]
            assert isinstance(msize, int), "reg_median_filter must be an integer"
            stack = median_filter(stack, footprint=disk(msize), axes=(0, 1))
        return stack

    tile_ref = load_tile_by_coors(
        data_path, tile_coors=ref_coors, suffix=suffix, prefix=prefix
    )
    tile_ref = prep_stack(tile_ref)

    ypix = tile_ref.shape[0]
    xpix = tile_ref.shape[1]
    reg_pix_x = int(xpix * ops["reg_fraction"])
    reg_pix_y = int(ypix * ops["reg_fraction"])

    roi_dims = get_roi_dimensions(data_path)
    ntiles = roi_dims[roi_dims[:, 0] == ref_coors[0], 1:][0] + 1

    if debug:
        db_dict = dict(reg_pix_x=reg_pix_x, reg_pix_y=reg_pix_y)

    # Register the right tile
    right_offset = 1 if ops["x_tile_direction"] == "left_to_right" else -1
    right_coors = (ref_coors[0], ref_coors[1] + right_offset, ref_coors[2])
    if right_coors[1] < 0 or right_coors[1] >= ntiles[0]:
        shift_right = [np.nan, np.nan]
        corr_right = np.nan
    else:
        tile_right = load_tile_by_coors(
            data_path, tile_coors=right_coors, suffix=suffix, prefix=prefix
        )
        tile_right = prep_stack(tile_right)
        fixed_part = tile_ref[:, -reg_pix_x * 2 :, ref_ch]
        moving_part = tile_right[:, : reg_pix_x * 2, ref_ch]
        fixed_mask = np.zeros_like(fixed_part, dtype=bool)
        fixed_mask[:, -reg_pix_x:] = True
        moving_mask = np.zeros_like(moving_part, dtype=bool)
        moving_mask[:, :reg_pix_x] = True
        shift_right, corr_right, _, _ = mpc.phase_correlation(
            fixed_part,
            moving_part,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
        )
        if debug:
            db_dict["raw_right"] = np.array(shift_right)
            db_dict["tile_right"] = tile_right[..., ref_ch]
        # roll back the shift, since we have reg_pix_x padding
        shift_right[1] = np.mod(shift_right[1], 2 * reg_pix_x) - reg_pix_x
        if verbose and (np.abs(shift_right[1]) >= reg_pix_x * 0.3):
            warnings.warn(
                f"Shift to right tile ({right_coors}) is large: {shift_right}"
                f"({shift_right/reg_pix_x*100}% of overlap). Check that everything is fine."
            )
        shift_right += [0, xpix - reg_pix_x]
        if ops["x_tile_direction"] != "left_to_right":
            shift_right = -shift_right

    # Register the down tile
    down_offset = 1 if ops["y_tile_direction"] == "bottom_to_top" else -1
    down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + down_offset)
    if down_coors[2] < 0 or down_coors[2] >= ntiles[1]:
        shift_down = [np.nan, np.nan]
        corr_down = np.nan
    else:
        tile_down = load_tile_by_coors(
            data_path, tile_coors=down_coors, suffix=suffix, prefix=prefix
        )
        tile_down = prep_stack(tile_down)
        fixed_part = tile_ref[: reg_pix_y * 2 :, :, ref_ch]
        moving_part = tile_down[-reg_pix_y * 2 :, :, ref_ch]
        fixed_mask = np.zeros_like(fixed_part, dtype=bool)
        fixed_mask[:reg_pix_y, :] = True
        moving_mask = np.zeros_like(moving_part, dtype=bool)
        moving_mask[-reg_pix_y:, :] = True

        shift_down, corr_down, _, _ = mpc.phase_correlation(
            fixed_part,
            moving_part,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            overlap_ratio=overlap_ratio,
        )
        if debug:
            db_dict["raw_down"] = np.array(shift_down)
            db_dict["tile_down"] = tile_down[..., ref_ch]

        # roll back the shift, since we have reg_pix_x padding
        shift_down[0] = np.mod(shift_down[0], 2 * reg_pix_y) - reg_pix_y
        if verbose and (np.abs(shift_down[0]) >= reg_pix_y * 0.3):
            warnings.warn(
                f"Shift to down tile ({down_coors}) is large: {shift_down}"
                f"({shift_down/reg_pix_y*100}% of overlap). Check that everything is fine."
            )
        shift_down -= [ypix - reg_pix_y, 0]
        if ops["y_tile_direction"] != "bottom_to_top":
            shift_down = -shift_down

    output = dict(
        shift_right=shift_right,
        shift_down=shift_down,
        tile_shape=(ypix, xpix),
        corr_right=corr_right,
        corr_down=corr_down,
    )
    if debug:
        db_dict.update(
            dict(
                tile_ref=tile_ref[..., ref_ch],
                ref_coors=ref_coors,
                right_coors=right_coors,
                down_coors=down_coors,
            )
        )
        output.update(db_dict)
    return output


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
        iss.io.get_processed_path(data_path)
        / "reg"
        / f"{prefix}_within"
        / f"{prefix}_{roi}_shifts.npz"
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
        shift_right (numpy.array): X and Y shifts between different columns. Either
            a 2-element array or a ntiles[0] x ntiles[1] x 2 matrix of shifts
        shift_down (numpy.array): X and Y shifts between different rows. Either
            a 2-element array or a ntiles[0] x ntiles[1] x 2 matrix of shifts
        tile_shape (numpy.array): shape of each tile
        ntiles (numpy.array): number of tile rows and columns

    Returns:
        numpy.ndarray: `tile_origins`, ntiles[0] x ntiles[1] x 2 matrix of tile origin
            coordinates
        numpy.ndarray: `tile_centers`, ntiles[0] x ntiles[1] x 2 matrix of tile center
            coordinates

    """
    # Either we have a single x/y shift for all tiles or one for each tile
    if shift_right.ndim == 1:
        assert shift_right.shape[0] == 2, "shift_right must have 2 elements"
        shift_right = np.tile(shift_right, (ntiles[0], ntiles[1], 1))
    else:
        assert shift_right.shape == (
            ntiles[0],
            ntiles[1],
            2,
        ), "shift_right has wrong shape"
    if shift_down.ndim == 1:
        assert shift_down.shape[0] == 2, "shift_down must have 2 elements"
        shift_down = np.tile(shift_down, (ntiles[0], ntiles[1], 1))
    else:
        assert shift_down.shape == (
            ntiles[0],
            ntiles[1],
            2,
        ), "shift_down has wrong shape"

    # replace the first row/col that are NaNs with 0
    # Add an assert to make sure that the first row/col are NaNs. It might not be
    # the case if we didn't handle the microscope direction correctly
    assert np.all(np.isnan(shift_right[0])), "First row of shift_right must be NaN"
    assert np.all(np.isnan(shift_down[:, 0])), "First col of shift_down must be NaN"
    shift_right[0] = 0
    shift_down[:, 0] = 0

    tile_origins = shift_right.cumsum(axis=0) + shift_down.cumsum(axis=1)
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
    shift_file = (
        processed_path
        / "reg"
        / f"{shifts_prefix}_within"
        / f"{shifts_prefix}_{roi}_shifts.npz"
    )
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
    shifts = np.load(
        processed_path
        / "reg"
        / f"{ref_prefix}_within"
        / f"{ref_prefix}_{roi}_shifts.npz"
    )
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    tile_shape = shifts["tile_shape"]
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"], shifts["shift_down"], shifts["tile_shape"], ntiles=ntiles
    )
    tile_origins = np.round(tile_origins).astype(int)
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
            if stack.ndim == 4:
                stack = stack[:, :, :, 0]  # unique round
            stack = stack[:, :, channels]
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

    if "mask" in prefix:
        max_val = np.max(stitched_stack)
        dtype = np.uint16 if max_val < 2**16 else np.uint32
        stitched_stack = stitched_stack.astype(dtype)
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

    for tx in range(ntiles[0]):
        for ty in range(ntiles[1]):
            try:
                spots = align_spots(data_path, tile_coors=(iroi, tx, ty), prefix=prefix)
                spots["x_in_tile"] = spots["x"].copy()
                spots["y_in_tile"] = spots["y"].copy()
                spots["tile"] = f"{iroi}_{tx}_{ty}"
                spots["x"] = spots["x"] + tile_origins[tx, ty, 1]
                spots["y"] = spots["y"] + tile_origins[tx, ty, 0]

                if not keep_all_spots:
                    # calculate distance to tile centers
                    spot_dist = (
                        spots["x"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 1]
                    ) ** 2 + (
                        spots["y"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 0]
                    ) ** 2
                    home_tile_dist = (spot_dist[:, tx, ty]).copy()
                    spot_dist[:, tx, ty] = np.inf
                    min_spot_dist = np.min(spot_dist, axis=(1, 2))
                    keep_spots = home_tile_dist < min_spot_dist
                else:
                    keep_spots = np.ones(spots.shape[0], dtype=bool)
                all_spots.append(spots[keep_spots])
            except FileNotFoundError:
                print(f"could not load roi {iroi}, tile {tx}, {ty}")

    spots = pd.concat(all_spots, ignore_index=True)
    return spots


@slurm_it(conda_env="iss-preprocessing", slurm_options={"time": "1:00:00", "mem": "8G"})
def stitch_cell_dataframes(data_path, prefix, ref_prefix=None):
    """Stitch cell dataframes across all tiles and ROI.

    Args:
        data_path (str): path to data
        prefix (str): prefix of the cell dataframe to load
        ref_prefix (str, optional): prefix of the reference tiles to use for stitching.
            Defaults to None.

    Returns:
        pandas.DataFrame: stitched cell dataframe
    """

    ops = load_ops(data_path)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]

    stitched_df = align_cell_dataframe(data_path, prefix, ref_prefix=None).copy()
    stitched_df["x_in_tile"] = stitched_df["x"].copy()
    stitched_df["y_in_tile"] = stitched_df["y"].copy()
    stitched_df["tile"] = "Not Processed"
    stitched_df["x"] = np.nan
    stitched_df["y"] = np.nan

    for roi, df in stitched_df.groupby("roi"):
        # find tile origin, final shape, and shifts in reference coordinates
        ref_corners = get_tile_corners(data_path, prefix=ref_prefix, roi=roi)
        ref_origins = ref_corners[..., 0]
        for (tx, ty), tdf in df.groupby(["tilex", "tiley"]):
            stitched_df.loc[tdf.index, "tile"] = f"{roi}_{tx}_{ty}"
            stitched_df.loc[tdf.index, "x"] = tdf["x_in_tile"] + ref_origins[tx, ty, 1]
            stitched_df.loc[tdf.index, "y"] = tdf["y_in_tile"] + ref_origins[tx, ty, 0]
    return stitched_df


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
    print(f"Aligning spots for ROI {roi}")
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
    fname = processed_path / f"{spots_prefix}_spots_{roi}.pkl"
    spots.to_pickle(fname)
    print(f"Saved spots for ROI in {fname}")
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
            scripts_name=f"iss_align_spots_{spots_prefix}_{roi}",
            job_dependency=dependency,
        )


def find_tile_order(
    data_path, prefix=None, xy_stage_name="XYStage", z_stage_name="ZDrive", verbose=True
):
    """Find the order of tiles in a multi-tile acquisition

    Args:
        data_path (str): Relative path to data
        prefix (str, optional): Acquisition prefix. If None, will use the one in ops.
            Defaults to None.
        xy_stage_name (str, optional): Name of the XY stage. Defaults to "XYStage".
        z_stage_name (str, optional): Name of the Z stage. If None, will not load Z
            positions. Defaults to "ZDrive".
        verbose (bool, optional): Print information about the number of tiles found.
            Defaults to True.

    Returns:
        dict: Dictionary of tile order with tuple (roi, col, row) as key and acquisition
            order (across all ROIs) as value.
        pandas.DataFrame: DataFrame containing tile position information.
    """
    ops = load_ops(data_path)
    if prefix is None:
        prefix = ops["reference_prefix"]

    # look for position file
    pos_files = list(iss.io.get_raw_path(data_path).glob("*.pos"))
    pos_files = [f for f in pos_files if prefix in f.stem]

    if len(pos_files) == 0:
        raise FileNotFoundError(f"No position file found for {prefix}")
    elif len(pos_files) > 1:
        warnings.warn(f"Found multiple position files for {prefix}.")
        warnings.warn(f"Using the first one: {pos_files[0]}.")

    pos_file = pos_files[0]
    pos_infos = yaml.safe_load(pos_file.read_text())
    positions = pos_infos["map"]["StagePositions"]["array"]
    tile_name = [pos["Label"]["scalar"] for pos in positions]
    roi = [int(p.split("-")[0]) for p in tile_name]
    if verbose:
        print(f"Found {len(positions)} positions for {len(np.unique(roi))} rois")

    rows = [pos["GridRow"]["scalar"] for pos in positions]
    cols = [pos["GridCol"]["scalar"] for pos in positions]
    x_pos = np.zeros(len(positions)) + np.nan
    y_pos = np.zeros(len(positions)) + np.nan
    z_pos = np.zeros(len(positions)) + np.nan
    for ipos, pos in enumerate(positions):
        device_pos = pos["DevicePositions"]["array"]
        for p in device_pos:
            if p["Device"]["scalar"] == xy_stage_name:
                stage_pos = p["Position_um"]
                x_pos[ipos], y_pos[ipos] = stage_pos["array"]
            if (z_stage_name is not None) and (p["Device"]["scalar"] == z_stage_name):
                z = p["Position_um"]["array"]
                assert len(z) == 1, "Z stage position should be a single value"
                z_pos[ipos] = z[0]

    out_df = pd.DataFrame(
        dict(row=rows, col=cols, x=x_pos, y=y_pos, roi=roi, tile_name=tile_name)
    )
    if z_stage_name is not None:
        out_df["z"] = z_pos

    tile_order = {}
    for irow, row in out_df.iterrows():
        tile = (row["roi"], row["col"], row["row"])
        assert tile not in tile_order, f"Tile {tile} already found"
        tile_order[tile] = irow

    return tile_order, out_df


def find_tile_overlap(data_path, ref_prefix, tile_coor1, tile_coor2):
    """Find the overlap between two tiles

    If tile1 is the stack, the overlap can be accessed by:
    tile1[overlap_tile_1[1]:overlap_tile_1[3], overlap_tile_1[0]:overlap_tile_1[2]]

    Args:
        rect1 (tuple): Rectangle coordinates (x0, y0, x1, y1)
        rect2 (tuple): Rectangle coordinates (x0, y0, x1, y1)

    Returns:
        tuple: Overlap in global coordinates (x0, y0, x1, y1)
        tuple: Overlap in tile 1 (x0, y0, x1, y1)
        tuple: Overlap in tile 2 (x0, y0, x1, y1)
    """
    if tile_coor1[0] != tile_coor2[0]:
        # not the same ROI
        return None, None, None
    corners = get_tile_corners(data_path, prefix=ref_prefix, roi=tile_coor1[0])
    rect1 = corners[tile_coor1[1], tile_coor1[2]]
    rect2 = corners[tile_coor2[1], tile_coor2[2]]

    # rect has the 4 corner, we just want the x0, y0, x1, y1
    # but tile corners are row/col, not x/y, so swap
    rect1 = (rect1[1, 0], rect1[0, 0], rect1[1, 2], rect1[0, 2])
    rect2 = (rect2[1, 0], rect2[0, 0], rect2[1, 2], rect2[0, 2])

    x0 = max(rect1[0], rect2[0])
    y0 = max(rect1[1], rect2[1])
    x1 = min(rect1[2], rect2[2])
    y1 = min(rect1[3], rect2[3])
    if x0 > x1 or y0 > y1:
        return None, None, None

    overlap = (x0, y0, x1, y1)
    overlap_tile_1 = (
        overlap[0] - rect1[0],
        overlap[1] - rect1[1],
        overlap[2] - rect1[0],
        overlap[3] - rect1[1],
    )
    overlap_tile_2 = (
        overlap[0] - rect2[0],
        overlap[1] - rect2[1],
        overlap[2] - rect2[0],
        overlap[3] - rect2[1],
    )

    return overlap, overlap_tile_1, overlap_tile_2
