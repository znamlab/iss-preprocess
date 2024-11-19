import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from znamutils import slurm_it

from ..image.correction import apply_illumination_correction
from ..io import (
    get_channel_round_transforms,
    get_processed_path,
    get_roi_dimensions,
    load_ops,
    load_sequencing_rounds,
    load_tile_by_coors,
)
from ..pipeline.register import load_and_register_sequencing_tile
from ..reg import correct_by_block, get_channel_reference_images
from ..vis import animate_sequencing_rounds
from ..vis.diagnostics import (
    plot_affine_debug_images,
    plot_round_registration_diagnostics,
)
from ..vis.utils import plot_matrix_difference, plot_matrix_with_colorbar
from . import _get_some_tiles


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_ref_tile_registration(data_path, prefix="genes_round"):
    """Plot the reference tile registration and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    print("Loading and registering sequencing tile")
    reg_stack, _ = load_and_register_sequencing_tile(
        data_path,
        filter_r=False,
        correct_channels=False,
        correct_illumination=False,
        corrected_shifts="reference",
        tile_coors=ops["ref_tile"],
        suffix=ops[f"{prefix.split('_')[0]}_projection"],
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=None,
    )
    reg_stack = reg_stack[:, :, np.argsort(ops["camera_order"]), :]
    plot_round_registration_diagnostics(
        reg_stack, target_folder, fname_base=f"initial_ref_tile_registration_{prefix}"
    )


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_tile_registration(
    data_path,
    prefix="genes_round",
    corrections=("best"),
    tile_coords=None,
):
    """Check the registration of sequencing data for some tiles

    If `tile_coords` is None, will select 10 tiles. If `ops` has a `xx_ref_tiles`
    matching prefix, these will be part of the 10 tiles. The remaining tiles will be
    selected randomly.

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
        corrections (tuple, optional): Corrections to plot. Defaults to ('best').
        tile_coords (list, optional): List of tile coordinates to process. If None, will
            select 10 tiles. Defaults to None.
    """
    if isinstance(corrections, str):
        corrections = [corrections]
    processed_path = get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / prefix
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    tile_coords = _get_some_tiles(
        data_path, prefix=f"{prefix}_1_1", tile_coords=tile_coords
    )

    for tile in tile_coords:
        for correction in corrections:
            reg_stack, bad_pixels = load_and_register_sequencing_tile(
                data_path,
                filter_r=False,
                correct_channels=False,
                correct_illumination=True,
                corrected_shifts=correction,
                tile_coors=tile,
                suffix=ops[f"{prefix.split('_')[0]}_projection"],
                prefix=prefix,
                nrounds=nrounds,
                specific_rounds=None,
            )
            reg_stack = reg_stack[:, :, np.argsort(ops["camera_order"]), :]
            tile_name = "_".join([str(x) for x in tile])
            fname_base = f"check_reg_{prefix}_{tile_name}_{correction}"
            plot_round_registration_diagnostics(reg_stack, target_folder, fname_base)


@slurm_it(conda_env="iss-preprocess")
def check_shift_correction(
    data_path,
    prefix="genes_round",
    roi_dimension_prefix="genes_round_1_1",
    within=True,
    between=True,
):
    """Plot the shift correction and save it in the figures folder

    Compare the ransac output to the tile-by-tile shifts and plot
    matrix of differences

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
        roi_dimension_prefix (str, optional): Prefix of the roi dimensions. Defaults to
            "genes_round_1_1".
        within (bool, optional): Plot within channel shifts. Defaults to True.
        between (bool, optional): Plot between channel shifts. Defaults to True.

    """
    print(f"Checking shift correction for {prefix}")
    processed_path = get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / prefix
    target_folder.mkdir(exist_ok=True, parents=True)

    reg_dir = processed_path / "reg" / prefix
    ndims = get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = load_ops(data_path)
    if "use_rois" in ops:
        ndims = ndims[np.in1d(ndims[:, 0], ops["use_rois"])]
    nc = len(ops["camera_order"])
    nr = ops.get(f"{prefix}s", 1)

    # Now plot them.
    def get_shifts(which, archive):
        if which == "within":
            return archive[f"shifts_{which}_channels"]
        elif which == "between":
            return archive["matrix_between_channels"][:, :2, 2]

    def get_angle(which, archive):
        if which == "within":
            return archive[f"angles_{which}_channels"]
        elif which == "between":
            matrix = archive["matrix_between_channels"]
            # we return an estimate of the angle assuming it is a pure rotation matrix
            return [np.rad2deg(np.arctan2(m[1, 0], m[0, 0])) for m in matrix]

    def get_data(which, roi, nr, ntiles):
        if nr > 1:
            raw = np.zeros([nc, nr, 3, *ntiles]) + np.nan
        else:
            raw = np.zeros([nc, 3, *ntiles]) + np.nan
        corrected = np.zeros_like(raw) + np.nan
        best = np.zeros_like(raw) + np.nan

        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                try:
                    data = np.load(reg_dir / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz")
                    raw[..., :2, ix, iy] = get_shifts(which, data)
                    raw[..., 2, ix, iy] = get_angle(which, data)
                except FileNotFoundError:
                    pass
                data = get_channel_round_transforms(
                    data_path,
                    prefix,
                    tile_coors=(roi, ix, iy),
                    shifts_type="corrected",
                    load_file=True,
                )
                data = np.load(
                    reg_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                corrected[..., :2, ix, iy] = get_shifts(which, data)
                corrected[..., 2, ix, iy] = get_angle(which, data)
                tf_best = reg_dir / f"tforms_best_{prefix}_{roi}_{ix}_{iy}.npz"
                if tf_best.exists():
                    data = np.load(tf_best)
                    best[..., :2, ix, iy] = get_shifts(which, data)
                    best[..., 2, ix, iy] = get_angle(which, data)
        return raw, corrected, best

    if within:
        print("Plotting within channel shifts")
        # For "within channels" we plot the shifts for each channel and each round
        fig = plt.figure(figsize=(4 * nr * 2, 2 * 4 * nc))
        for roi, *ntiles in ndims:
            # need to add 1 because of get_roi_dim
            ntiles = [n + 1 for n in ntiles]
            raw, corrected, best = get_data("within", roi, nr=nr, ntiles=ntiles)
            corr_feature = ["Shift x", "Shift y"]
            fig.clear()
            axes = fig.subplots(nrows=nc * 4, ncols=nr * 2)
            for c in range(nc):
                for ifeat, feat in enumerate(corr_feature):
                    raw_to_plot = raw[c, :, ifeat, ...]
                    corr_to_plot = corrected[c, :, ifeat, ...]
                    best_to_plot = best[c, :, ifeat, ...]
                    plot_matrix_difference(
                        raw=raw_to_plot,
                        corrected=corr_to_plot,
                        col_labels=[f"Round {i} {feat}" for i in np.arange(nr)],
                        range_min=[5 if ifeat < 2 else 0.1] * nr,
                        range_max=[10 if ifeat < 2 else 1] * nr,
                        axes=axes[c * 4 : c * 4 + 3, ifeat * nr : (ifeat + 1) * nr],
                        line_labels=("Raw", f"CHANNEL {c}\nCorrected", "Difference"),
                    )
                    # also plot best
                    for ir in range(nr):
                        ax = axes[c * 4 + 3, ifeat * nr + ir]
                        data = best_to_plot[ir]
                        vmin, vmax = data.min(), data.max()
                        rng = vmin - vmax
                        rng_min = 5 if ifeat < 2 else 0.1
                        if rng < rng_min:
                            vmin -= (rng_min - rng) / 2
                            vmax += (rng_min - rng) / 2
                        plot_matrix_with_colorbar(
                            best_to_plot[ir].T, ax, vmin=vmin, vmax=vmax
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                    axes[c * 4 + 3, ifeat * nr].set_ylabel("Best")
            fig_title = f"{prefix} Correct shift within channels\n"
            fig_title += f"ROI {roi}"
            fig.suptitle(fig_title)
            fig.subplots_adjust(
                wspace=0.15, hspace=0, bottom=0.01, top=0.95, right=0.95, left=0.1
            )

            fname = fig_title.lower().replace(" ", "_").replace("\n", "_")
            fig.savefig(target_folder / (fname + ".png"))
    if between:
        print("Plotting between channel shifts")
        # now do "between channels"
        nrois = len(ndims)
        nrows = nrois * 4
        ncols = 2 * nc
        fig = plt.figure(figsize=(4 * ncols, 2 * nrows))
        axes = fig.subplots(nrows=nrows, ncols=ncols)
        for ir, (roi, *ntiles) in enumerate(ndims):
            raw, corrected, best = get_data("between", roi, nr=1, ntiles=ntiles)
            corr_feature = ["Shift x", "Shift y"]
            for ifeat, feat in enumerate(corr_feature):
                raw_to_plot = raw[:, ifeat, ...]
                corr_to_plot = corrected[:, ifeat, ...]
                best_to_plot = best[:, ifeat, ...]
                plot_matrix_difference(
                    raw=raw_to_plot,
                    corrected=corr_to_plot,
                    col_labels=[f"Channel {i} {feat}" for i in np.arange(nc)],
                    range_min=[5 if ifeat < 2 else 0.1] * nc,
                    range_max=[20 if ifeat < 2 else 1] * nc,
                    axes=axes[ir * 4 : ir * 4 + 3, ifeat * nc : (ifeat + 1) * nc],
                    line_labels=("Raw", f"ROI {ir}\nCorrected", "Difference"),
                )
                # also plot best
                for ic in range(nc):
                    ax = axes[ir * 4 + 3, ifeat * nc + ic]
                    data = best_to_plot[ic]
                    vmin, vmax = data.min(), data.max()
                    rng = vmin - vmax
                    rng_min = 1 if ifeat < 2 else 0.1
                    if rng < rng_min:
                        vmin -= (rng_min - rng) / 2
                        vmax += (rng_min - rng) / 2
                    plot_matrix_with_colorbar(
                        best_to_plot[ic].T, ax, vmin=vmin, vmax=vmax
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                axes[ir * 4 + 3, ifeat * nc].set_ylabel("Best")
        fig_title = f"{prefix} Correct shift between channels"
        fig.suptitle(fig_title)
        fig.subplots_adjust(
            wspace=0.15, hspace=0, bottom=0.01, top=0.95, right=0.95, left=0.1
        )
        fname = fig_title.lower().replace(" ", "_").replace("\n", "_")
        fig.savefig(target_folder / (fname + ".png"))
    print("Done")


def check_sequencing_tile_registration(data_path, tile_coords, prefix="genes_round"):
    """Plot the a mp4 of registered tile and save it in the figures folder

    This will load the data after ransac correction

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"

    target_folder.mkdir(exist_ok=True, parents=True)

    ops = load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    reg_stack, bad_pixels = load_and_register_sequencing_tile(
        data_path,
        filter_r=False,
        correct_channels=False,
        correct_illumination=False,
        corrected_shifts=ops["corrected_shifts"],
        tile_coors=tile_coords,
        suffix=ops[f"{prefix.split('_')[0]}_projection"],
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=None,
    )

    # compute vmax based on round 0
    vmaxs = np.quantile(reg_stack[..., 0], 0.9999, axis=(0, 1))
    center = np.array(reg_stack.shape[:2]) // 2
    view = np.array([center - 200, center + 200]).T

    tilename = "_".join([str(x) for x in tile_coords])
    animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"registration_tile{tilename}_{prefix}.mp4",
        vmax=vmaxs,
        extent=(view[0], view[1]),
        channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    )


def check_affine_channel_registration(
    data_path,
    prefix="genes_round",
    tile_coords=None,
    projection=None,
    binarise_quantile="ops",
    block_size="ops",
    overlap="ops",
    max_residual="ops",
    ref_ch="ops",
    correct_illumination="ops",
):
    ops = load_ops(data_path)
    ops_pref = prefix.split("_")[0].lower()
    if binarise_quantile == "ops":
        binarise_quantile = ops[f"{ops_pref}_binarise_quantile"]
    if block_size == "ops":
        block_size = ops.get(f"{ops_pref}_reg_block_size", 256)
    if overlap == "ops":
        overlap = ops.get(f"{ops_pref}_reg_overlap", 0.5)
    if max_residual == "ops":
        max_residual = ops.get(f"{ops_pref}_max_residual", 2)
    if correct_illumination == "ops":
        correct_illumination = ops.get(f"{ops_pref}_reg_correct_illumination", False)
    if ref_ch == "ops":
        ref_ch = ops["ref_ch"]
        ref_ch = ops.get(f"{ops_pref}_ref_ch", ref_ch)
    if "_1" not in prefix:
        roi_dims = get_roi_dimensions(data_path, prefix=f"{prefix}_1_1")
        multi_rounds = True
    else:
        roi_dims = get_roi_dimensions(data_path, prefix=f"{prefix}")
        multi_rounds = False

    # select some tiles
    if tile_coords is None:
        # check if ops has a ref tile
        if f"{prefix.split('_')[0]}_ref_tiles" in ops:
            tile_coords = ops[f"{prefix.split('_')[0]}_ref_tiles"]
            nrandom = 10 - len(tile_coords)
        else:
            tile_coords = []
            nrandom = 10
        # select random tiles
        if nrandom > 0:
            for i in range(nrandom):
                # pick a roi randomly
                roi = np.random.choice(roi_dims[:, 0])
                # pick a tile inside that roi
                ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0]
                tile_coords.append([roi, *np.random.randint(0, ntiles)])
    elif isinstance(tile_coords[0], int):
        tile_coords = [tile_coords]

    target_folder = (
        get_processed_path(data_path)
        / "figures"
        / "registration"
        / f"affine_transform_{prefix}"
    )
    if not target_folder.exists():
        target_folder.mkdir(parents=True, exist_ok=True)
    # this is fast to run, so we re-run with diag mode
    if projection is None:
        projection = ops[f"{prefix.split('_')[0].lower()}_projection"]

    fig = plt.figure(figsize=(2 * 7, 1.5 * 3))
    for tile_coors in tile_coords:
        fig.clear()
        if ops["align_method"] != "affine":
            print("This function is only for affine registration")
            return
        ops = load_ops(data_path)

        median_filter = ops["reg_median_filter"]

        if multi_rounds:
            nrounds = ops[prefix + "s"]
            stack = load_sequencing_rounds(
                data_path, tile_coors, prefix=prefix, suffix=projection, nrounds=nrounds
            )
            # load corrections
            tforms = get_channel_round_transforms(
                data_path, prefix, tile_coors, ops["corrected_shifts"]
            )
            (
                std_stack,
                mean_stack,
            ) = get_channel_reference_images(
                stack,
                tforms["angles_within_channels"],
                tforms["shifts_within_channels"],
            )
        else:
            std_stack = load_tile_by_coors(
                data_path=data_path,
                tile_coors=tile_coors,
                prefix=prefix,
                suffix=projection,
            )

        if correct_illumination:
            std_stack = apply_illumination_correction(data_path, std_stack, prefix)

        matrices, debug_info = correct_by_block(
            std_stack,
            ch_to_align=ref_ch,
            median_filter_size=median_filter,
            binarise_quantile=binarise_quantile,
            max_residual=max_residual,
            debug=True,
            block_size=block_size,
            overlap=overlap,
        )
        plot_affine_debug_images(debug_info, fig=fig)
        fig.suptitle(f"{prefix} - Tile {tile_coors}")
        tile_name = "_".join([str(x) for x in tile_coors])
        fig.savefig(target_folder / f"affine_debug_{prefix}_{tile_name}.png")

    return tile_coors, matrices, debug_info


@slurm_it(conda_env="iss-preprocess")
def check_tile_shifts(
    data_path, prefix, rois=None, roi_dimension_prefix="genes_round_1_1"
):
    """Plot estimation of shifts/angle for registration to ref

    Compare raw measures to ransac

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        rois (list): List of ROIs to process. If None, will either use ops["use_rois"]
            if it is defined, or all ROIs otherwise. Defaults to None
        roi_dimension_prefix (str, optional): prefix to load roi dimension. Defaults to
            "genes_round_1_1"

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "registration" / prefix
    figure_folder.mkdir(exist_ok=True, parents=True)
    roi_dims = get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], ops["use_rois"])]
    roi_dims[:, 1:] = roi_dims[:, 1:] + 1
    figs = {}
    data = get_channel_round_transforms(
        data_path, prefix, tile_coors=(roi_dims[0, 0], 0, 0), shifts_type="single_tile"
    )
    nchannels = data["shifts_within_channels"].shape[0]
    nrounds = data["shifts_within_channels"].shape[1]
    for roi, *ntiles in roi_dims:
        shifts_within_channels_raw = np.zeros([nchannels, nrounds, 2, *ntiles]) + np.nan
        shifts_within_channels_corrected = shifts_within_channels_raw.copy()
        shifts_between_channels_raw = np.zeros([nchannels, 2, *ntiles]) + np.nan
        shifts_between_channels_corrected = shifts_between_channels_raw.copy()

        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                try:
                    data = get_channel_round_transforms(
                        data_path, prefix, (roi, ix, iy), shifts_type="single_tile"
                    )
                    shifts_within_channels_raw[:, :, :, ix, iy] = data[
                        "shifts_within_channels"
                    ]
                    shifts_between_channels_raw[:, :, ix, iy] = data[
                        "matrix_between_channels"
                    ][:, :2, 2]
                except FileNotFoundError:
                    pass
                data = get_channel_round_transforms(
                    data_path, prefix, (roi, ix, iy), shifts_type="corrected"
                )
                shifts_within_channels_corrected[:, :, :, ix, iy] = data[
                    "shifts_within_channels"
                ]
                shifts_between_channels_corrected[:, :, ix, iy] = data[
                    "matrix_between_channels"
                ][:, :2, 2]
        # create a PDF for each roi
        with PdfPages(figure_folder / f"tile_shifts_{prefix}_roi{roi}.pdf") as pdf:
            for ch in range(nchannels):
                for dim in range(2):
                    fig = plot_matrix_difference(
                        raw=shifts_within_channels_raw[ch, :, dim, :, :],
                        corrected=shifts_within_channels_corrected[ch, :, dim, :, :],
                        col_labels=[f"round {i}" for i in range(nrounds)],
                        range_min=np.ones(nrounds) * 5,
                    )
                    fig.suptitle(f"Dim {dim} shifts. {prefix} ROI {roi} channel {ch}")
                    pdf.savefig(fig)
                    figs[roi] = fig
    return figs
