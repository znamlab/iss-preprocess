"""
Module containing diagnostic plots to make sure steps of the pipeline run smoothly

The functions in here do not compute anything useful, but create figures
"""
import numbers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image_tools.similarity_transforms import transform_image
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import median_filter
from skimage.filters import gaussian
from skimage.measure import block_reduce
from skimage.morphology import disk
from znamutils import slurm_it

import iss_preprocess as iss
from iss_preprocess import vis
from iss_preprocess.pipeline import sequencing
from iss_preprocess.pipeline.stitch import stitch_registered


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_ref_tile_registration(data_path, prefix="genes_round"):
    """Plot the reference tile registration and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = iss.io.load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    print("Loading and registering sequencing tile")
    reg_stack, _ = sequencing.load_and_register_sequencing_tile(
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


def plot_round_registration_diagnostics(
    reg_stack, target_folder, fname_base, view_window=200, round_labels=None
):
    """Generate 3 diagnostics tools for the registration of a tile

    - A static figure showing the stack of all rounds
    - An animated figure showing the stack of all rounds
    - A stack of RGB images for fiji

    Args:
        reg_stack (np.ndarray): Registered stack of images
        target_folder (pathlib.Path): Folder to save the figures
        fname_base (str): Base name for the figures
        view_window (int, optional): Half size of the window to show in the figures.
            Defaults to 200.

    Returns:
        None

    """
    # compute vmax based on round 0
    vmins, vmaxs = np.percentile(reg_stack[..., 0], (0.01, 99.99), axis=(0, 1))
    center = np.array(reg_stack.shape[:2]) // 2
    view = np.array([center - view_window, center + view_window]).T
    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])

    print("Static figure")
    fig, rgb_stack = vis.plot_all_rounds(
        reg_stack, view, channel_colors, round_labels=round_labels
    )
    fig.tight_layout()
    fname = target_folder / f"{fname_base}.png"
    fig.savefig(fname)
    print(f"Saved to {fname}")

    # also save the stack for fiji
    iss.io.save.write_stack(
        (rgb_stack * 255).astype("uint8"),
        target_folder / f"{fname_base}_rgb_stack_{reg_stack.shape[-1]}rounds.tif",
    )

    print("Animating")
    vis.animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"{fname_base}.mp4",
        vmax=vmaxs,
        vmin=vmins,
        extent=(view[0], view[1]),
        channel_colors=channel_colors,
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
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / prefix
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = iss.io.load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    tile_coords = _get_some_tiles(
        data_path, prefix=f"{prefix}_1_1", tile_coords=tile_coords
    )

    for tile in tile_coords:
        for correction in corrections:
            reg_stack, bad_pixels = sequencing.load_and_register_sequencing_tile(
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


def _get_some_tiles(data_path, prefix, tile_coords=None):
    """Get some tiles to check registration

    If `tile_coords` is None, will select 10 tiles. If `ops` has a `xx_ref_tiles`
    matching prefix, these will be part of the 10 tiles. The remaining tiles will be
    selected randomly.

    Args:
        data_path (str): Relative path to data folder
        prefix (str): Prefix of the images to load
        tile_coords (list, optional): List of tile coordinates to process. If None, will
            select 10 tiles. Defaults to None.

    Returns:
        list: List of tile coordinates
    """
    ops = iss.io.load_ops(data_path)
    # get stack registered between channel and rounds
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=prefix)
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
    return tile_coords


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_registration_to_reference(data_path, prefix, ref_prefix, tile_coords=None):
    ops = iss.io.load_ops(data_path)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]
    if prefix.endswith("_round"):
        # we have genes_round or barcode_round
        roi_dim_prefix = f"{prefix}_1_1"
        ref_round = ops["ref_round"]
        full_prefix = f"{prefix}_{ref_round + 1}_1"
    else:
        roi_dim_prefix = prefix
        full_prefix = prefix
    tile_coords = _get_some_tiles(
        data_path, prefix=roi_dim_prefix, tile_coords=tile_coords
    )
    processed = iss.io.get_processed_path(data_path)
    target_folder = processed / "figures" / "registration" / f"{prefix}_to_ref"
    target_folder.mkdir(exist_ok=True, parents=True)
    round_labels = [
        "Ref and Reg chan 0/1",
        "Ref and Reg chan 2/3",
        f"Ref: {ref_prefix}",
        "Reg: {prefix}",
    ]
    for tile in tile_coords:
        # get the reference tile
        ref_stack, _ = iss.pipeline.load_and_register_tile(
            data_path, tile, ref_prefix, filter_r=False
        )
        # get the tile to register
        reg_stack, _ = iss.pipeline.load_and_register_tile(
            data_path, tile, full_prefix, filter_r=False
        )
        # concatenate the stacks
        stack = np.concatenate([ref_stack, reg_stack], axis=3)
        # also generate mix stack with 2 channels of each
        mix_stack1 = np.concatenate([ref_stack[:, :, :2], reg_stack[:, :, :2]], axis=2)
        mix_stack2 = np.concatenate([ref_stack[:, :, 2:], reg_stack[:, :, 2:]], axis=2)
        stack = np.concatenate([mix_stack1, mix_stack2, stack], axis=3)

        # reorder the channels
        stack = stack[:, :, np.argsort(ops["camera_order"]), :]
        tile_name = "_".join([str(x) for x in tile])
        fname_base = f"check_reg2ref_{prefix}_{tile_name}"
        plot_round_registration_diagnostics(
            stack, target_folder, fname_base, round_labels=round_labels
        )


def check_affine_channel_registration(
    data_path,
    prefix="genes_round",
    tile_coords=None,
    projection=None,
    binarise_quantile="ops",
    block_size="ops",
    overlap="ops",
):
    ops = iss.io.load_ops(data_path)
    ops_pref = prefix.split("_")[0].lower()
    if binarise_quantile == "ops":
        binarise_quantile = ops[f"{ops_pref}_binarise_quantile"]
    if block_size == "ops":
        block_size = ops.get([f"{ops_pref}_reg_block_size"], 256)
    if overlap == "ops":
        overlap = ops.get([f"{ops_pref}_reg_overlap"], 0.5)
    if "_1" not in prefix:
        roi_dims = iss.io.get_roi_dimensions(data_path, prefix=f"{prefix}_1_1")
        multi_rounds = True
    else:
        roi_dims = iss.io.get_roi_dimensions(data_path, prefix=f"{prefix}")
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
        iss.io.get_processed_path(data_path)
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
        ops = iss.io.load_ops(data_path)
        ref_ch = ops["ref_ch"]
        median_filter = ops["reg_median_filter"]

        if multi_rounds:
            nrounds = ops[prefix + "s"]
            stack = iss.pipeline.load_sequencing_rounds(
                data_path, tile_coors, prefix=prefix, suffix=projection, nrounds=nrounds
            )
            # load corrections
            tforms = iss.pipeline.sequencing.get_channel_round_shifts(
                data_path, prefix, tile_coors, ops["corrected_shifts"]
            )
            (
                std_stack,
                mean_stack,
            ) = iss.reg.rounds_and_channels.get_channel_reference_images(
                stack,
                tforms["angles_within_channels"],
                tforms["shifts_within_channels"],
            )
        else:
            std_stack = iss.pipeline.load_tile_by_coors(
                data_path=data_path,
                tile_coors=tile_coors,
                prefix=prefix,
                suffix=projection,
            )

        matrices, debug_info = iss.reg.rounds_and_channels.correct_by_block(
            std_stack,
            ch_to_align=ref_ch,
            median_filter_size=median_filter,
            binarise_quantile=binarise_quantile,
            debug=True,
            block_size=block_size,
            overlap=overlap,
        )
        iss.vis.diagnostics.plot_affine_debug_images(debug_info, fig=fig)
        fig.suptitle(f"{prefix} - Tile {tile_coors}")
        tile_name = "_".join([str(x) for x in tile_coors])
        fig.savefig(target_folder / f"affine_debug_{prefix}_{tile_name}.png")

    return tile_coors, matrices, debug_info


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
    """
    print(f"Checking shift correction for {prefix}")
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / prefix
    target_folder.mkdir(exist_ok=True, parents=True)

    reg_dir = processed_path / "reg"
    ndims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
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
            raw, corrected, best = get_data("within", roi, nr=nr, ntiles=ntiles)
            corr_feature = ["Shift x", "Shift y"]
            fig.clear()
            axes = fig.subplots(nrows=nc * 4, ncols=nr * 2)
            for c in range(nc):
                for ifeat, feat in enumerate(corr_feature):
                    raw_to_plot = raw[c, :, ifeat, ...]
                    corr_to_plot = corrected[c, :, ifeat, ...]
                    best_to_plot = best[c, :, ifeat, ...]
                    iss.vis.plot_matrix_difference(
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
                        iss.vis.plot_matrix_with_colorbar(
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
                iss.vis.plot_matrix_difference(
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
                    iss.vis.plot_matrix_with_colorbar(
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
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"

    target_folder.mkdir(exist_ok=True, parents=True)

    ops = iss.io.load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    reg_stack, bad_pixels = sequencing.load_and_register_sequencing_tile(
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
    vis.animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"registration_tile{tilename}_{prefix}.mp4",
        vmax=vmaxs,
        extent=(view[0], view[1]),
        channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    )


def check_hybridisation_setup(data_path, prefixes):
    """Plot the hybridisation spot clusters scatter plots and bleedthrough matrices

    Args:
        data_path (str): Relative path to data folder
        prefixes (list): Prefix of the acquisition to check

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    if prefixes is None:
        metadata = iss.io.load_metadata(data_path)
        prefixes = metadata["hybridisation"].keys()
    for hyb_round in prefixes:
        reference_hyb_spots = np.load(
            processed_path / f"{hyb_round}_cluster_means.npz", allow_pickle=True
        )
        figs = iss.vis.plot_clusters(
            [reference_hyb_spots["cluster_means"]],
            reference_hyb_spots["spot_colors"],
            [reference_hyb_spots["cluster_inds"]],
        )
        for fig in figs:
            fig.savefig(figure_folder / f"{hyb_round}_{fig.get_label()}.png")


def check_barcode_calling(data_path):
    """Plot the barcode cluster scatter plots and cluster means and save them in the
    figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "barcode_round"
    figure_folder.mkdir(exist_ok=True)
    reference_barcode_spots = np.load(
        processed_path / "reference_barcode_spots.npz", allow_pickle=True
    )
    cluster_means = np.load(processed_path / "barcode_cluster_means.npy")
    figs = iss.vis.plot_clusters(
        cluster_means,
        reference_barcode_spots["spot_colors"],
        reference_barcode_spots["cluster_inds"],
    )
    for fig in figs:
        fig.savefig(figure_folder / f"barcode_{fig.get_label()}.png")


@slurm_it(conda_env="iss-preprocess")
def check_barcode_basecall(data_path, tile_coords=None):
    """Check that the basecall is correct

    Args:
        path (str): Path to data folder
        tile_coords (list, optional): Tile coordinates to use. Defaults to None.

    Returns:
        plt.Figure: Figure(s) with the basecall
    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "barcode_round"
    figure_folder.mkdir(exist_ok=True)

    # get one of the reference tiles
    ops = iss.io.load_ops(data_path)

    if tile_coords is None:
        tile_coords = _get_some_tiles(
            data_path, prefix="barcode_round_1_1", tile_coords=tile_coords
        )

    if not isinstance(tile_coords[0], numbers.Number):
        # it's a list of tile coordinates
        figs = []
        for tile in tile_coords:
            figs.append(check_barcode_basecall(data_path, tile))
        return figs

    # we have been called with a single tile coordinate
    stack, spot_sign_image, spots = iss.pipeline.sequencing.basecall_tile(
        data_path, tile_coords, save_spots=False
    )

    # Find the place with the highest density of spots
    x, y = spots["x"].values, spots["y"].values
    # Create a grid of potential disk centers
    window = 200
    x_grid, y_grid = np.meshgrid(
        np.arange(window, stack.shape[1] - window, 50),
        np.arange(window, stack.shape[0] - window, 50),
    )
    # Compute the Euclidean distance from each spot to each potential center
    distances = np.sqrt(
        (x[:, None, None] - x_grid) ** 2 + (y[:, None, None] - y_grid) ** 2
    )
    # Count the number of spots within a 200px radius for each potential center
    counts = np.sum(distances <= 100, axis=0)
    center = np.unravel_index(counts.argmax(), counts.shape)
    center = (x_grid[center], y_grid[center])

    nr = ops["barcode_rounds"]
    stack_part = stack[
        center[1] - window : center[1] + window, center[0] - window : center[0] + window
    ]
    valid_spots = spots[
        (spots.x > center[0] - window)
        & (spots.x < center[0] + window)
        & (spots.y > center[1] - window)
        & (spots.y < center[1] + window)
    ]

    # Do the plot
    fig = plt.figure(figsize=(3 * nr, 10))
    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])
    axes = []
    for iround in range(nr):
        rgb_stack = iss.vis.round_to_rgb(
            stack_part, iround, extent=None, channel_colors=channel_colors
        )
        # plot raw fluo
        ax = fig.add_subplot(3, nr, iround + 1)
        axes.append(ax)
        ax.imshow(rgb_stack)
        ax.set_title(f"Round {iround}")
        if iround == nr - 1:
            iss.vis.add_bases_legend(channel_colors, ax.transAxes, fontsize=14)

        # plot basecall, a letter per spot
        ax = fig.add_subplot(3, nr, nr + iround + 1)
        axes.append(ax)
        spots_in_frame = valid_spots.copy()
        spots_in_frame["x"] -= center[0] - window
        spots_in_frame["y"] -= center[1] - window
        vis.plot_spot_called_base(spots_in_frame, ax, iround)

    # plot spot scores
    scores = ["dot_product_score", "spot_score", "mean_score", "mean_intensity"]
    empty = np.zeros(stack_part.shape[:2]) + np.nan
    cmap = plt.cm.viridis
    cmap.set_bad("black")

    for isc, score in enumerate(scores):
        ax = fig.add_subplot(3, len(scores), 2 * len(scores) + isc + 1)
        axes.append(ax)
        sc = ax.scatter(
            valid_spots.x - (center[0] - window),
            valid_spots.y - (center[1] - window),
            c=valid_spots[score],
            s=5,
        )
        cax, cb = iss.vis.plot_matrix_with_colorbar(empty, ax, cmap=cmap)
        cax.clear()
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(score.replace("_", " "))

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, 2 * window)
        ax.set_facecolor("black")
        ax.set_ylim(2 * window, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(
        left=1e-3, right=0.99, bottom=0.01, top=0.98, wspace=0.05, hspace=0.05
    )
    tc = "_".join([str(x) for x in tile_coords])
    fig.savefig(figure_folder / f"barcode_basecall_example_tile_{tc}.png")
    return fig


def check_omp_setup(data_path):
    """Plot the OMP setup, including clustering of reference gene spots and
    gene templates, and save them in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    reference_gene_spots = np.load(
        processed_path / "reference_gene_spots.npz", allow_pickle=True
    )
    omp_stat = np.load(processed_path / "gene_dict.npz", allow_pickle=True)
    nrounds = reference_gene_spots["spot_colors"].shape[0]
    figs = iss.vis.plot_clusters(
        omp_stat["cluster_means"],
        reference_gene_spots["spot_colors"],
        reference_gene_spots["cluster_inds"],
    )
    figs.append(
        iss.vis.plot_gene_templates(
            omp_stat["gene_dict"],
            omp_stat["gene_names"],
            iss.call.BASES,
            nrounds=nrounds,
        )
    )
    for fig in figs:
        fig.savefig(figure_folder / f"omp_{fig.get_label()}.png")


def check_spot_sign_image(data_path):
    """Plot the average spot sign image and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    spot_image = np.load(processed_path / "spot_sign_image.npy")
    iss.vis.plot_spot_sign_image(spot_image)
    plt.savefig(figure_folder / "spot_sign_image.png")


@slurm_it(conda_env="iss-preprocess")
def check_illumination_correction(
    data_path,
    grand_averages=("barcode_round", "genes_round"),
    plot_tilestats=True,
    verbose=True,
):
    """Check if illumination correction average look reasonable

    Args:
        data_path (str): Relative path to data folder
        grand_averages (list, optional): List of grand averages to plot.
            Defaults to ("barcode_round", "genes_round")
        plot_titlestats (bool, optional): Plot a figure of tilestats change. Defaults
            to True
        verbose (bool, optional): Print info about progress. Defaults to True

    """
    processed_path = iss.io.get_processed_path(data_path)
    average_dir = processed_path / "averages"
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    correction_images = dict()
    distributions = dict()

    for fname in average_dir.glob("*average.tif"):
        correction_images[fname.name.replace("_average.tif", "")] = iss.io.load_stack(
            fname
        )
    for fname in average_dir.glob("*_tilestats.npy"):
        distributions[fname.name.replace("_tilestats.npy", "")] = np.load(fname)
    if verbose:
        print(
            f"Found {len(correction_images)} averages"
            + f" and {len(distributions)} tilestats"
        )

    iss.vis.plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
    if plot_tilestats:
        iss.vis.plot_tilestats_distributions(
            data_path, distributions, grand_averages, figure_folder
        )


def debug_reg_to_ref(
    data_path,
    reg_prefix,
    ref_prefix,
    tile_coords=None,
    ref_channels=None,
    reg_channels=None,
    binarise_quantile=0.7,
):
    """Diagnostic functions helping to debug registration to reference

    This redo the steps of register_to_reference to plot intermediate figures
    """
    ops = iss.io.load.load_ops(data_path)
    if tile_coords is None:
        tile_coords = ops["ref_tile"]
    naxes = 1
    if ops["reg_median_filter"]:
        naxes += 1
    if binarise_quantile is not None:
        naxes += 1

    fig, axes = plt.subplots(2, naxes, figsize=(6 * naxes, 5))

    ref_all_channels, _ = iss.pipeline.load_and_register_tile(
        data_path=data_path,
        tile_coors=tile_coords,
        prefix=ref_prefix,
        filter_r=False,
    )
    reg_all_channels, _ = iss.pipeline.load_and_register_tile(
        data_path=data_path, tile_coors=tile_coords, prefix=reg_prefix, filter_r=False
    )

    if ref_channels is not None:
        if isinstance(ref_channels, int):
            ref_channels = [ref_channels]
        ref_all_channels = ref_all_channels[:, :, ref_channels]
    ref = np.nanmean(ref_all_channels, axis=(2, 3))
    axes[0, 0].imshow(ref)
    axes[0, 0].set_title("Reference")
    if reg_channels is not None:
        if isinstance(reg_channels, int):
            reg_channels = [reg_channels]
        reg_all_channels = reg_all_channels[:, :, reg_channels]
    reg = np.nanmean(reg_all_channels, axis=(2, 3))
    axes[1, 0].imshow(reg)
    axes[1, 0].set_title("Target")

    iax = 1
    if ops["reg_median_filter"]:
        ref = median_filter(ref, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))
        reg = median_filter(reg, footprint=disk(ops["reg_median_filter"]), axes=(0, 1))
        axes[0, iax].imshow(ref)
        axes[0, iax].set_title("Median filtered")
        axes[1, iax].imshow(reg)
        iax += 1

    if binarise_quantile is not None:
        reg = reg > np.quantile(reg, binarise_quantile)
        ref = ref > np.quantile(ref, binarise_quantile)
        axes[0, iax].imshow(ref)
        axes[0, iax].set_title(f"Binarised at {binarise_quantile}")
        axes[1, iax].imshow(reg)

    for ax in fig.axes:
        ax.axis("off")
    fig.tight_layout()
    figure_folder = iss.io.get_processed_path(data_path) / "figures" / "registration"
    figure_folder.mkdir(exist_ok=True)
    fig.savefig(figure_folder / f"debug_reg_to_ref_{reg_prefix}_to_{ref_prefix}.png")


@slurm_it(conda_env="iss-preprocess")
def check_reg_to_ref_correction(
    data_path,
    prefix,
    rois=None,
    roi_dimension_prefix="genes_round_1_1",
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
    processed_path = iss.io.get_processed_path(data_path)
    reg_dir = processed_path / "reg"
    figure_folder = processed_path / "figures" / "registration" / f"{prefix}_to_ref"
    figure_folder.mkdir(exist_ok=True, parents=True)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], ops["use_rois"])]
    figs = {}
    roi_dims[:, 1:] = roi_dims[:, 1:] + 1
    for roi, *ntiles in roi_dims:
        raw = np.zeros([3, *ntiles]) + np.nan
        corrected = np.zeros_like(raw) + np.nan
        best = np.zeros_like(raw) + np.nan
        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                fname = reg_dir / f"tforms_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                if not fname.exists():
                    continue
                try:
                    tform = np.load(fname)["matrix_between_channels"][0]
                    raw[:2, ix, iy] = tform[:2, 2]
                except ValueError:
                    print(f"Could not load {fname}. Skipping.")
                    continue
                tform = np.load(
                    reg_dir / f"tforms_corrected_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )["matrix_between_channels"][0]
                corrected[:2, ix, iy] = tform[:2, 2]
                tform = np.load(
                    reg_dir / f"tforms_best_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )["matrix_between_channels"][0]
                best[:2, ix, iy] = tform[:2, 2]
        fig, axes = plt.subplots(4, 3, figsize=(12, 8))
        fig = iss.vis.plot_matrix_difference(
            raw=raw,
            corrected=corrected,
            col_labels=["Shift x", "Shift y"],
            line_labels=["Raw", "Corrected", "Difference"],
            axes=axes[:3, :],
        )
        for i in range(3):
            # get the clim from the `raw` plot
            vmin, vmax = axes[0, i].get_images()[0].get_clim()
            iss.vis.plot_matrix_with_colorbar(
                best[i].T, axes[3, i], vmin=vmin, vmax=vmax
            )
        axes[3, 0].set_ylabel("Best")
        fig.tight_layout()
        fig.suptitle(f"Registration to reference. {prefix} ROI {roi}")
        fig.savefig(
            figure_folder / f"registration_to_ref_estimation_{prefix}_roi{roi}.png"
        )
        figs[roi] = fig
    return fig


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
    processed_path = iss.io.get_processed_path(data_path)
    reg_dir = processed_path / "reg"
    figure_folder = processed_path / "figures" / "registration" / prefix
    figure_folder.mkdir(exist_ok=True, parents=True)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], ops["use_rois"])]
    roi_dims[:, 1:] = roi_dims[:, 1:] + 1
    figs = {}
    data = np.load(reg_dir / f"tforms_{prefix}_{roi_dims[0,0]}_0_0.npz")
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
                    data = np.load(reg_dir / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz")
                    shifts_within_channels_raw[:, :, :, ix, iy] = data[
                        "shifts_within_channels"
                    ]
                    shifts_between_channels_raw[:, :, ix, iy] = data[
                        "matrix_between_channels"
                    ][:, :2, 2]
                except FileNotFoundError:
                    pass
                data = np.load(
                    reg_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
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
                    fig = iss.vis.plot_matrix_difference(
                        raw=shifts_within_channels_raw[ch, :, dim, :, :],
                        corrected=shifts_within_channels_corrected[ch, :, dim, :, :],
                        col_labels=[f"round {i}" for i in range(nrounds)],
                        range_min=np.ones(nrounds) * 5,
                    )
                    fig.suptitle(f"Dim {dim} shifts. {prefix} ROI {roi} channel {ch}")
                    pdf.savefig(fig)
                    figs[roi] = fig
    return figs


def check_omp_thresholds(
    data_path,
    spot_score_thresholds=(0.05, 0.075, 0.1, 0.125, 0.15, 0.2),
    omp_thresholds=(0.10, 0.125, 0.15, 0.2, 0.25, 0.3),
    rhos=(0.5, 1.0, 2.0, 4.0, 8.0),
    tile_coors=None,
):
    processed_path = iss.io.get_processed_path(data_path)
    ops = iss.io.load_ops(data_path)
    if tile_coors is None:
        tile_coors = ops["ref_tile"]
    stack, bad_pixels = iss.pipeline.load_and_register_sequencing_tile(
        data_path,
        tile_coors,
        filter_r=ops["filter_r"],
        prefix="genes_round",
        suffix=ops["genes_projection"],
        nrounds=ops["genes_rounds"],
        correct_channels=ops["genes_correct_channels"],
        corrected_shifts=ops["corrected_shifts"],
        correct_illumination=True,
    )
    stack = stack[1400:1800, 1400:1800, np.argsort(ops["camera_order"]), :]

    all_gene_spots = []
    omp_stat = np.load(processed_path / "gene_dict.npz", allow_pickle=True)
    for omp_threshold in omp_thresholds:
        g, _, _ = iss.call.run_omp(
            stack,
            omp_stat["gene_dict"],
            tol=omp_threshold,
            weighted=True,
            refit_background=True,
            alpha=ops["omp_alpha"],
            beta_squared=ops["omp_beta_squared"],
            norm_shift=omp_stat["norm_shift"],
            max_comp=ops["omp_max_genes"],
            min_intensity=ops["omp_min_intensity"],
        )
        spot_sign_image = iss.pipeline.load_spot_sign_image(
            data_path, ops["spot_shape_threshold"]
        )
        gene_spots = iss.call.find_gene_spots(
            g,
            spot_sign_image,
            rho=ops["genes_spot_rho"],
            spot_score_threshold=0.05,
        )
        for df, gene in zip(gene_spots, omp_stat["gene_names"]):
            df["gene"] = gene
        all_gene_spots.append(gene_spots)

    im = np.std(stack, axis=(2, 3))
    vmax = np.percentile(im, 99.99)
    neg_max = np.sum(np.sign(spot_sign_image) == -1)
    pos_max = np.sum(np.sign(spot_sign_image) == 1)
    # white background figure
    for rho in rhos:
        plt.figure(figsize=(30, 30), facecolor="w")
        for i in range(len(omp_thresholds)):
            for j in range(len(spot_score_thresholds)):
                spots = pd.concat(all_gene_spots[i])
                spots["spot_score"] = (
                    spots["neg_pixels"] + spots["pos_pixels"] * rho
                ) / (neg_max + pos_max * rho)
                spots = spots[spots["spot_score"] > spot_score_thresholds[j]]
                plt.subplot(
                    len(omp_thresholds),
                    len(spot_score_thresholds),
                    i * len(spot_score_thresholds) + j + 1,
                )
                plt.imshow(im, cmap="bwr", vmax=vmax, vmin=-vmax)
                plt.plot(spots.x, spots.y, "xk", ms=2)
                plt.axis("off")
                plt.title(
                    f"OMP {omp_thresholds[i]:.3f}; spot score {spot_score_thresholds[j]:.3f}"
                )
        plt.tight_layout()
        plt.savefig(
            processed_path / "figures" / f"omp_spot_score_thresholds_rho_{rho}.png",
            dpi=300,
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(im, cmap="inferno", vmax=vmax)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        processed_path / "figures" / "omp_spot_score_thresholds_image.png", dpi=300
    )


def check_segmentation(
    data_path,
    roi,
    prefix,
    reference="genes_round_1_1",
    stitched_stack=None,
    masks=None,
    save_fig=True,
):
    """Check that segmentation is working properly

    Compare masks to the original images

    Args:
        data_path (str): Relative path to data
        roi (int): ROI to process
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        reference (str, optional): Reference prefix. Defaults to "genes_round_1_1".
        stitched_stack (np.ndarray, optional): Stitched stack to use. If None, will
            stitch and align the images. Defaults to None.
        masks (np.ndarray, optional): Masks to use. If None, will load them. Defaults
            to None.
        save_fig (bool, optional): Save the figure. Defaults to True.

    Returns:
        plt.Figure: Figure
    """
    figure_folder = iss.io.get_processed_path(data_path) / "figures" / "segmentation"
    figure_folder.mkdir(exist_ok=True, parents=True)
    # get a tile in the middle of roi
    if stitched_stack is None:
        print(f"stitching {prefix} and aligning to {reference}", flush=True)
        stitched_stack = stitch_registered(
            data_path, ref_prefix=reference, prefix=prefix, roi=roi
        )[..., 0]
    elif stitched_stack.ndim == 3:
        ops = iss.io.load_ops(data_path)
        stitched_stack = stitched_stack[..., ops["cellpose_channels"][0]]

    # normalize the stack and downsample by 2 using block_reduce
    stitched_stack = block_reduce(stitched_stack, (2, 2), np.mean)
    mi, ma = np.percentile(stitched_stack, [0.01, 99.99])
    stitched_stack = np.clip((stitched_stack - mi) / (ma - mi), 0, 1)

    if masks is None:
        print("loading segmentation", flush=True)
        masks = np.load(iss.io.get_processed_path(data_path) / f"masks_{roi}.npy")
    # Make the masks binary and downsample by 2
    masks = (masks > 0)[::2, ::2]

    print("plotting", flush=True)
    half_box = 500
    plot_boxes = stitched_stack.shape[0] > half_box * 5
    plot_boxes = plot_boxes or stitched_stack.shape[1] > half_box * 5

    fig = plt.figure(figsize=(20, 10))
    if plot_boxes:
        main_ax = plt.subplot2grid((2, 5), (0, 0), rowspan=2, colspan=3)
    else:
        main_ax = plt.subplot(111)

    main_ax.imshow(
        stitched_stack,
        cmap="Greys_r",
        vmin=stitched_stack.min(),
        vmax=np.percentile(stitched_stack, 99.9),
    )
    main_ax.contour(masks, colors="orange", levels=[0.5], linewidths=0.2)
    main_ax.axis("off")
    main_ax.set_title(f"Segmentation of {prefix} ROI {roi}")

    if plot_boxes:
        box = np.array([-half_box, half_box])
        # pick 6 boxes in the stiched stack. We want them uniformely distributed
        # but at least 10% from the border
        tile_x_center = (np.array([0.25, 0.75]) * stitched_stack.shape[0]).astype(int)
        tile_y_center = (np.array([0.25, 0.75]) * stitched_stack.shape[1]).astype(int)
        for i, x in enumerate(tile_x_center):
            for j, y in enumerate(tile_y_center):
                ax = plt.subplot2grid((2, 5), (i, j + 3))
                xpart = slice(*np.clip(box + x, 0, stitched_stack.shape[0] - 1))
                ypart = slice(*np.clip(box + y, 0, stitched_stack.shape[1] - 1))
                # add a rectangle to the main plot
                main_ax.add_patch(
                    plt.Rectangle(
                        (ypart.start, xpart.start),
                        ypart.stop - ypart.start,
                        xpart.stop - xpart.start,
                        edgecolor="k",
                        facecolor="none",
                    )
                )
                # gaussian filter to make it look better with skimage
                data = gaussian(stitched_stack[xpart, ypart], 2)
                ax.imshow(data, cmap="Greys_r")
                mask = masks[xpart, ypart]
                if np.any(mask):
                    ax.contour(mask, colors="orange", levels=[0.5], linewidths=0.3)
                ax.axis("off")
    fig.tight_layout()
    if save_fig:
        fig.savefig(figure_folder / f"segmentation_{prefix}_roi{roi}.png", dpi=600)
        print(f"Saved to {figure_folder / f'segmentation_{prefix}_roi{roi}.png'}")
    return fig


@slurm_it(conda_env="iss-preprocess")
def check_tile_reg2ref(
    data_path,
    reg_prefix="barcode_round",
    ref_prefix="genes_round",
    correction="best",
    tile_coords=None,
    reg_channels=None,
    ref_channels=None,
    binarise_quantile=0.7,
    window=None,
):
    """Check the registration to reference for some tiles

    If `tile_coords` is None, will select 10 tiles. If `ops` has a `xx_ref_tiles`
    matching prefix, these will be part of the 10 tiles. The remaining tiles will be
    selected randomly.

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
        correction (str, optional): Corrections to plot. Defaults to 'best'.
        tile_coords (list, optional): List of tile coordinates to process. If None, will
            select 10 tiles. Defaults to None.
        reg_channels (list, optional): List of channels to plot for the registered images.
            If None, will use the average of all channels. Defaults to None.
        ref_channels (list, optional): List of channels to plot for the reference images.
            If None, will use the average of all channels. Defaults to None.
        binarise_quantile (float, optional): Quantile to binarise the images. Defaults to
            0.7.
        window (int, optional): Size of the window to plot around the center of the
            image. Full image if None. Defaults to None.
    """
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / f"{reg_prefix}_to_ref"
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = iss.io.load_ops(data_path)

    # get stack registered between channel and rounds
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=f"{reg_prefix}_1_1")
    if tile_coords is None:
        # check if ops has a ref tile
        if f"{reg_prefix.split('_')[0]}_ref_tiles" in ops:
            tile_coords = ops[f"{reg_prefix.split('_')[0]}_ref_tiles"]
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

    for tile in tile_coords:
        # get the data with default correction for within prefix registration
        ref_all_channels, _ = iss.pipeline.load_and_register_tile(
            data_path=data_path,
            tile_coors=tile,
            prefix=ref_prefix,
            filter_r=False,
        )
        reg_all_channels, _ = iss.pipeline.load_and_register_tile(
            data_path=data_path, tile_coors=tile, prefix=reg_prefix, filter_r=False
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

        if binarise_quantile is not None:
            reg_b = reg > np.quantile(reg, binarise_quantile)
            ref_b = ref > np.quantile(ref, binarise_quantile)
        else:
            reg_b = reg
            ref_b = ref

        print("Loading shifts and angle")
        tname = f"{reg_prefix}_{tile[0]}_{tile[1]}_{tile[2]}"
        fname = processed_path / "reg" / f"tforms_{correction}_to_ref_{tname}.npz"
        assert fname.exists(), f"File {fname} does not exist"
        t_form = np.load(fname)
        angle = t_form["angles"][0]
        shift = t_form["shifts"][0]

        # transform the reg image to match the ref
        reg_t = transform_image(reg, angle=angle, shift=shift)
        reg_bt = transform_image(reg_b, angle=angle, shift=shift)

        # add an rgb overlay
        vmins = [np.percentile(ref, 1), np.percentile(reg_t, 1)]
        vmaxs = [np.percentile(ref, 99.5), np.percentile(reg_t, 99.5)]
        rgb = iss.vis.to_rgb(
            np.stack([ref, reg_t], axis=2),
            colors=([1, 0, 0], [0, 1, 0]),
            vmin=vmins,
            vmax=vmaxs,
        )
        rgb_b = iss.vis.to_rgb(
            np.stack([ref_b, reg_bt], axis=2),
            colors=([1, 0, 0], [0, 1, 0]),
            vmin=[0, 0],
            vmax=[1, 1],
        )

        # Plot it
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        axes[0, 0].imshow(ref, cmap="inferno", vmin=vmins[0], vmax=vmaxs[0])
        axes[0, 0].set_title("Reference")
        axes[0, 1].imshow(reg, cmap="inferno", vmin=vmins[1], vmax=vmaxs[1])
        axes[0, 1].set_title("Target")
        axes[0, 2].imshow(reg_t, cmap="inferno", vmin=vmins[1], vmax=vmaxs[1])
        axes[0, 2].set_title("Transformed")
        axes[0, 3].imshow(rgb)
        axes[0, 3].set_title("Overlay")
        axes[1, 0].imshow(ref_b, cmap="gray")
        axes[1, 0].set_title("Reference")
        axes[1, 1].imshow(reg_b, cmap="gray")
        axes[1, 1].set_title("Target")
        axes[1, 2].imshow(reg_bt, cmap="gray")
        axes[1, 2].set_title("Transformed")
        axes[1, 3].imshow(rgb_b)
        axes[1, 3].set_title("Overlay")

        center = (ref.shape[0] // 2, ref.shape[1] // 2)
        for ax in axes.flatten():
            ax.axis("off")
            if window is not None:
                ax.set_xlim(center[1] - window, center[1] + window)
                ax.set_ylim(center[0] + window, center[0] - window)
        fig.tight_layout()
        fname = f"check_reg2ref_{tname}_{correction}"
        fig.savefig(target_folder / f"{fname}.png")
        plt.close(fig)  # close the figure to avoid memory leak


def check_reg2ref_using_stitched(
    data_path,
    reg_prefix,
    ref_prefix,
    roi,
    stitched_stack_reference,
    stitched_stack_target,
    ref_centers=None,
    trans_centers=None,
):
    """Check the registration to reference done on stitched images

    Args:
        data_path (str): Relative path to data
        reg_prefix (str): Prefix of the registered images
        ref_prefix (str): Prefix of the reference images
        roi (int): ROI to process
        stitched_stack_reference (np.ndarray): Stitched stack of the reference images
        stitched_stack_target (np.ndarray): Stitched stack of the registered images
        ref_centers (np.ndarray): Reference centers to plot. Defaults
            to None.
        trans_centers (np.ndarray): Transformed centers to plot. Defaults

    Returns:
        plt.Figure: Figure
    """

    save_folder = iss.io.get_processed_path(data_path) / "figures" / "registration"
    save_folder /= f"{reg_prefix}_to_{ref_prefix}"
    save_folder.mkdir(parents=True, exist_ok=True)
    save_path = save_folder / f"{reg_prefix}_to_{ref_prefix}_roi_{roi}.png"
    st = np.dstack([stitched_stack_reference, stitched_stack_target])
    rgb = iss.vis.to_rgb(
        st, colors=[(0, 1, 0), (1, 0, 1)], vmax=np.nanpercentile(st, 99, axis=(0, 1))
    )
    del st
    aspect_ratio = rgb.shape[0] / rgb.shape[1]
    fig = plt.figure(figsize=(5, 5 * aspect_ratio))
    ax = fig.add_subplot(111)
    ax.imshow(rgb, zorder=0)
    if ref_centers is not None:
        flat_refc = ref_centers.reshape(-1, 2)
        ax.scatter(flat_refc[:, 1], flat_refc[:, 0], c="blue", s=1, zorder=2)
    if trans_centers is not None:
        flat_trac = trans_centers.reshape(-1, 2)
        ax.scatter(flat_trac[:, 1], flat_trac[:, 0], c="orange", s=2, zorder=3)
    if trans_centers is not None and ref_centers is not None:
        for c, t in zip(flat_refc, flat_trac):
            ax.plot([c[1], t[1]], [c[0], t[0]], "k", zorder=1)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=1200)
    print(f"Saving plot to {save_path}")
    del rgb
    return fig
