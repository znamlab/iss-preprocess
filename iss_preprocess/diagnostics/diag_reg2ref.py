from pathlib import Path

import flexiznam as flz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from image_tools.similarity_transforms import transform_image
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.ndimage import median_filter
from skimage.morphology import disk
from znamutils import slurm_it

from ..diagnostics import _get_some_tiles
from ..io import get_processed_path, get_roi_dimensions, load_ops
from ..pipeline.register import (
    load_and_register_sequencing_tile,
    load_and_register_tile,
)
from ..vis import add_bases_legend, round_to_rgb, to_rgb
from ..vis.diagnostics import plot_round_registration_diagnostics
from ..vis.utils import plot_matrix_difference, plot_matrix_with_colorbar


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
    ops = load_ops(data_path)
    if tile_coords is None:
        tile_coords = ops["ref_tile"]
    naxes = 1
    if ops["reg_median_filter"]:
        naxes += 1
    if binarise_quantile is not None:
        naxes += 1

    fig, axes = plt.subplots(2, naxes, figsize=(6 * naxes, 5))

    ref_all_channels, _ = load_and_register_tile(
        data_path=data_path,
        tile_coors=tile_coords,
        prefix=ref_prefix,
        filter_r=False,
    )
    reg_all_channels, _ = load_and_register_tile(
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
    figure_folder = get_processed_path(data_path) / "figures" / "registration"
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
    processed_path = get_processed_path(data_path)
    reg_dir = processed_path / "reg"
    figure_folder = processed_path / "figures" / "registration" / f"{prefix}_to_ref"
    figure_folder.mkdir(exist_ok=True, parents=True)
    roi_dims = get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.isin(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.isin(roi_dims[:, 0], ops["use_rois"])]
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
        fig = plot_matrix_difference(
            raw=raw,
            corrected=corrected,
            col_labels=["Shift x", "Shift y"],
            line_labels=["Raw", "Corrected", "Difference"],
            axes=axes[:3, :],
        )
        for i in range(3):
            # get the clim from the `raw` plot
            vmin, vmax = axes[0, i].get_images()[0].get_clim()
            plot_matrix_with_colorbar(best[i].T, axes[3, i], vmin=vmin, vmax=vmax)
        axes[3, 0].set_ylabel("Best")
        fig.tight_layout()
        fig.suptitle(f"Registration to reference. {prefix} ROI {roi}")
        fig.savefig(
            figure_folder / f"registration_to_ref_estimation_{prefix}_roi{roi}.png"
        )
        figs[roi] = fig
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
        reg_channels (list, optional): List of channels to plot for the registered
            images. If None, will use the average of all channels. Defaults to None.
        ref_channels (list, optional): List of channels to plot for the reference
            images. If None, will use the average of all channels. Defaults to None.
        binarise_quantile (float, optional): Quantile to binarise the images. Defaults
            to 0.7.
        window (int, optional): Size of the window to plot around the center of the
            image. Full image if None. Defaults to None.
    """
    processed_path = get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration" / f"{reg_prefix}_to_ref"
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = load_ops(data_path)

    # get stack registered between channel and rounds
    roi_dims = get_roi_dimensions(data_path, prefix=f"{reg_prefix}_1_1")
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
        ref_all_channels, _ = load_and_register_tile(
            data_path=data_path,
            tile_coors=tile,
            prefix=ref_prefix,
            filter_r=False,
        )
        reg_all_channels, _ = load_and_register_tile(
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
        rgb = to_rgb(
            np.stack([ref, reg_t], axis=2),
            colors=([1, 0, 0], [0, 1, 0]),
            vmin=vmins,
            vmax=vmaxs,
        )
        rgb_b = to_rgb(
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


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_registration_to_reference(data_path, prefix, ref_prefix, tile_coords=None):
    ops = load_ops(data_path)
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
    processed = get_processed_path(data_path)
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
        ref_stack, _ = load_and_register_tile(
            data_path, tile, ref_prefix, filter_r=False
        )
        # get the tile to register
        reg_stack, _ = load_and_register_tile(
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


def check_rolonies_registration(
    savefname,
    data_path,
    tile_coors,
    n_rolonies,
    prefix="genes_round",
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    vmax=0.5,
    correct_illumination=True,
    corrected_shifts="best",
):
    """Check the registration of rolonies

    Will plot a random selection of rolonies overlaid on the spot sign image circles.

    Args:
        savefname (str): Path to save the figure to
        data_path (str): Path to the data folder
        tile_coors (tuple): Tile coordinates
        n_rolonies (int): Number of rolonies to plot
        prefix (str, optional): Prefix to use. Defaults to "genes_round".
        channel_colors (list, optional): List of colors for each channel. Defaults to
            ([1,0,0],[0,1,0],[1,0,1],[0,1,1]).
        vmax (float, optional): Max value image scale. Defaults to 0.5.
        correct_illumination (bool, optional): Whether to correct for illumination.
            Defaults to True.
        corrected_shifts (str, optional): Which shifts to use. One of `best`, `ransac`,
            `single_tile`, `reference`, Defaults to 'best'.

    """
    ops = load_ops(data_path)
    processed_path = Path(flz.config.PARAMETERS["data_root"]["processed"])
    spot_sign = np.load(processed_path / data_path / "spot_sign_image.npy")
    # cut a line in the middle
    line = spot_sign[spot_sign.shape[0] // 2, :]
    positive_radius = np.diff(np.where(line > 0)[0][[0, -1]]) / 2 + 0.5
    negative_radius = np.diff(np.where(line < -0.1)[0][[0, -1]]) / 2

    spot_folder = processed_path / data_path / "spots"
    spots = pd.read_pickle(
        spot_folder
        / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
    spots.reset_index(inplace=True)
    nrounds = ops[f"{prefix}s"]
    # get stack registered between channel and rounds
    reg_stack, bad_pixels = load_and_register_sequencing_tile(
        data_path,
        filter_r=ops["filter_r"],
        correct_channels=True,
        correct_illumination=correct_illumination,
        corrected_shifts=corrected_shifts,
        tile_coors=tile_coors,
        suffix=ops[f"{prefix.split('_')[0]}_projection"],
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=None,
    )
    extent = np.array([-1, 1]) * 100

    rng = np.random.default_rng(42)
    rolonies = rng.choice(spots.index, n_rolonies, replace=False)
    if n_rolonies == 1:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes = np.array([axes])
    else:
        nrow = int(np.sqrt(n_rolonies))
        ncol = int(np.ceil(n_rolonies / nrow))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3, nrow * 3))
    fig.subplots_adjust(
        top=0.9, wspace=0.01, hspace=0.1, bottom=0.01, left=0.01, right=0.99
    )

    # get codebook
    codebook = pd.read_csv(
        Path(__file__).parent.parent / "call" / ops["codebook"],
        header=None,
        names=["gii", "seq", "gene"],
    )
    base_params = dict(color="white", fontsize=20, fontweight="bold")
    stacks_list = []
    sequences = []
    labels = []
    for i, ax in enumerate(axes.flatten()):
        spot = spots.loc[rolonies[i]]
        center = spot[["x", "y"]].astype(int).values
        wx = np.clip(center[0] + extent, 0, reg_stack.shape[0])
        wy = np.clip(center[1] + extent, 0, reg_stack.shape[1])
        stack_part = reg_stack[slice(*wy), slice(*wx)]
        stacks_list.append(stack_part)
        pc = plt.Circle(center, radius=positive_radius, ec="r", fc="none", lw=2)
        ax.add_artist(pc)
        nc = plt.Circle(center, radius=negative_radius, ec="b", fc="none", lw=2)
        ax.add_artist(nc)
        ax.imshow(
            round_to_rgb(stack_part, 0, None, channel_colors, vmax),
            extent=np.hstack([wx, wy]),
            origin="lower",
        )
        ax.axis("off")
        ax.set_title(f"Rol #{rolonies[i]}. X: {center[0]}, Y: {center[1]}")
        gene = spots.loc[rolonies[i], "gene"]
        sequences.append(codebook.loc[codebook["gene"] == gene, "seq"].values[0])
        txt = ax.text(0.05, 0.9, sequences[i][0], transform=ax.transAxes, **base_params)
        labels.append(txt)

    add_bases_legend(channel_colors=channel_colors)

    def animate(iround):
        for iax, stack in enumerate(stacks_list):
            ax = axes.flatten()[iax]

            ax.images[0].set_data(
                round_to_rgb(stack, iround, None, channel_colors, vmax),
            )
            labels[iax].set_text(sequences[iax][iround])
        fig.suptitle(
            f"Tile {tile_coors}. Corrected shifts: {corrected_shifts}. Round {iround}"
        )

    anim = FuncAnimation(fig, animate, frames=reg_stack.shape[3], interval=200)
    anim.save(savefname, writer=FFMpegWriter(fps=2))
