"""Functions accessory to the pipeline.

This module contains functions that are useful for checking that the pipeline works
but that are not required per se to run it."""


from warnings import warn
import matplotlib.pyplot as plt
from znamutils import slurm_it
from ..vis.diagnostics import plot_correction_images, plot_tilestats_distributions, plot_spot_sign_image
from ..vis.utils import plot_matrix_with_colorbar
from ..io import load_ops, load_stack, get_roi_dimensions, get_processed_path, get_zprofile


import numpy as np


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
    ops = load_ops(data_path)
    # get stack registered between channel and rounds
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    if tile_coords is None:
        # check if ops has a ref tile
        ref_ops_name = f"{prefix.split('_')[0]}_ref_tiles"
        tile_coords = ops.get(ref_ops_name, [])
        if tile_coords is None:
            warn(f'{ref_ops_name} is not defined in ops, analysis is likely to fail.')
            tile_coords = []
        nrandom = 10 - len(tile_coords)
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


def check_spot_sign_image(data_path):
    """Plot the average spot sign image and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    spot_image = np.load(processed_path / "spot_sign_image.npy")
    plot_spot_sign_image(spot_image)
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
    processed_path = get_processed_path(data_path)
    average_dir = processed_path / "averages"
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    correction_images = dict()
    distributions = dict()

    for fname in average_dir.glob("*average.tif"):
        correction_images[fname.name.replace("_average.tif", "")] = load_stack(fname)
    for fname in average_dir.glob("*_tilestats.npy"):
        distributions[fname.name.replace("_tilestats.npy", "")] = np.load(fname)
    if verbose:
        print(
            f"Found {len(correction_images)} averages"
            + f" and {len(distributions)} tilestats"
        )

    plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
    if plot_tilestats:
        plot_tilestats_distributions(
            data_path, distributions, grand_averages, figure_folder
        )


def plot_zfocus(data_path, prefix, rois=None, verbose=True):
    """Plot z-profiles and z-focus for a given prefix

    Args:
        data_path (str): Relative path to data folder
        prefix (str): Prefix of the images to load
        rois (int | list, optional): List of rois to process. If None, will process all
            rois. Defaults to None.
        verbose (bool, optional): Print info about progress. Defaults to True
    """
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    if rois is None:
        rois = roi_dims[:, 0]
    elif isinstance(rois, int):
        rois = [rois]

    # collect all data
    for roi in rois:
        if verbose:
            print(f"Plotting ROI {roi}")
        nx, ny = roi_dims[roi_dims[:, 0] == roi, 1:][0]
        zfocus = np.zeros((nx, ny, 4))
        zprofiles = None
        for tile_x in range(nx):
            for tile_y in range(ny):
                zprof = get_zprofile(data_path, prefix, (roi, tile_x, tile_y))
                # use std proj, max was too noisy
                zprof = zprof["std"]
                if zprofiles is None:
                    zprofiles = np.zeros((nx, ny, *zprof.shape))
                zprofiles[tile_x, tile_y, :] = zprof
                zfocus[tile_x, tile_y, :] = np.argmax(zprof, axis=1)

        # find out of focus tiles
        nz = zprofiles.shape[-1]
        correct = (zfocus > 4) & (zfocus < nz - 4)

        # normalise for plots
        channel_max = np.max(zprofiles, axis=(0, 1, 3))
        channel_min = np.min(zprofiles, axis=(0, 1, 3))
        zprofiles = (zprofiles - channel_min[None,None,:,None]) / (channel_max - channel_min)[None,None,:,None]
        center = nz // 2
        aspect = nx / ny
        fig, axs = plt.subplots(nx, ny, figsize=(12, 12 * aspect), sharex=True, sharey=True)
        for i in range(nx):
            for j in range(ny):
                axs[i, j].plot(np.arange(nz)-center, zprofiles[i, j].T)
                if not np.all(correct[i, j]):
                    # make the border of the axis red
                    for spine in axs[i, j].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(2)
        fig.suptitle(f"Z-profiles for {prefix} in ROI {roi}")
        fig.tight_layout()

        target_folder = get_processed_path(data_path) / "figures" / prefix
        target_folder.mkdir(exist_ok=True, parents=True)
        fig.savefig(target_folder / f"zprofile_roi{roi}.png")
        plt.close(fig)

        # Plot a z focus matrix per channel
        fig, axs = plt.subplots(2, 2, figsize=(10, 10 * aspect))
        for i_chan, ax in enumerate(axs.flatten()):
            plot_matrix_with_colorbar(zfocus[..., i_chan] - center, ax, cmap="coolwarm", vmin=-center, vmax=center)
            ax.set_title(f"Channel {i_chan}")
            # Plot a square around each tile which is black if the tile is in focus, red
            # otherwise and whose width is proportional to amplitude of the z-profile
            for i in range(nx):
                for j in range(ny):
                    if correct[i, j, i_chan]:
                        color = "black"
                    else:
                        color = "red"
                    ampl = np.max(zprofiles[i, j, :, i_chan]) - np.min(zprofiles[i, j, :, i_chan])
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor=color, linewidth=0.5 + 2 * ampl, facecolor="White",
                                        alpha=max(0.5-ampl,0))
                    ax.add_patch(rect)

        fig.suptitle(f"Z-focus for {prefix} in ROI {roi}")
        fig.tight_layout()
        fig.savefig(target_folder / f"zfocus_roi{roi}.png")
        plt.close(fig)
