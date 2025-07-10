import numbers

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from znamutils import slurm_it

from ..call import BASES
from ..call.omp import run_omp
from ..call.spot_shape import find_gene_spots
from ..io import get_processed_path, load_ops
from ..pipeline.register import load_and_register_sequencing_tile
from ..pipeline.sequencing import (
    basecall_tile,
    load_spot_sign_image,
)
from ..vis import (
    add_bases_legend,
    plot_clusters,
    plot_gene_templates,
    plot_spot_called_base,
    round_to_rgb,
)
from ..vis.utils import (
    get_spot_part,
    get_stack_part,
    plot_matrix_with_colorbar,
)
from . import _get_some_tiles


@slurm_it(conda_env="iss-preprocess")
def check_omp_setup(data_path):
    """Plot the OMP setup, including clustering of reference gene spots and
    gene templates, and save them in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "sequencing"
    figure_folder.mkdir(exist_ok=True)
    reference_gene_spots = np.load(
        processed_path / "reference_gene_spots.npz", allow_pickle=True
    )
    omp_stat = np.load(processed_path / "gene_dict.npz", allow_pickle=True)
    nrounds = reference_gene_spots["spot_colors"].shape[0]
    print("Plotting clusters")
    figs = plot_clusters(
        omp_stat["cluster_means"],
        reference_gene_spots["spot_colors"],
        reference_gene_spots["cluster_inds"],
    )
    print("Plotting gene templates")
    figs.append(
        plot_gene_templates(
            omp_stat["gene_dict"],
            omp_stat["gene_names"],
            BASES,
            nrounds=nrounds,
        )
    )
    for fig in figs:
        fig.savefig(figure_folder / f"omp_{fig.get_label()}.png")
    print(f"Saved figures in {figure_folder}")


@slurm_it(conda_env="iss-preprocess")
def check_omp_alpha_thresholds(
    data_path,
    spot_score_thresholds=(0.05, 0.075, 0.1, 0.125, 0.15, 0.2),
    omp_thresholds=(0.10, 0.125, 0.15, 0.2, 0.25, 0.3),
    alphas=(10, 50, 100, 200, 300, 400),
    tile_coors=None,
):
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    if tile_coors is None:
        tile_coors = ops["ref_tile"]
    fig_folder = processed_path / "figures" / "sequencing"
    fig_folder.mkdir(exist_ok=True)
    print("Loading and registering tile")
    stack, bad_pixels = load_and_register_sequencing_tile(
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

    print("Running all alpha/OMP threshold combinations")
    for alpha in alphas:
        temp_gene_spots = []
        for omp_threshold in omp_thresholds:
            g, _, _ = run_omp(
                stack,
                omp_stat["gene_dict"],
                tol=omp_threshold,
                weighted=True,
                refit_background=True,
                alpha=alpha,  # Use the current alpha value
                beta_squared=ops["omp_beta_squared"],
                norm_shift=omp_stat["norm_shift"],
                max_comp=ops["omp_max_genes"],
                min_intensity=ops["omp_min_intensity"],
            )
            spot_sign_image = load_spot_sign_image(
                data_path, ops["spot_shape_threshold"]
            )
            gene_spots, _ = find_gene_spots(
                g,
                spot_sign_image,
                gene_names=omp_stat["gene_names"],
                rho=2,  # Set rho value to 2
                spot_score_threshold=0.05,
            )
            for df, gene in zip(gene_spots, omp_stat["gene_names"]):
                df["gene"] = gene
            temp_gene_spots.append(gene_spots)
        all_gene_spots.append(temp_gene_spots)

    print("Plotting results")
    im = np.std(stack, axis=(2, 3))
    vmax = np.percentile(im, 99.99)
    neg_max = np.sum(np.sign(spot_sign_image) == -1)
    pos_max = np.sum(np.sign(spot_sign_image) == 1)
    # white background figure
    for alpha_index, alpha in enumerate(alphas):
        plt.figure(figsize=(30, 30), facecolor="w")
        for i in range(len(omp_thresholds)):
            for j in range(len(spot_score_thresholds)):
                spots = pd.concat(all_gene_spots[alpha_index][i])
                spots["spot_score"] = (
                    spots["neg_pixels"] + spots["pos_pixels"] * 2
                ) / (neg_max + pos_max * 2)
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
                    f"OMP {omp_thresholds[i]:.3f}; spot score "
                    + f"{spot_score_thresholds[j]:.3f}; alpha {alpha}"
                )
        plt.tight_layout()
        plt.savefig(
            fig_folder / f"omp_spot_score_thresholds_alpha_{alpha}.png",
            dpi=300,
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(im, cmap="inferno", vmax=vmax)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_folder / "omp_spot_score_thresholds_image.png", dpi=300)
    print(f"Saved figures in {fig_folder}")


@slurm_it(conda_env="iss-preprocess")
def check_omp_thresholds(
    data_path,
    spot_score_thresholds=(0.05, 0.075, 0.1, 0.125, 0.15, 0.2),
    omp_thresholds=(0.10, 0.125, 0.15, 0.2, 0.25, 0.3),
    rhos=(0.5, 1.0, 2.0, 4.0, 8.0),
    tile_coors=None,
):
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    if tile_coors is None:
        tile_coors = ops["ref_tile"]
    fig_folder = processed_path / "figures" / "sequencing"
    fig_folder.mkdir(exist_ok=True)

    print("Loading and registering tile")
    stack, bad_pixels = load_and_register_sequencing_tile(
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
    print("Running all OMP thresholds")
    for omp_threshold in omp_thresholds:
        g, _, _ = run_omp(
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
        spot_sign_image = load_spot_sign_image(data_path, ops["spot_shape_threshold"])
        gene_spots, _ = find_gene_spots(
            g,
            spot_sign_image,
            gene_names=omp_stat["gene_names"],
            rho=ops["genes_spot_rho"],
            spot_score_threshold=0.05,
        )
        for df, gene in zip(gene_spots, omp_stat["gene_names"]):
            df["gene"] = gene
        all_gene_spots.append(gene_spots)

    print("Plotting results")
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
                    f"OMP {omp_thresholds[i]:.3f}; spot score "
                    + f"{spot_score_thresholds[j]:.3f}"
                )
        plt.tight_layout()
        plt.savefig(
            fig_folder / f"omp_spot_score_thresholds_rho_{rho}.png",
            dpi=300,
        )

    plt.figure(figsize=(20, 20))
    plt.imshow(im, cmap="inferno", vmax=vmax)
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_folder / "omp_spot_score_thresholds_image.png", dpi=300)
    print(f"Saved figures in {fig_folder}")


@slurm_it(conda_env="iss-preprocess")
def check_barcode_basecall(
    data_path, tile_coords=None, center=None, window=200, show_scores=True, savefig=True
):
    """Check that the basecall is correct

    Plots the basecall for a tile, with the raw data, the basecall, and the scores

    Args:
        path (str): Path to data folder
        tile_coords (list, optional): Tile coordinates to use. Defaults to None.
        center (list, optional): Center of the tile to use. Defaults to None.
        window (int, optional): Half size of the window to show in the figures.
            Defaults to 200.
        savefig (bool, optional): Save the figure. Defaults to True.

    Returns:
        plt.Figure: Figure(s) with the basecall
    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "barcode_round"
    if savefig:
        figure_folder.mkdir(exist_ok=True)

    # get one of the reference tiles
    ops = load_ops(data_path)

    if tile_coords is None:
        tile_coords = _get_some_tiles(
            data_path, prefix="barcode_round_1_1", tile_coords=tile_coords
        )

    if not isinstance(tile_coords[0], numbers.Number):
        # it's a list of tile coordinates
        figs = []
        for tile in tile_coords:
            figs.append(
                check_barcode_basecall(
                    data_path,
                    tile,
                    center=center,
                    window=window,
                    show_scores=show_scores,
                    savefig=savefig,
                )
            )
        return figs

    # we have been called with a single tile coordinate
    stack, spot_sign_image, spots = basecall_tile(
        data_path, tile_coords, save_spots=False
    )

    if center is None:
        # Find the place with the highest density of spots
        x, y = spots["x"].values, spots["y"].values
        # Create a grid of potential disk centers

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

    center = np.array(center)

    lims = np.vstack([center - window, center + window]).astype(int)
    lims = np.clip(lims, 0, np.array([stack.shape[1], stack.shape[0]]) - 1)
    nr = ops["barcode_rounds"]
    stack_part = get_stack_part(stack, lims[:, 0], lims[:, 1])
    valid_spots = get_spot_part(spots, lims[:, 0], lims[:, 1])

    # Do the plot
    ncol = 3 if show_scores else 2
    fig = plt.figure(figsize=(3 * nr, 10))
    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])
    axes = []
    for iround in range(nr):
        rgb_stack = round_to_rgb(
            stack_part, iround, extent=None, channel_colors=channel_colors
        )
        # plot raw fluo
        ax = fig.add_subplot(ncol, nr, iround + 1)
        axes.append(ax)
        ax.imshow(rgb_stack)
        ax.set_title(f"Round {iround}")
        if iround == nr - 1:
            add_bases_legend(channel_colors, ax.transAxes, fontsize=14)

        # plot basecall, a letter per spot
        ax = fig.add_subplot(ncol, nr, nr + iround + 1)
        axes.append(ax)
        spots_in_frame = valid_spots.copy()
        spots_in_frame["x"] -= center[0] - window
        spots_in_frame["y"] -= center[1] - window
        plot_spot_called_base(spots_in_frame, ax, iround)

    if show_scores:
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
            cax, cb = plot_matrix_with_colorbar(empty, ax, cmap=cmap)
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
    if savefig:
        fig.savefig(figure_folder / f"barcode_basecall_example_tile_{tc}.png")
    return fig


@slurm_it(conda_env="iss-preprocess")
def check_barcode_calling(data_path):
    """Plot the barcode cluster scatter plots and cluster means and save them in the
    figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures" / "barcode_round"
    figure_folder.mkdir(exist_ok=True)
    reference_barcode_spots = np.load(
        processed_path / "reference_barcode_spots.npz", allow_pickle=True
    )
    cluster_means = np.load(processed_path / "barcode_cluster_means.npy")
    figs = plot_clusters(
        cluster_means,
        reference_barcode_spots["spot_colors"],
        reference_barcode_spots["cluster_inds"],
    )
    for fig in figs:
        fig.savefig(figure_folder / f"barcode_{fig.get_label()}.png")
