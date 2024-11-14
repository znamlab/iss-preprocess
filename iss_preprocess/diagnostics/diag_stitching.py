from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.ticker import FixedLocator
from skimage.measure import block_reduce
from znamutils import slurm_it

from ..io import (
    get_processed_path,
    get_roi_dimensions,
    load_metadata,
    load_micromanager_metadata,
    load_ops,
)
from ..pipeline.stitch import stitch_tiles
from ..vis.vis import to_rgb


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="64G"))
def plot_single_overview(
    data_path,
    prefix,
    roi,
    ch,
    nx=None,
    ny=None,
    plot_grid=True,
    downsample_factor=25,
    save_raw=False,
    correct_illumination=True,
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    vmin=None,
    vmax=None,
):
    """Plot a single channel overview image.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition
        roi (int): ROI number
        ch (int or list): Channel number
        nx (int, optional): Number of tiles in x. If None will read from roi_dimensions
        ny (int, optional): Number of tiles in y. If None will read from roi_dimensions
        plot_axis (bool, optional): Whether to plot gridlines at tile boundaries.
            Defaults to True.
        downsample_factor (int, optional): Amount to downsample overview. Defaults to 25
        save_raw (bool, optional): Whether to save a full size tif with no gridlines.
            Defaults to False.
        correct_illumination (bool, optional): Whether to correct for uneven
            illumination. Defaults to True.
        channel_colors (list, optional): List of colors for each channel. Use only if
            ch is a list. Defaults to ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]).
        vmin (float, optional): Minimum value for each channel. Defaults to None.
        vmax (float, optional): Maximum value for each channel. Defaults to None.

    Returns:
        fig: Figure object
    """
    if nx is None or ny is None:
        roi_dims = get_roi_dimensions(data_path, prefix=prefix)
        for roi_dim in roi_dims:
            if roi_dim[0] == roi:
                nx = roi_dim[1] + 1
                ny = roi_dim[2] + 1
                break

    fig = plt.figure()
    fig.clear()

    single_channel = True if isinstance(ch, int) else False

    if single_channel:
        ch = [ch]
        suffix = f"channel_{ch[0]}"
    else:
        suffix = f"channels_{'_'.join([str(c) for c in ch])}"

    stack = None
    print(f"Doing roi {roi}")
    for ic, c in enumerate(ch):
        print(f"    Channel {c}")
        print("   ... stitching", flush=True)
        small_stack = stitch_tiles(
            data_path,
            prefix=prefix,
            roi=roi,
            suffix="max",
            ich=c,
            correct_illumination=correct_illumination,
            register_channels=False,  # so that it can run before reg
            allow_quick_estimate=True,
        )
        if downsample_factor > 1:
            small_stack = block_reduce(
                small_stack, (downsample_factor, downsample_factor), np.max
            )
        if stack is None:
            stack = np.zeros(small_stack.shape + (len(ch),), dtype="uint16")
        stack[:, :, ic] = small_stack.astype("uint16")

    figure_folder = get_processed_path(data_path) / "figures" / "round_overviews"
    figure_folder.mkdir(parents=True, exist_ok=True)
    mouse_name = Path(data_path).parts[1]
    extracted_chamber = Path(data_path).parts[2]

    if save_raw:
        print("   ... saving raw", flush=True)
        tifffile.imwrite(
            figure_folder
            / f"{extracted_chamber}_roi_{roi:02d}_{prefix}_{suffix}.ome.tif",
            np.moveaxis(stack, -1, 0),
            imagej=True,
        )
    print("   ... plotting", flush=True)

    if single_channel:
        if vmax is None:
            vmax = np.percentile(stack, 99.9)
        plt.imshow(stack, vmax=vmax, vmin=vmin)
    else:
        if vmax is None:
            nch = stack.shape[2]
            vmax = [np.percentile(stack[:, :, ic], 99.9) for ic in range(nch)]
            rgb = to_rgb(stack, channel_colors, vmax=vmax, vmin=vmin)
        else:
            rgb = to_rgb(stack, channel_colors, vmax=vmax, vmin=vmin)
        plt.imshow(rgb)
    ax = plt.gca()
    ax.set_aspect("equal")
    if plot_grid:
        plt.title(
            f"{mouse_name} {extracted_chamber}, ROI: {roi}, {prefix}, Channel: {ch}"
        )
        dim = np.array(stack.shape)[1::-1]
        tile_size = dim / np.array([nx, ny])
        for ix, x in enumerate("xy"):
            # Add gridlines at approximate tile boundaries
            getattr(ax, f"set_{x}lim")(0, dim[ix])
            tcks = np.arange(0, dim[ix], tile_size[ix]) + (tile_size[ix] / 2)
            getattr(ax, f"set_{x}ticks")(tcks)
            minor_locator = FixedLocator(np.arange(0, dim[ix], tile_size[ix]))
            getattr(ax, f"{x}axis").set_minor_locator(minor_locator)

        # Adjust tick labels to display between the ticks
        ops = load_ops(data_path)
        if ops["x_tile_direction"] == "left_to_right":
            ax.set_xticklabels(np.arange(0, len(ax.get_xticks())), rotation=90)
        else:
            ax.set_xticklabels(np.arange(0, len(ax.get_xticks()))[::-1], rotation=90)
        if ops["y_tile_direction"] == "top_to_bottom":
            ax.set_yticklabels(np.arange(0, len(ax.get_yticks())))
        else:
            ax.set_yticklabels(np.arange(0, len(ax.get_yticks()))[::-1])

        ax.grid(which="minor", color="lightgrey")
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=True,
            labelbottom=True,
        )
        ax.invert_yaxis()
    else:
        ax.axis("off")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    print("   ... saving", flush=True)
    plt.savefig(
        figure_folder / f"{extracted_chamber}_roi_{roi:02d}_{prefix}_{suffix}.png",
        dpi=300,
    )
    print("   ... done", flush=True)
    return fig


def plot_overview_images(
    data_path,
    prefix,
    plot_grid=True,
    downsample_factor=25,
    save_raw=True,
    dependency=None,
    group_channels=True,
    use_slurm=True,
    vmin=None,
    vmax=None,
):
    """Plot individual channel overview images.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition
        plot_axis (bool, optional): Whether to plot gridlines at tile boundaries.
            Defaults to True.
        downsample_factor (int, optional): Amount to downsample overview. Defaults to 25
        save_raw (bool, optional): Whether to save a tif with no gridlines. Defaults to
            True.
        dependency (str, optional): Dependency for the generates slurm scripts
        group_channels (bool, optional): Whether to group channels together. Defaults to
            True.
        use_slurm (bool, optional): Whether to use slurm to run the jobs. Defaults to
            True.
        vmin (list, optional): vmin for each channel. Default to None
        vmax (list, optional): vmax for each channel. Default to None
    """
    processed_path = get_processed_path(data_path)
    roi_dims = get_roi_dimensions(data_path, prefix)
    image_metadata = load_micromanager_metadata(data_path, prefix)
    nchannels = image_metadata["Summary"]["Channels"]
    # Check if average image exists for illumination correction
    correct_illumination = (
        processed_path / "averages" / f"{prefix}_average.tif"
    ).exists()
    job_ids = []
    for roi_dim in roi_dims:
        roi = roi_dim[0]
        if group_channels:
            channels = [list(range(nchannels))]
        else:
            channels = range(nchannels)

        for ch in channels:
            if isinstance(ch, list):
                ch_name = "_".join([str(c) for c in ch])
                scripts_name = f"plot_overview_{prefix}_{roi}_channels_{ch_name}"
            else:
                scripts_name = f"plot_overview_{prefix}_{roi}_channel_{ch}"
            slurm_folder = Path.home() / "slurm_logs" / data_path / "plot_overview/"
            slurm_folder.mkdir(parents=True, exist_ok=True)
            job_ids.append(
                plot_single_overview(
                    data_path=data_path,
                    prefix=prefix,
                    roi=roi,
                    ch=ch,
                    nx=roi_dim[1] + 1,
                    ny=roi_dim[2] + 1,
                    plot_grid=plot_grid,
                    downsample_factor=downsample_factor,
                    save_raw=save_raw,
                    correct_illumination=correct_illumination,
                    vmin=vmin,
                    vmax=vmax,
                    use_slurm=use_slurm,
                    slurm_folder=slurm_folder,
                    scripts_name=scripts_name,
                    dependency_type="afterany",
                    job_dependency=dependency if dependency else None,
                )
            )
    return job_ids


def combine_overview_plots(data_path, prefix, chamber_list):
    """
    Combine all the round overviews into a single figure

    Args:
        data_path (str): path to the data
        prefix (str): prefix for the round overview files, e.g. "genes_round",
            "barcode_round", "DAPI"
        chamber_list (list): list of chambers to include
    """
    processed_path = get_processed_path(data_path)
    metadata = load_metadata(data_path)
    if prefix in ("genes_round", "barcode_round"):
        rounds = metadata[f"{prefix}s"]
    else:
        rounds = 1
    for round in range(1, rounds + 1):
        fig, axs = plt.subplots(8, 5, figsize=(10, 12), dpi=500)
        chamber_count = 0
        for chamber in chamber_list:
            processed_path = get_processed_path(data_path).parent / f"chamber_{chamber}"
            roi_dims = get_roi_dimensions(data_path, f"{prefix}_1_1")
            num_rois = len(roi_dims)
            for roi in range(1, num_rois + 1):
                # Initialize empty image
                row_index = chamber_count * 2 + (roi - 1) // 5
                column_index = (roi - 1) % 5
                empty_img = np.zeros((1000, 1200))
                axs[row_index, column_index].set_title(
                    f"Chamber {chamber} ROI {roi}",
                    loc="center",
                    backgroundcolor="white",
                    fontsize=6,
                    y=0.05,
                )
                axs[row_index, column_index].imshow(empty_img, cmap="gray", vmax=1500)
                axs[row_index, column_index].axis("off")
                try:
                    roi_str = str(roi).zfill(2)
                    file_end = "1_channels_0_1_2_3.ome.tif"
                    img = tifffile.TiffFile(
                        processed_path
                        / "figures"
                        / "round_overviews"
                        / f"chamber_{chamber}_roi_{roi_str}_{prefix}_{round}_{file_end}"
                    ).asarray()
                    img = np.rot90(img, k=1, axes=(1, 2))
                    img = np.moveaxis(img, 0, 2)
                    colors = ([0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0])
                    vmax_defaults = (1000, 1000, 1000, 1000)
                    # Adjust vmax based on prefix
                    vmax_mapping = {
                        "barcode_round": (2500, 1000, 1000, 2000),
                        "genes_round": (2000, 2000, 2000, 2000),
                        "DAPI": (2000, 2000, 2000, 2000),
                        "mCherry": (3000, 2000, 800, 800),
                    }
                    vmax = vmax_mapping.get(prefix, vmax_defaults)
                    # Convert to RGB using the determined colors and vmax
                    rgb = to_rgb(img, colors=colors, vmax=vmax)
                    axs[row_index, column_index].imshow(rgb)
                    axs[row_index, column_index].axis("off")
                except FileNotFoundError:
                    print(f"Chamber {chamber} roi {roi} round {round} not found")
            chamber_count += 1
        plt.tight_layout()
        plt.savefig(processed_path / "figures" / f"{prefix}_{round}.png", dpi=500)
        plt.close()
