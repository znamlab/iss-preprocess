import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib.ticker import FixedLocator
from skimage.measure import block_reduce
from znamutils import slurm_it

from ..io import (
    get_pixel_size,
    get_processed_path,
    get_roi_dimensions,
    load_metadata,
    load_micromanager_metadata,
    load_ops,
)
from ..pipeline.stitch import stitch_registered, stitch_tiles
from ..vis.vis import to_rgb


def overview_tif_name(chamber, roi, prefix, suffix, downsample_factor):
    """File name of the overview tif saved by `plot_single_overview`.

    Args:
        chamber (str): Chamber folder name, e.g. "chamber_01"
        roi (int): ROI number
        prefix (str): Prefix of acquisition
        suffix (str): Channels, e.g. "channels_0_1_2_3" or "channel_0"
        downsample_factor (int): Downsampling of the overview relative to the tiles

    Returns:
        str: File name, in `figures/round_overviews`
    """
    return (
        f"{chamber}_roi_{roi:02d}_{prefix}_{suffix}"
        + f"_downsample_{downsample_factor}.ome.tif"
    )


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="128G"))
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
    correct_illumination="auto",
    filter_r=False,
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
        downsample_factor (int, optional): Amount to downsample overview, taking the
            maximum of each block of downsample_factor x downsample_factor pixels.
            Defaults to 25
        save_raw (bool, optional): Whether to save the downsampled overview as a tif,
            with no gridlines, named by `overview_tif_name`. Its pixel size (Micro-Manager
            `PixelSizeUm` times downsample_factor) is saved for ImageJ, if the
            acquisition metadata is found. Defaults to False.
        correct_illumination (bool or str, optional): Whether to correct for uneven
            illumination. If "auto", correct only if the acquisition average
            (`averages/{prefix}_max_average.tif`) exists when the function runs.
            Defaults to "auto".
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

    if correct_illumination == "auto":
        # decided here rather than when the job is submitted, as averages are
        # usually created by jobs that run just before this one. Same file as loaded
        # by `stitch_tiles`
        average = (
            get_processed_path(data_path) / "averages" / f"{prefix}_max_average.tif"
        )
        correct_illumination = average.exists()
        if not correct_illumination:
            print(f"Not correcting illumination: {average} not found", flush=True)
    print(f"Illumination correction: {correct_illumination}", flush=True)

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
            filter_r=filter_r,
        )
        if downsample_factor > 1:
            small_stack = block_reduce(
                small_stack, (downsample_factor, downsample_factor), np.max
            )
        if stack is None:
            stack = np.zeros(small_stack.shape + (len(ch),), dtype="uint16")
        # Corrected images are negative where the background is below the black
        # level, and can exceed the uint16 range where the average is dim: clip so
        # that the conversion does not wrap around
        stack[:, :, ic] = np.clip(small_stack, 0, np.iinfo("uint16").max).astype(
            "uint16"
        )

    figure_folder = get_processed_path(data_path) / "figures" / "round_overviews"
    figure_folder.mkdir(parents=True, exist_ok=True)
    mouse_name = Path(data_path).parts[1]
    extracted_chamber = Path(data_path).parts[2]

    if save_raw:
        print("   ... saving raw", flush=True)
        try:
            # one overview pixel is a block of downsample_factor tile pixels
            pixel_size = get_pixel_size(data_path, prefix) * max(downsample_factor, 1)
        except (AssertionError, FileNotFoundError, KeyError) as err:
            warnings.warn(f"Pixel size not saved in the overview tif: {err}")
            scale = {}
        else:
            # ImageJ reads the resolution as pixels per unit
            scale = dict(
                resolution=(1 / pixel_size, 1 / pixel_size), metadata={"unit": "um"}
            )
        tifffile.imwrite(
            figure_folder
            / overview_tif_name(
                extracted_chamber, roi, prefix, suffix, downsample_factor
            ),
            np.moveaxis(stack, -1, 0),
            imagej=True,
            **scale,
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
            print("setting vmax")
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


@slurm_it(conda_env="iss-preprocess")
def plot_overview_images(
    data_path,
    prefix,
    plot_grid=True,
    downsample_factor=25,
    save_raw=True,
    dependency=None,
    group_channels=True,
    vmin=None,
    vmax=None,
    correct_illumination="auto",
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
        vmin (list, optional): vmin for each channel. Default to None
        vmax (list, optional): vmax for each channel. Default to None
        correct_illumination (bool or str, optional): Passed to
            `plot_single_overview`. "auto" corrects if the average exists when each
            job runs. Defaults to "auto".
    """
    roi_dims = get_roi_dimensions(data_path, prefix)
    image_metadata = load_micromanager_metadata(data_path, prefix)
    nchannels = image_metadata["Summary"]["Channels"]
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
                    use_slurm=True,
                    slurm_folder=slurm_folder,
                    scripts_name=scripts_name,
                    dependency_type="afterany",
                    job_dependency=dependency if dependency else None,
                )
            )
    return job_ids


def combine_overview_plots(data_path, prefix, chamber_list, downsample_factor=25):
    """
    Combine all the round overviews into a single figure

    Args:
        data_path (str): path to the data
        prefix (str): prefix for the round overview files, e.g. "genes_round",
            "barcode_round", "DAPI"
        chamber_list (list): list of chambers to include
        downsample_factor (int, optional): downsampling of the overview tifs to use.
            Tifs saved before the factor was in their name are used if there is no
            tif with this factor. Defaults to 25.
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
                    folder = processed_path / "figures" / "round_overviews"
                    chamber_name = f"chamber_{chamber}"
                    acquisition = f"{prefix}_{round}_1"
                    fname = folder / overview_tif_name(
                        chamber_name,
                        roi,
                        acquisition,
                        "channels_0_1_2_3",
                        downsample_factor,
                    )
                    if not fname.exists():
                        # saved before the downsample factor was in the name
                        fname = (
                            folder
                            / f"{chamber_name}_roi_{roi:02d}_{acquisition}"
                            "_channels_0_1_2_3.ome.tif"
                        )
                    img = tifffile.TiffFile(fname).asarray()
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


@slurm_it(conda_env="iss-preprocess")
def check_barcode_mcherry_reg(
    data_path,
    roi,
    barcode_prefix="barcode_round_1_1",
    mcherry_prefix="mCherry_1",
    target=None,
):
    """Check registration of barcode and mCherry to reference on whole stitched image

    Args:
        data_path (str): Path to data
        roi (int): ROI number
        barcode_prefix (str, optional): Prefix for barcode. Defaults to
            "barcode_round_1_1".
        mcherry_prefix (str, optional): Prefix for mCherry. Defaults to "mCherry_1".
        target (str, optional): Path to save the figure to. Defaults to None.

    Returns:
        plt.Figure: Figure instance
    """

    ops = load_ops(data_path)
    ref_prefix = ops["reference_prefix"]
    ref_ch = ops["reg2ref_reference_channels"]
    print("Stitching ref")
    ref = stitch_registered(data_path, prefix=ref_prefix, roi=roi, channels=ref_ch)
    ref = np.nanmean(ref, axis=2)

    bc_chans = ops.get("reg2ref_barcode_channels", [0, 1, 2, 3])
    print("Stitching barcode")
    barcode_round_1 = stitch_registered(
        data_path, prefix=barcode_prefix, roi=roi, channels=bc_chans
    )
    barcode_round_1 = np.nanmax(barcode_round_1, axis=2)
    mch_chan = ops.get("reg2ref_mCherry_channels", [3])
    print("Stitching mCherry")
    mcherry = stitch_registered(
        data_path, prefix=mcherry_prefix, roi=roi, channels=mch_chan
    )
    mcherry = np.nanmean(mcherry, axis=2)

    st = np.dstack([ref, mcherry, barcode_round_1])
    rgb = to_rgb(
        st,
        colors=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        vmin=np.nanpercentile(st, 1, axis=(0, 1)),
        vmax=np.nanpercentile(st, 99.9, axis=(0, 1)),
    )

    aspect_ratio = rgb.shape[1] / rgb.shape[0]
    print(aspect_ratio)
    fig = plt.figure(figsize=(10 * aspect_ratio, 20))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(rgb, interpolation="none")
    ax.set_axis_off()

    fw, fh = ref.shape[:2]
    w = int(fw // 4 * aspect_ratio)
    h = fh // 4
    margin = [fw // 6, fh // 6]
    for ix in range(2):
        for iy in range(2):
            ax = fig.add_subplot(4, 2, ix + 2 * iy + 5)
            xlims = np.array([margin[0], margin[0] + w])
            if ix:
                xlims = fw - xlims[::-1]
            ylims = np.array([margin[1], margin[1] + h])
            if iy:
                ylims = fh - ylims[::-1]
            s_corner = st[ylims[0] : ylims[1], xlims[0] : xlims[1]]
            rgb_part = to_rgb(
                s_corner,
                colors=[(0, 0, 1), (1, 0, 0), (0, 1, 0)],
                vmin=np.nanpercentile(s_corner, 1, axis=(0, 1)),
                vmax=np.nanpercentile(s_corner, 99.9, axis=(0, 1)),
            )
            ax.imshow(rgb_part, interpolation="none")
            ax.set_axis_off()
    fig.tight_layout()
    if target is not None:
        fig.savefig(target, dpi=300)
    return fig
