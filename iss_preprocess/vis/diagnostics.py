from pathlib import Path
import warnings
import cv2
import flexiznam as flz
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from natsort import natsorted
from znamutils import slurm_it
import iss_preprocess as iss
from iss_preprocess.pipeline import ara_registration as ara_registration
from iss_preprocess.pipeline import sequencing
from iss_preprocess.vis import add_bases_legend, round_to_rgb

from ..io import load_ops


def plot_correction_images(
    correction_images, grand_averages, figure_folder, verbose=True
):
    """Plot average illumination correction images.

    Args:
        correction_images (dict): Dictionary containing image stacks.
        grand_averages (list): List of grand averages to plot.
        figure_folder (pathlib.Path): Path where to save the figures.
        verbose (bool): Print info about progress. Defaults to True

    """
    fig = plt.figure(figsize=(10, 10))
    for prefix in grand_averages:
        if verbose:
            print(f"Doing {prefix}")
        # first plot
        fig.clear()
        if verbose:
            print("    Grand average, all channels")
        fig.suptitle(prefix)
        grand_av = correction_images[prefix]
        ax_ids = np.unravel_index(range(4), (2, 2))
        axes = [
            plt.subplot2grid(shape=(4, 3), loc=(i, j), fig=fig) for i, j in zip(*ax_ids)
        ]
        titles = [f"Channel {i}" for i in range(4)]
        _plot_channels_intensity(axes, grand_av, chan_names=titles)
        if verbose:
            print("    Grand average, inter channel comparison")
        axes = [
            plt.subplot2grid(shape=(4, 3), loc=(i + 2, j), fig=fig)
            for i, j in zip(*ax_ids)
        ]
        titles = [f"Ch{i} - Ch0" for i in range(4)]
        _plot_channels_intensity(axes, grand_av, subtract_chan=0, chan_names=titles)
        if verbose:
            print("    Collecting single rounds")
        all_ch0 = {}
        for k in correction_images:
            if (k.startswith(prefix)) and (len(k) > len(prefix)):
                all_ch0[k] = correction_images[k][:, :, 0]
        av_names = natsorted(all_ch0.keys())
        all_ch0 = np.dstack([grand_av[:, :, 0]] + [all_ch0[k] for k in av_names])
        av_names = ["Grand average"] + [
            name.replace(prefix + "_", "") for name in av_names
        ]
        if verbose:
            print("    Single rounds, channel 0")
        axes = [
            plt.subplot2grid(shape=(len(av_names) + 1, 6), loc=(i, 4), fig=fig)
            for i in range(len(av_names))
        ]
        _plot_channels_intensity(axes, all_ch0, chan_names=av_names)
        axes = [
            plt.subplot2grid(shape=(len(av_names) + 1, 6), loc=(i, 5), fig=fig)
            for i in range(len(av_names))
        ]
        if verbose:
            print("    Single rounds, compare to grand average")
        _plot_channels_intensity(axes, all_ch0, subtract_chan=0)
        plt.subplots_adjust(wspace=0.5)
        fig.savefig(figure_folder / f"average_for_correction_{prefix}.png", dpi=600)

    if verbose:
        print("Doing remaining prefix")
    other_prefix = []
    for prefix in correction_images:
        if any([prefix.startswith(grandav) for grandav in grand_averages]):
            continue
        other_prefix.append(prefix)
    fig.clear()
    fig.set_size_inches(10, 2 * len(other_prefix) + 0.5)
    for ip, prefix in enumerate(other_prefix):
        axes = [
            fig.add_subplot(len(other_prefix) * 2, 4, ip * 8 + i + 1) for i in range(4)
        ]
        _plot_channels_intensity(
            axes, correction_images[prefix], chan_names=[prefix, "", "", ""]
        )
        axes = [
            fig.add_subplot(len(other_prefix) * 2, 4, ip * 8 + i + 5) for i in range(4)
        ]
        sub_images = correction_images[prefix] - correction_images["genes_round"]
        sub_images = np.dstack([sub_images, np.zeros_like(sub_images[:, :, 0])])
        _plot_channels_intensity(axes, sub_images, subtract_chan=4)
    plt.subplots_adjust(wspace=0.2)
    fig.savefig(figure_folder / "average_for_correction_other_prefix.png", dpi=600)


def _plot_channels_intensity(
    axes, correction_image, subtract_chan=None, chan_names=None
):
    """Simple subfunction to plot image with or without subtracting a reference

    This will plot the first len(axes) channels of correction image
    Args:
        axes (list): List of plt.Axes to plot. Must be same lenght as number of channel
        correction_image (np.array): A X x Y x Nch image to plot
        subtract_chan (int, optional): Channel to subtract before displaying. Defaults
            to None, no subtraction
        chan_names (list, optional): list of channel names to label axes
    """
    assert len(axes) <= correction_image.shape[-1]
    for i, ax in enumerate(axes):
        if subtract_chan is None:
            img = ax.imshow(correction_image[:, :, i], vmin=0, vmax=1)
        else:
            subtract_chan = int(subtract_chan)
            img = ax.imshow(
                correction_image[:, :, i] - correction_image[:, :, subtract_chan],
                cmap="RdBu_r",
                vmin=-0.1,
                vmax=0.1,
            )
        if chan_names is not None:
            ax.set_ylabel(chan_names[i])
        plt.colorbar(img, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_affine_debug_images(debug_info, fig=None):
    """Plot debug images for affine registration

    It will plot the correlation, shifts, affine predictions and residuals for each
    channel.

    Args:
        debug_info (dict): Dictionary containing debug information
        fig (plt.Figure, optional): Figure to plot into. Defaults to None, will create
            a new figure.

    Returns:
        plt.Figure: Figure instance
    """

    if fig is None:
        fig = plt.figure(figsize=(2 * 7, 1.5 * len(debug_info)))
    nchans = len(debug_info)
    axes = fig.subplots(nchans, 7)
    labels = [
        "Correlation",
        "X Shift",
        "Y Shift",
        "Affine X",
        "Affine Y",
        "Residual X",
        "Residual Y",
    ]
    for il, lab in enumerate(labels):
        axes[0, il].set_title(lab)

    for i, ch in enumerate(debug_info.keys()):
        axes[i, 0].set_ylabel(f"Channel {ch}")

        db = debug_info[ch]
        nb = db["nblocks"]
        co = db["corr"].reshape(nb[:-1])
        ce = db["centers"].reshape(nb)
        s = db["shifts"].reshape(nb)

        plot_matrix_with_colorbar(co, axes[i, 0])
        plot_matrix_with_colorbar(s[..., 0], axes[i, 1], vmin=-10, vmax=10, cmap="bwr")
        plot_matrix_with_colorbar(s[..., 1], axes[i, 2], vmin=-10, vmax=10, cmap="bwr")
        aff_x = db["fit_x"].predict(db["centers"]).reshape(nb[:-1]) - ce[..., 0]
        aff_y = db["fit_y"].predict(db["centers"]).reshape(nb[:-1]) - ce[..., 1]
        plot_matrix_with_colorbar(aff_x, axes[i, 3], vmin=-10, vmax=10, cmap="bwr")
        plot_matrix_with_colorbar(aff_y, axes[i, 4], vmin=-10, vmax=10, cmap="bwr")
        plot_matrix_with_colorbar(
            aff_x - s[..., 0], axes[i, 5], cmap="bwr", vmin=-5, vmax=5
        )
        plot_matrix_with_colorbar(
            aff_y - s[..., 1], axes[i, 6], cmap="bwr", vmin=-5, vmax=5
        )

    for x in axes.flatten():
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    return fig


def adjacent_tiles_registration(
    data_path,
    prefix,
    roi,
    shifts,
    raw_shifts,
    xcorr_max,
    max_shift=20,
    min_corrcoef=0.5,
    max_delta_shift=30,
):
    """Save figure of tile registration for within acquisition stitching

    see pipeline.stitch.register_within_acquisition for usage.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of acquisition
        roi (int): ROI number
        shifts (np.array): (tilex x tiley x 4) vector of shifts per tile
        raw_shifts (np.array): (tilex x tiley x 4) vector of raw shifts per tile
        xcorr_max (np.array): (tilex x tiley x 2) vector of max correlation per tile
        max_shift (int): Maximum shift to plot, in pixels (default 20)
        min_corrcoef (float): Minimum correlation coefficient to plot

    Returns:
        plt.Figure: Figure instance
    """
    fig, axes = plt.subplots(4, 5)
    fig.set_size_inches(7, 5)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignore nan median
    for idir, direction in enumerate(["right", "down"]):
        sl = [0, 1] if idir == 0 else [2, 3]
        rsh = raw_shifts[..., sl]
        sh = shifts[..., sl]
        xc = xcorr_max[..., idir]

        tmp = rsh.copy()
        bad = xc < min_corrcoef
        bad2plot = bad.astype(float)
        tmp[bad] = np.nan

        if idir:
            med = np.nanmedian(tmp, axis=0)
            med = med[None, :]
        else:
            med = np.nanmedian(tmp, axis=1)
            med = med[:, None]
        delta_shift = np.linalg.norm(rsh - med, axis=2)
        # replace shifts that are either low corr or too far from median
        bad = bad | (delta_shift > max_delta_shift)
        bad2plot += bad

        for i, x in enumerate("xy"):
            ax = axes[idir * 2 + i, 0]
            ax.set_ylabel(f"{direction}\nshift - {x}")
            m = np.nanmedian(rsh[..., i])
            iss.vis.plot_matrix_with_colorbar(
                rsh[..., i].T, ax, vmin=m - max_shift, vmax=m + max_shift
            )
            if idir == 0 and i == 0:
                ax.set_title(f"Raw")
            ax = axes[idir * 2 + i, 1]
            iss.vis.plot_matrix_with_colorbar(
                sh[..., i].T, ax, vmin=m - max_shift, vmax=m + max_shift
            )
            if idir == 0 and i == 0:
                ax.set_title(f"Corrected")
            ax = axes[idir * 2 + i, 2]
            iss.vis.plot_matrix_with_colorbar(
                delta_shift.T, ax, vmin=0, vmax=max_delta_shift * 1.1
            )
            if idir == 0 and i == 0:
                ax.set_title(f"Delta")
            ax = axes[idir * 2 + i, 3]
            if idir == 0 and i == 0:
                ax.set_title(f"Bad tiles")
            iss.vis.plot_matrix_with_colorbar(bad2plot.T, ax, vmin=0, vmax=2)
        axes[idir * 2, 4].set_title(f"Max corr")
        iss.vis.plot_matrix_with_colorbar(
            xc.T, axes[idir * 2, 4], vmin=0, vmax=1, cmap="coolwarm"
        )
        axes[idir * 2 + 1, 4].axis("off")
    # put back runtime warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    fig.tight_layout()
    fig_file = (
        iss.io.get_processed_path(data_path)
        / "figures"
        / "registration"
        / prefix
        / f"adjacent_tile_reg_{prefix}_roi{roi}.png"
    )
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_file, dpi=300)
    print(f"Saving {fig_file}")
    return fig


def plot_ara_registration(data_path, roi, reference_prefix="genes_round_1_1"):
    """Overlay reference image to ARA borders

    Args:
        data_path (str): Relative path to data
        roi (int): ROI number to plot
        reference_prefix (str, optional): Image to use as reference. Defaults to
            "genes_round_1_1".

    Raises:
        ImportError: If `cricksaw_analysis` is not installed

    Returns:
        plt.Figure: Reference to the figure created.
    """
    try:
        from cricksaw_analysis import atlas_utils
    except ImportError:
        raise ImportError("`plot_registration requires `cricksaw_analysis")
    area_ids = ara_registration.make_area_image(
        data_path=data_path, roi=roi, atlas_size=10, full_scale=False
    )
    reg_metadata = ara_registration.load_registration_reference_metadata(
        data_path, roi=roi
    )

    ops = load_ops(data_path)
    stitched_stack = iss.pipeline.stitch_registered(
        data_path,
        reference_prefix,
        roi=roi,
        channels=ops["ref_ch"],
    ).astype(np.single)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    new_shape = reg_metadata["new_shape"]
    downsampled = cv2.resize(stitched_stack, (new_shape[1], new_shape[0]))
    ax.imshow(
        downsampled,
        extent=(0, new_shape[1], 0, new_shape[0]),
        vmax=np.quantile(downsampled, 0.99),
        vmin=0,
        origin="lower",
        cmap="gray",
    )
    atlas_utils.plot_borders_and_areas(
        ax,
        area_ids,
        border_kwargs=dict(colors="purple", alpha=0.6, linewidths=0.1),
    )
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot_tilestats_distributions(
    data_path, distributions, grand_averages, figure_folder
):
    """Plot histogram of pixel values.

    Args:
        data_path (str): Relative path to data
        distributions (dict): Dictionary containing tilestats distributions of pixel
            values per image.
        grand_averages (list): List of grand averages to plot.
        figure_folder (pathlib.Path): Path where to save the figures.
        camera_order (list): Order list of camera as in ops['camera_order']
    """
    ops = iss.io.load_ops(data_path)
    camera_order = ops["camera_order"]
    distri = distributions.copy()
    fig = plt.figure(figsize=(10, 20), facecolor="white")
    colors = ["blue", "green", "red", "purple"]
    # TODO: adapt to varying number of rounds
    nrounds = [np.sum([k.startswith(p) for k in distri]) for p in grand_averages]
    for ip, prefix in enumerate(grand_averages):
        grand_data = distri.pop(prefix)
        ax = fig.add_subplot(np.max(nrounds), 2, 1 + ip)
        ax.set_title(prefix)
        for c, i in enumerate(np.argsort(camera_order)):
            ax.plot(
                grand_data[:, i].cumsum() / np.sum(grand_data[:, i]),
                label=f"Channel {c}",
                color=colors[c],
            )
        ax.set_ylabel("All rounds")

        single_rounds = natsorted([k for k in distri if k.startswith(prefix)])
        for ir, round_name in enumerate(single_rounds):
            ax.axvline(ops["average_clip_value"], color="black")
            ax = fig.add_subplot(nrounds[ip], 2, ir * 2 + ip + 3, sharex=fig.axes[0])
            ax.set_ylabel(f"{round_name.split('_')[-2]} - all")
            data = distri.pop(round_name)
            for c, i in enumerate(np.argsort(camera_order)):
                ax.plot(
                    (
                        (data[:, i] / np.sum(data[:, i]))
                        - (grand_data[:, i] / np.sum(grand_data[:, i]))
                    ).cumsum(),
                    label=f"Channel {c}",
                    color=colors[c],
                )
            ax.set_ylim(-0.4, 0.4)

    for ax in fig.axes:
        ax.axvline(2**12, color="k", zorder=-10)
        for c, i in enumerate(np.argsort(camera_order)):
            ax.axvline(ops["black_level"][i], color=colors[c], zorder=-10)
        ax.set_xlim(np.min(ops["black_level"]) - 2, 2**12)
        ax.semilogx()
    ax.legend()
    fig.savefig(figure_folder / "pixel_value_distributions.png", dpi=600)


def plot_matrix_difference(
    raw,
    corrected,
    col_labels=None,
    line_labels=("Raw", "Corrected", "Difference"),
    range_min=(5, 5, 0.1),
    range_max=None,
    axes=None,
):
    """Plot the raw, corrected matrices and their difference

    Args:
        raw (np.array): n feature x tilex x tiley array of raw estimates
        corrected (np.array): n feature x tilex x tiley array of corrected estimates
        col_labels (list, optional): List of feature names for axes titles. Defaults to
            None.
        line_labels (list, optional): List of names for ylabel of leftmost plots.
            Defaults to ('Raw', 'Corrected', 'Difference').
        range_min (tuple, optional): N features long tuple of minimal range for color
            bars. Defaults to (5,5,0.1)
        range_max (tuple, optional): N features long tuple of maximal range for color
            bars. Defaults to None, no max.
        axes (np.array, optional): 3 x n features array of axes to plot into. Defaults
            to None.

    Returns:
        plt.Figure: Figure instance
    """
    ncols = raw.shape[0]
    if axes is None:
        fig, axes = plt.subplots(3, ncols)
        fig.set_size_inches((ncols * 3.5, 6))
        fig.subplots_adjust(top=0.9, wspace=0.15, hspace=0)
    else:
        fig = axes[0, 0].figure

    for col in range(ncols):
        vmin = corrected[col].min()
        vmax = corrected[col].max()
        rng = vmax - vmin
        if rng < range_min[col]:
            rng = range_min[col]
            vmin = vmin - rng
            vmax = vmax + rng
        elif (range_max is not None) and (rng > range_max[col]):
            rng = range_max[col]
            vmin = vmin - rng
            vmax = vmax + rng
        plot_matrix_with_colorbar(
            raw[col].T, axes[0, col], vmin=vmin - rng / 5, vmax=vmax + rng / 5
        )
        plot_matrix_with_colorbar(
            corrected[col].T,
            axes[1, col],
            vmin=vmin - rng / 5,
            vmax=vmax + rng / 5,
        )
        plot_matrix_with_colorbar(
            (raw[col] - corrected[col]).T,
            axes[2, col],
            cmap="RdBu_r",
            vmin=-rng,
            vmax=rng,
        )

    for x in axes.flatten():
        x.set_xticks([])
        x.set_yticks([])
    if col_labels is not None:
        for il, label in enumerate(col_labels):
            axes[0, il].set_title(label, fontsize=11)
    if line_labels is not None:
        for il, label in enumerate(line_labels):
            axes[il, 0].set_ylabel(label, fontsize=11)
    return fig


def plot_all_rounds(
    stack, view=None, channel_colors=None, grid=True, round_labels=None
):
    """Plot all rounds of a stack in a grid

    Args:
        stack (np.array): Image stack to plot (x y z round)
        view (np.array, optional): View to plot. Defaults to None, full view.
        channel_colors (list, optional): List of colors for each channel. Defaults to
            None, which will use the default colors (r, g, m, c).
        grid (bool, optional): Whether to plot a grid. Defaults to True.
        round_labels (list, optional): List of round labels. Defaults to None, which
            will use "Round {iround}".

    Returns:
        plt.Figure: Figure instance
        np.array: RGB stack
    """
    if channel_colors is None:
        channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])
    if view is None:
        view = np.array([[0, stack.shape[0]], [0, stack.shape[1]]])
    nrounds = stack.shape[3]

    def round_image(iround):
        vmax = np.percentile(
            stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
            99.99,
            axis=(0, 1),
        )
        vmin = np.percentile(
            stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
            0.01,
            axis=(0, 1),
        )
        return iss.vis.to_rgb(
            stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
            channel_colors,
            vmin=vmin,
            vmax=vmax,
        )

    # Make the smallest rectangle that contains `nrounds` axes
    nrows = int(np.sqrt(nrounds))
    ncols = int(np.ceil(nrounds / nrows))
    fig = plt.figure(figsize=(3.5 * ncols, 3.2 * nrows))
    rgb_stack = np.empty(np.diff(view, axis=1).ravel().tolist() + [3, nrounds])
    if round_labels is None:
        round_labels = [f"Round {iround}" for iround in range(nrounds)]
    for iround in range(nrounds):
        ax = fig.add_subplot(nrows, ncols, iround + 1)
        rgb = round_image(iround)
        rgb_stack[..., iround] = rgb
        ax.imshow(rgb)
        ax.set_title(round_labels[iround])
        if grid:
            ax.grid(color="w", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    fig.tight_layout()
    iss.vis.add_bases_legend(channel_colors)
    return fig, rgb_stack


def plot_matrix_with_colorbar(mtx, ax=None, **kwargs):
    """Plot a matrix with a colorbar just on the side

    Args:
        mtx (np.array): Matrix to plot
        ax (plt.Axes, optional): Axes instance. Will be created if None. Defaults to
            None.

    Returns:
        plt.Axes: Colorbar axes
        plt.colorbar: Colorbar instance
    """
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    im = ax.imshow(mtx, **kwargs)
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    cb = ax.figure.colorbar(im, cax=cax)
    return cax, cb


def plot_registration_correlograms(
    data_path,
    prefix,
    figure_name,
    debug_dict,
):
    target_folder = iss.io.get_processed_path(data_path) / "figures" / prefix
    print(f"Creating figures in {target_folder}")
    if not target_folder.exists():
        target_folder.mkdir()
    ops = iss.io.load_ops(data_path)
    mshift = ops["rounds_max_shift"]
    for what, data in debug_dict.items():
        print(f"Plotting {what}", flush=True)
        if what == "align_within_channels":
            _plot_within_channel_correlogram(data, target_folder, figure_name, mshift)
        elif what == "estimate_correction":
            _plot_between_channels_correlogram(data, target_folder, figure_name, mshift)
        elif what == "correct_by_block":
            tile_coors = ops["ref_tile"]
            fig = plt.figure(figsize=(2 * 7, 1.5 * 3))
            iss.vis.diagnostics.plot_affine_debug_images(data, fig=fig)
            fig.suptitle(f"{prefix} - Tile {tile_coors}")
            tile_name = "_".join([str(x) for x in tile_coors])
            fig.savefig(target_folder / f"affine_debug_{prefix}_{tile_name}.png")
        else:
            raise NotImplementedError(f"Unknown correlogram output: {what}")
    plt.close("all")
    print(f"Saved figures to {target_folder}")


def _plot_between_channels_correlogram(data, target_folder, figure_name, max_shift=100):
    angle_scales = set()
    for d in data:
        angle_scales.update(d.keys())
    angle_scales = natsorted(angle_scales)
    columns = natsorted(np.unique([a.split("_")[-1] for a in angle_scales]))
    rows = natsorted(np.unique([a.split("_")[-3] for a in angle_scales]))
    fig = plt.figure(figsize=(len(columns) * 3.5, len(rows) * 3))
    for ich, ch_data in enumerate(data):
        if not ch_data:
            continue
        for irow, row_name in enumerate(rows):
            for icol, col_name in enumerate(columns):
                ax = fig.add_subplot(
                    len(rows), len(columns), 1 + icol + irow * len(columns)
                )
                angle_scale = f"estimate_angle_{row_name}_scale_{col_name}"
                xcorr = ch_data[angle_scale]["xcorr"]
                angles = ch_data[angle_scale]["angles"]
                best_angle_id = xcorr.max(axis=(1, 2)).argmax()
                xcorr = xcorr[best_angle_id]
                title = f"Best angle: {angles[best_angle_id]:.2f}"
                if icol == 0:
                    title += f" (range {angles.min():.2f} - {angles.max():.2f})"
                ax.set_title(title)
                _draw_correlogram(ax, xcorr, max_shift, 0, np.percentile(xcorr, 99.999))
                if icol == 0:
                    ax.set_ylabel(f"Angle {row_name}")
                if irow == len(rows) - 1:
                    ax.set_xlabel(f"Scale {col_name}")

        fig.tight_layout()
        fig.savefig(
            target_folder / f"{figure_name}_register_ch{ich}_to_ref_ch.pdf",
            dpi=300,
            transparent=True,
        )
        fig.clear()


def _plot_within_channel_correlogram(data, target_folder, figure_name, max_shift=100):
    # find the number of channels and rounds
    chan_and_round = np.vstack(tuple(data.keys()))
    nchannels, nrounds = np.max(chan_and_round, axis=0) + 1
    ncol = nrounds
    fig = plt.figure()
    for ch in range(nchannels):
        for rnd in range(nrounds):
            rnd_data = data[(ch, rnd)]
            nrow = len(rnd_data)
            fig.set_size_inches(ncol * 3.5, nrow * 3)
            for irow, row_name in enumerate(rnd_data):
                xcorr = rnd_data[row_name]
                ax = fig.add_subplot(nrow, ncol, 1 + rnd + irow * ncol)
                if rnd == 0:
                    ax.set_ylabel(row_name)
                if irow == nrow - 1:
                    ax.set_xlabel(f"Round {rnd}")
                if isinstance(xcorr, dict):
                    angles = xcorr["angles"]
                    xcorr = xcorr["xcorr"]
                    best_angle_id = xcorr.max(axis=(1, 2)).argmax()
                    xcorr = xcorr[best_angle_id]
                    ax.set_title(
                        f"Best angle: {angles[best_angle_id]:.3f}"
                        + f" (range {angles.min():.3f} - {angles.max():.3f})"
                    )
                else:
                    # find x,y of max of xcorr
                    hrow, hcol = np.asarray(xcorr.shape) // 2
                    max_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
                    selected_shift = np.asarray(max_idx) - np.asarray([hrow, hcol])
                    ax.set_title(
                        f"Shift: {selected_shift[0]:.3f}, {selected_shift[1]:.3f}"
                    )
                _draw_correlogram(ax, xcorr, max_shift, 0, np.percentile(xcorr, 99.999))
        fig.tight_layout()
        fig.savefig(
            target_folder / f"{figure_name}_shifts_channel_{ch}.pdf", transparent=True
        )
        fig.clear()


def _draw_correlogram(ax, xcorr, max_shift, vmin, vmax):
    ax.imshow(xcorr, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw horizontal and vertical lines intersecting at the center
    ax.axhline(y=max_shift, color="black", linestyle="-", linewidth=1, alpha=0.2)
    ax.axvline(x=max_shift, color="black", linestyle="-", linewidth=1, alpha=0.2)
    # Draw circle at the center
    center_circle = patches.Circle(
        (max_shift, max_shift),
        radius=max_shift / 10,
        edgecolor="black",
        facecolor="none",
        linewidth=1,
        alpha=0.3,
    )
    ax.add_patch(center_circle)
    # Draw a hollow circle around the maximum of the cross-correlation
    max_idx = np.unravel_index(np.argmax(xcorr), xcorr.shape)
    circle = patches.Circle(
        (max_idx[1], max_idx[0]),
        radius=max_shift / 11,
        edgecolor="white",
        facecolor="none",
        linewidth=1,
    )
    ax.add_patch(circle)
    # Also draw a red dot in the center of the circle
    ax.scatter(
        max_idx[1],
        max_idx[0],
        color="red",
        s=0.5,
        alpha=0.2,
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
    reg_stack, bad_pixels = sequencing.load_and_register_sequencing_tile(
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

    ops = iss.io.load_ops(data_path)
    ref_prefix = ops["reference_prefix"]
    ref_ch = ops["reg2ref_reference_channels"]
    print(f"Stitching ref")
    ref = iss.pipeline.stitch_registered(
        data_path, prefix=ref_prefix, roi=roi, channels=ref_ch
    )
    ref = np.nanmean(ref, axis=2)

    bc_chans = ops.get("reg2ref_barcode_channels", [0, 1, 2, 3])
    print(f"Stitching barcode")
    barcode_round_1 = iss.pipeline.stitch_registered(
        data_path, prefix=barcode_prefix, roi=roi, channels=bc_chans
    )
    barcode_round_1 = np.nanmax(barcode_round_1, axis=2)
    mch_chan = ops.get("reg2ref_mCherry_channels", [3])
    print(f"Stitching mCherry")
    mcherry = iss.pipeline.stitch_registered(
        data_path, prefix=mcherry_prefix, roi=roi, channels=mch_chan
    )
    mcherry = np.nanmean(mcherry, axis=2)

    st = np.dstack([ref, mcherry, barcode_round_1])
    rgb = iss.vis.to_rgb(
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
            rgb_part = iss.vis.to_rgb(
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
