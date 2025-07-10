import warnings

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted

from ..io import get_processed_path, load_ops, write_stack
from . import add_bases_legend, animate_sequencing_rounds, to_rgb
from .utils import plot_matrix_with_colorbar


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
        axes (list): List of plt.Axes to plot. Must be same length as number of channel
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
    axes = fig.subplots(nchans, 7, squeeze=False)
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
            plot_matrix_with_colorbar(
                rsh[..., i].T, ax, vmin=m - max_shift, vmax=m + max_shift
            )
            if idir == 0 and i == 0:
                ax.set_title("Raw")
            ax = axes[idir * 2 + i, 1]
            plot_matrix_with_colorbar(
                sh[..., i].T, ax, vmin=m - max_shift, vmax=m + max_shift
            )
            if idir == 0 and i == 0:
                ax.set_title("Corrected")
            ax = axes[idir * 2 + i, 2]
            plot_matrix_with_colorbar(
                delta_shift.T, ax, vmin=0, vmax=max_delta_shift * 1.1
            )
            if idir == 0 and i == 0:
                ax.set_title("Delta")
            ax = axes[idir * 2 + i, 3]
            if idir == 0 and i == 0:
                ax.set_title("Bad tiles")
            plot_matrix_with_colorbar(bad2plot.T, ax, vmin=0, vmax=2)
        axes[idir * 2, 4].set_title("Max corr")
        plot_matrix_with_colorbar(
            xc.T, axes[idir * 2, 4], vmin=0, vmax=1, cmap="coolwarm"
        )
        axes[idir * 2 + 1, 4].axis("off")
    # put back runtime warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    fig.tight_layout()
    fig_file = (
        get_processed_path(data_path)
        / "figures"
        / "registration"
        / prefix
        / f"adjacent_tile_reg_{prefix}_roi{roi}.png"
    )
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_file, dpi=300)
    print(f"Saving {fig_file}")
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
    ops = load_ops(data_path)
    camera_order = ops["camera_order"]
    distri = distributions.copy()
    fig = plt.figure(figsize=(10, 20), facecolor="white")
    colors = ["blue", "green", "red", "purple"]
    # TODO: adapt to varying number of rounds
    nrounds = [np.sum([k.startswith(p) for k in distri]) for p in grand_averages]
    if not len(grand_averages):
        print("No grand averages to plot. Not doing anything.")
        return

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


def plot_all_rounds(
    stack,
    view=None,
    channel_colors=None,
    grid=True,
    round_labels=None,
    fig=None,
    axes=None,
    vmin=None,
    vmax=None,
    legend_kwargs=None,
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
        fig (plt.Figure, optional): Figure to plot into. Defaults to None, will create
            a new figure.
        axes (list, optional): List of axes to plot into. Defaults to None, will create
            new axes.
        vmin (float, optional): Minimum value for the colormap. Defaults to None.
        vmax (float, optional): Maximum value for the colormap. Defaults to None.

    Returns:
        plt.Figure: Figure instance
        np.array: RGB stack
    """
    if channel_colors is None:
        channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])
    if view is None:
        view = np.array([[0, stack.shape[0]], [0, stack.shape[1]]])
    nrounds = stack.shape[3]

    def round_image(iround, vmin, vmax):
        if vmax is None:
            vmax = np.percentile(
                stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
                99.99,
                axis=(0, 1),
            )
        if vmin is None:
            vmin = np.percentile(
                stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
                0.01,
                axis=(0, 1),
            )
        return to_rgb(
            stack[view[0, 0] : view[0, 1], view[1, 0] : view[1, 1], :, iround],
            channel_colors,
            vmin=vmin,
            vmax=vmax,
        )

    # Make the smallest rectangle that contains `nrounds` axes
    nrows = int(np.sqrt(nrounds))
    ncols = int(np.ceil(nrounds / nrows))
    if fig is None:
        fig = plt.figure(figsize=(3.5 * ncols, 3.2 * nrows))
    rgb_stack = np.empty(np.diff(view, axis=1).ravel().tolist() + [3, nrounds])
    if round_labels is None:
        round_labels = [f"Round {iround}" for iround in range(nrounds)]
    for iround in range(nrounds):
        if axes is None:
            ax = fig.add_subplot(nrows, ncols, iround + 1)
        else:
            ax = axes[iround]
        rgb = round_image(iround, vmin=vmin, vmax=vmax)
        rgb_stack[..., iround] = rgb
        ax.imshow(rgb)
        ax.set_title(round_labels[iround])
        if grid:
            ax.grid(color="w", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    fig.tight_layout()
    if legend_kwargs is None:
        legend_kwargs = {}
    add_bases_legend(channel_colors, **legend_kwargs)
    return fig, rgb_stack


def plot_registration_correlograms(
    data_path,
    prefix,
    figure_name,
    debug_dict,
):
    target_folder = get_processed_path(data_path) / "figures" / "registration" / prefix
    print(f"Creating figures in {target_folder}")
    target_folder.mkdir(exist_ok=True, parents=True)
    ops = load_ops(data_path)
    mshift = ops["rounds_max_shift"]
    figs = {}
    for what, data in debug_dict.items():
        print(f"Plotting {what}", flush=True)
        if what == "align_within_channels":
            fig = _plot_within_channel_correlogram(
                data, target_folder, figure_name, mshift
            )
        elif what == "estimate_correction":
            fig = _plot_between_channels_correlogram(
                data, target_folder, figure_name, mshift
            )
        elif what == "correct_by_block":
            tile_coors = ops["ref_tile"]
            fig = plt.figure(figsize=(2 * 7, 1.5 * 3))
            plot_affine_debug_images(data, fig=fig)
            fig.suptitle(f"{prefix} - Tile {tile_coors}")
            tile_name = "_".join([str(x) for x in tile_coors])
            fig.savefig(target_folder / f"affine_debug_{prefix}_{tile_name}.png")
        else:
            raise NotImplementedError(f"Unknown correlogram output: {what}")
        figs[what] = fig
    plt.close("all")
    print(f"Saved figures to {target_folder}")
    return figs


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
        fig.clear()
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

    return fig


def _plot_within_channel_correlogram(data, target_folder, figure_name, max_shift=100):
    # find the number of channels and rounds
    chan_and_round = np.vstack(tuple(data.keys()))
    nchannels, nrounds = np.max(chan_and_round, axis=0) + 1
    ncol = nrounds
    fig = plt.figure()
    for ch in range(nchannels):
        fig.clear()
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

    return fig


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


def plot_spot_sign_image(spot_image):
    """
    Plot the average spot sign image.

    Args:
        spot_image: X x Y array of average spot sign values.

    """
    plt.figure(figsize=(5, 5), facecolor="white")
    plt.pcolormesh(
        spot_image, cmap="bwr", vmin=-1, vmax=1, edgecolors="white", linewidths=1
    )
    image_size = spot_image.shape[0]
    ticks_labels = np.arange(image_size) - int(image_size / 2)
    plt.xticks(np.arange(image_size) + 0.5, ticks_labels)
    plt.yticks(np.arange(image_size) + 0.5, ticks_labels)
    plt.colorbar()
    plt.gca().set_aspect("equal")


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
    fig, rgb_stack = plot_all_rounds(
        reg_stack, view, channel_colors, round_labels=round_labels
    )
    fig.tight_layout()
    fname = target_folder / f"{fname_base}.png"
    fig.savefig(fname)
    print(f"Saved to {fname}")

    # also save the stack for fiji
    write_stack(
        (rgb_stack * 255).astype("uint8"),
        target_folder / f"{fname_base}_rgb_stack_{reg_stack.shape[-1]}rounds.tif",
    )

    print("Animating")
    animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"{fname_base}.mp4",
        vmax=vmaxs,
        vmin=vmins,
        extent=(view[0], view[1]),
        channel_colors=channel_colors,
    )


def check_reg2ref_using_stitched(
    save_path,
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

    st = np.dstack([stitched_stack_reference, stitched_stack_target])
    rgb = to_rgb(
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
