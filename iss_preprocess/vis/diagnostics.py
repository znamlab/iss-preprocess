from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from flexiznam import PARAMETERS
from iss_preprocess.pipeline import stitch_tiles, register_adjacent_tiles
from iss_preprocess.pipeline import ara_registration as ara_reg
from iss_preprocess.io.load import load_ops


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
    fig.savefig(figure_folder / f"average_for_correction_other_prefix.png", dpi=600)


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


def plot_registration(data_path, roi, reference_prefix="genes_round_1_1"):
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

    area_ids = ara_reg.make_area_image(
        data_path=data_path, roi=roi, atlas_size=10, full_scale=False
    )
    reg_metadata = ara_reg.load_registration_reference_metadata(data_path, roi=roi)

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    shift_right, shift_down, tile_shape = register_adjacent_tiles(
        data_path, ref_coors=ops["ref_tile"], prefix=reference_prefix
    )

    stitched_stack = stitch_tiles(
        data_path,
        reference_prefix,
        shift_right,
        shift_down,
        suffix=ops["projection"],
        roi=roi,
        ich=ops["ref_ch"],
        correct_illumination=True,
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
    ops = load_ops(data_path)
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
            ax = fig.add_subplot(11, 2, ir * 2 + ip + 3, sharex=fig.axes[0])
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
    fig.savefig(figure_folder / f"pixel_value_distributions.png", dpi=600)
