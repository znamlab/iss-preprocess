import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.ticker import FixedLocator
from scipy.cluster import hierarchy
from skimage.measure import block_reduce
import iss_preprocess as iss
import seaborn as sns
import pandas as pd
from pathlib import Path
import tifffile
from znamutils import slurm_it
from iss_preprocess.io import get_processed_path, load_micromanager_metadata


def plot_clusters(cluster_means, spot_colors, cluster_inds):
    """
    Plot the cluster means and spot colors for each channel.

    Args:
        cluster_means: list of nch x nclusters cluster means.
        spot_colors: round x channels x spots array of spot colors.
        cluster_inds: list of arrays of cluster indices for each round.

    Returns:
        figs: list of figures

    """
    nclusters, nch = cluster_means[0].shape
    nrounds = len(cluster_means)

    # Initialize min and max values for each channel
    global_min = np.full(nch, np.inf)
    global_max = np.full(nch, -np.inf)

    # Find global min and max values across all rounds
    for iround in range(nrounds):
        for ich in range(nch):
            global_min[ich] = min(global_min[ich], np.min(spot_colors[iround, ich, :]))
            global_max[ich] = max(global_max[ich], np.max(spot_colors[iround, ich, :]))

    figs = []
    for iround in range(nrounds):
        df = pd.DataFrame(
            np.hstack((spot_colors[iround].T, cluster_inds[iround][:, np.newaxis])),
            columns=[f"ch{i}" for i in range(nch)] + ["cluster"],
        )
        g = sns.PairGrid(df, hue="cluster", palette="tab10")
        g.map_diag(sns.histplot, bins=20)
        g.map_offdiag(sns.scatterplot, size=1, alpha=0.25, edgecolor=None)
        # Set the same number of ticks for each channel
        for ax in g.axes.flatten():
            ax.locator_params(axis="both", nbins=4)
        # Set the same axis limits for each channel
        for ich in range(nch):
            for jch in range(nch):
                if ich != jch:
                    g.axes[ich, jch].set_xlim(global_min[jch], global_max[jch])
                    g.axes[ich, jch].set_ylim(global_min[ich], global_max[ich])
        g.add_legend()
        g.figure.set_label(f"clusters_round_{iround}")
        g.figure.set_facecolor("w")
        figs.append(g.figure)

    fig, ax = plt.subplots(
        nrows=1, ncols=nclusters, facecolor="w", label="cluster_means"
    )
    for icluster in range(nclusters):
        plt.sca(ax[icluster])
        plt.imshow(np.stack(cluster_means, axis=2)[icluster, :, :])
        plt.xlabel("rounds")
        plt.ylabel("channels")
        plt.title(f"Cluster {icluster+1}")
        plt.locator_params(axis="both", nbins=4)
    plt.tight_layout()
    figs.append(fig)

    return figs


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


def to_rgb(stack, colors, vmax=None, vmin=None):
    """
    Convert multichannel stack to RGB image.

    Args:
        stack: X x Y x C stack.
        colors: maximum RGB values for each channel.
        vmax: maximum level for each channel
        vmin: minimum level for each channel

    Returns:
        X x Y x 3 RGB image

    """
    nchannels = stack.shape[2]

    if vmax is None:
        vmax = np.max(stack, axis=(0, 1))
    else:
        vmax = np.asarray(vmax)
    if vmin is None:
        vmin = np.min(stack, axis=(0, 1))
    else:
        vmin = np.asarray(vmin)

    scale = vmax - vmin

    stack_norm = (
        (stack - vmin[np.newaxis, np.newaxis, :])
        / scale[np.newaxis, np.newaxis, :]
        * 255
    ).astype(int)
    stack_norm = np.clip(stack_norm, 0, 255)
    rgb_stack = np.zeros([stack.shape[0], stack.shape[1], 3])
    for ich in range(nchannels):
        lut = make_lut(colors[ich])
        rgb_stack += lut[stack_norm[:, :, ich], :]

    return np.clip(rgb_stack, 0.0, 1.0)


def make_lut(color, nlevels=256):
    """
    Create a look-up table by interpolating between (0,0,0) and provided color.

    Args:
        color: the maximum RGB value for look-up table.
        nlevels: nummber of LUT levels

    Returns:
        nlevels x 3 matrix.

    """
    r = np.linspace(0.0, 1.0, nlevels)
    return r[:, np.newaxis] * color


def plot_spots(
    stack,
    spots,
    colors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    vmax=10000.0,
    vmin=500.0,
):
    """
    Visualize detected plots.

    Args:
        stack: X x Y x C stack
        spots: pandas.DataFrame of spot locations
        colors: maximum RGB values for each channel.
        vmax: maximum level for each channel
        vmin: minimum level for each channel

    """
    im = to_rgb(stack, colors, vmax=np.array([vmax]), vmin=np.array([vmin]))
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(im)
    for _, spot in spots.iterrows():
        c = plt.Circle(
            (spot["x"], spot["y"]), spot["size"], color="w", linewidth=0.5, fill=False
        )
        ax.add_patch(c)


def plot_gene_matrix(gene_df, cmap="inferno", vmax=2):
    """
    Plot matrix of gene expression after sorting rows and columns using
    hierarchical clustering.

    Args:
        gene_df (DataFrame): table of gene counts.

    """
    gene_mat = np.log(1 + gene_df.to_numpy())
    cell_order = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(hierarchy.ward(gene_mat), gene_mat)
    )
    gene_order = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(hierarchy.ward(gene_mat.T), gene_mat.T)
    )

    plt.figure(figsize=(20, 10))
    gene_mat_reordered = gene_mat[cell_order, :]
    gene_mat_reordered = gene_mat_reordered[:, gene_order]
    ax = plt.subplot(1, 1, 1)
    plt.imshow(gene_mat_reordered, cmap=cmap, vmax=vmax, interpolation="nearest")
    plt.xticks(
        range(gene_mat.shape[1]),
        gene_df.columns[gene_order],
        rotation=45,
        horizontalalignment="right",
    )
    ax.set_aspect("auto")


def plot_gene_templates(gene_dict, gene_names, BASES, nrounds=7, nchannels=4):
    """
    Plot gene templates.

    Args:
        gene_dict (ndarray): X x G matrix of gene templates. X = nrounds * nchannels
        gene_names (list): list of gene names
        BASES (list): list of base names
        nrounds (int): number of rounds. Default: 7
        nchannels (int): number of channels. Default: 4

    """
    ncols = 9
    nrows = int(np.ceil(len(gene_names) / ncols))
    fig = plt.figure(figsize=(10, 2 * nrows), facecolor="w", label="gene_templates")
    for igene, gene in enumerate(gene_names):
        plt.subplot(nrows, ncols, igene + 1)
        plt.imshow(np.reshape(gene_dict[:, igene], (nrounds, nchannels)), cmap="gray")
        plt.title(gene)
        plt.xticks(np.arange(4), BASES)
    plt.tight_layout()
    return fig


def add_bases_legend(channel_colors, transform=None):
    """
    Add legend for base colors to a plot.

    Args:
        channel_colors (list): list of colors for each channel.
        transform (matplotlib.transforms.Transform): transform for legend.

    """
    if transform is None:
        transform = plt.gca().transAxes
    for i, (color, base) in enumerate(zip(channel_colors, iss.call.BASES)):
        plt.text(
            0.6 + i * 0.1,
            0.05,
            base,
            color=color,
            fontweight="bold",
            fontsize=32,
            transform=transform,
        )


def round_to_rgb(stack, iround, extent, channel_colors, vmax):
    """
    Convert a single sequencing round to RGB image.

    Args:
        stack (ndarray): X x Y x C x R stack
        iround (int): sequencing round to visualize
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]] or None, in which
            case the full image is used.
        channel_colors (list): list of colors for each channel.
        vmax (float): maximum value for each channel.

    Returns:
        RGB image.
    """
    if extent is None:
        extent = ((0, stack.shape[0]), (0, stack.shape[1]))

    return to_rgb(
        stack[extent[0][0] : extent[0][1], extent[1][0] : extent[1][1], :, iround],
        channel_colors,
        vmin=np.array([0, 0, 0, 0]),
        vmax=np.array([1, 1, 1, 1]) * vmax,
    )


def plot_sequencing_rounds(
    stack,
    vmax=0.5,
    extent=((0, 2000), (0, 2000)),
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
):
    """
    Plot sequencing rounds as RGB images.

    Args:
        stack (ndarray): X x Y x C x R stack
        vmax (float, optional): maximum value for each channel. Default: 0.5
        extent (list, optional): extent of plot. [[xmin, xmax], [ymin, ymax]]. If None,
            use full image. Default: ((0, 2000), (0, 2000))
        channel_colors (list, optional): list of colors for each channel.
            Default: red, green, magenta, cyan = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])

    """
    nrounds = stack.shape[3]

    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_facecolor("black")
    for iround in range(nrounds):
        plt.subplot(2, 5, iround + 1)
        plt.imshow(round_to_rgb(stack, iround, extent, channel_colors, vmax))
        plt.axis("off")
        plt.title(f"Round {iround+1}", color="white")
    add_bases_legend(channel_colors)
    plt.tight_layout()


def animate_sequencing_rounds(
    stack,
    savefname,
    vmax=0.5,
    extent=((0, 2000), (0, 2000)),
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
):
    """
    Animate sequencing rounds as RGB images amd save as an MPEG file.

    Args:
        stack (ndarray): X x Y x C x R stack
        savefname (str): filename to save animation
        vmax (float): maximum value for each channel.
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]]
        channel_colors (list): list of colors for each channel.
            Default: red, green, magenta, cyan = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])

    """
    fig = plt.figure(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    nrounds = stack.shape[3]
    im = plt.imshow(round_to_rgb(stack, 0, extent, channel_colors, vmax))
    add_bases_legend(channel_colors)

    plt.axis("off")

    def animate(iround):
        im.set_data(round_to_rgb(stack, iround, extent, channel_colors, vmax))

    anim = FuncAnimation(fig, animate, frames=nrounds, interval=200)
    plt.show()
    anim.save(savefname, writer=FFMpegWriter(fps=2))


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
        plot_axis (bool, optional): Whether to plot gridlines at tile boundaries. Defaults to True.
        downsample_factor (int, optional): Amount to downsample overview. Defaults to 25.
        save_raw (bool, optional): Whether to save a tif with no gridlines. Defaults to True.
        dependency (str, optional): Dependency for the generates slurm scripts
        group_channels (bool, optional): Whether to group channels together. Defaults to True.
        use_slurm (bool, optional): Whether to use slurm to run the jobs. Defaults to True.
        vmin (list, optional): vmin for each channel. Default to None
        vmax (list, optional): vmax for each channel. Default to None
    """
    processed_path = get_processed_path(data_path)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix)
    image_metadata = load_micromanager_metadata(data_path, prefix)
    nchannels = image_metadata["Summary"]["Channels"]
    # Check if average image exists for illumination correction
    correct_illumination = (
        processed_path / "averages" / f"{prefix}_average.tif"
    ).exists()
    job_ids = []
    kwargs = dict(
        data_path=data_path,
        prefix=prefix,
        plot_grid=plot_grid,
        downsample_factor=downsample_factor,
        save_raw=save_raw,
        correct_illumination=correct_illumination,
        use_slurm=use_slurm,
        job_dependency=dependency,
        vmin=vmin,
        vmax=vmax,
    )

    for roi_dim in roi_dims:
        roi = roi_dim[0]
        if group_channels:
            channels = [list(range(nchannels))]
        else:
            channels = range(nchannels)

        for ch in channels:
            if isinstance(ch, list):
                scripts_name = f"plot_overview_{prefix}_{roi}_channels_{'_'.join([str(c) for c in ch])}"
            else:
                scripts_name = f"plot_overview_{prefix}_{roi}_channel_{ch}"
            job_ids.append(
                plot_single_overview(
                    roi=roi,
                    ch=ch,
                    nx=roi_dim[1] + 1,
                    ny=roi_dim[2] + 1,
                    slurm_folder=f"{Path.home()}/slurm_logs",
                    scripts_name=scripts_name,
                    **kwargs,
                )
            )
    return job_ids


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
        downsample_factor (int, optional): Amount to downsample overview. Defaults to 25.
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
        roi_dims = iss.io.get_roi_dimensions(data_path, prefix=prefix)
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
        small_stack = iss.pipeline.stitch_tiles(
            data_path,
            prefix=prefix,
            roi=roi,
            suffix="max",
            ich=c,
            correct_illumination=correct_illumination,
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
            vmax = np.percentile(stack, 98)
        plt.imshow(stack, vmax=vmax, vmin=vmin)
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
        ax.set_xticklabels(np.arange(0, len(ax.get_xticks())), rotation=90)
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
