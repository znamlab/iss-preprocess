import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.cluster import hierarchy

from ..call import BASES

__all__ = [
    "plot_clusters",
    "to_rgb",
    "make_lut",
    "plot_spots",
    "plot_gene_matrix",
    "plot_gene_templates",
    "add_bases_legend",
    "round_to_rgb",
    "plot_sequencing_rounds",
    "animate_sequencing_rounds",
    "plot_spot_called_base",
]


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

    # importing seaborn is slow. Do it only when needed
    import seaborn as sns

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
        nrows=1, ncols=nclusters, facecolor="w", label="cluster_means", figsize=(8, 2)
    )
    for icluster in range(nclusters):
        plt.sca(ax[icluster])
        plt.imshow(np.stack(cluster_means, axis=2)[icluster, :, :])
        plt.xlabel("rounds")
        plt.ylabel("channels")
        plt.xticks(np.arange(nrounds), np.arange(1, nrounds + 1, dtype=int))
        plt.yticks(np.arange(nch), np.arange(nch, dtype=int))
        plt.title(f"Cluster {icluster+1}")

    plt.tight_layout()
    figs.append(fig)

    return figs


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
        vmax = np.nanmax(stack, axis=(0, 1))
    elif np.isscalar(vmax):
        vmax = np.full(nchannels, vmax)
    else:
        vmax = np.asarray(vmax)
    if vmin is None:
        vmin = np.nanmin(stack, axis=(0, 1))
    elif np.isscalar(vmin):
        vmin = np.full(nchannels, vmin)
    else:
        vmin = np.asarray(vmin)

    scale = vmax - vmin

    # remove nans to make int
    stack_norm = np.nan_to_num(
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
        nlevels: number of LUT levels

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


def add_bases_legend(channel_colors, transform=None, **kwargs):
    """
    Add legend for base colors to a plot.

    Args:
        channel_colors (list): list of colors for each channel.
        transform (matplotlib.transforms.Transform): transform for legend.
        kwargs: additional keyword arguments for plt.text.

    """
    default_kwargs = dict(
        fontweight="bold",
        fontsize=32,
    )
    default_kwargs.update(kwargs)
    if transform is None:
        transform = plt.gca().transAxes
    for i, (color, base) in enumerate(zip(channel_colors, BASES)):
        plt.text(
            0.6 + i * 0.1,
            0.05,
            base,
            color=color,
            transform=transform,
            **default_kwargs,
        )


def round_to_rgb(
    stack,
    iround,
    extent=None,
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    vmax=None,
    vmin=None,
):
    """
    Convert a single sequencing round to RGB image.

    Args:
        stack (ndarray): X x Y x C x R stack
        iround (int): sequencing round to visualize
        extent (list, optional): extent of plot. [[xmin, xmax], [ymin, ymax]] or None,
             in which case the full image is used. Default: None
        channel_colors (list, optional): list of colors for each channel. Default to
            red, green, magenta, cyan
        vmax (float, optional): maximum value for each channel.
        vmin (float, optional): minimum value for each channel.

    Returns:
        RGB image.
    """
    if extent is None:
        extent = ((0, stack.shape[0]), (0, stack.shape[1]))

    if vmin is None:
        vmin = 0
    if vmax is None:
        vmax = 1

    return to_rgb(
        stack[extent[0][0] : extent[0][1], extent[1][0] : extent[1][1], :, iround],
        channel_colors,
        vmin=np.array([1, 1, 1, 1]) * vmin,
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
            Default: red, green, magenta, cyan = ([1, 0, 0], [0, 1, 0], [1, 0, 1],
            [0, 1, 1])

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
    vmin=0,
    extent=((0, 2000), (0, 2000)),
    channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    axes_titles=None,
):
    """
    Animate sequencing rounds as RGB images amd save as an mp4 file.

    Args:
        stack (ndarray): X x Y x C x R stack or list of such stacks
        savefname (str): filename to save animation
        vmax (float): maximum value for each channel.
        vmin (float): minimum value for each channel.
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]]
        channel_colors (list): list of colors for each channel.
            Default: red, green, magenta, cyan = ([1, 0, 0], [0, 1, 0], [1, 0, 1],
            [0, 1, 1])
        axes_titles (list, optional): list of titles for each stack

    """
    if not isinstance(stack, list):
        stack = [stack]
    nimg = len(stack)
    nrounds = [s.shape[3] for s in stack]
    assert len(set(nrounds)) == 1, "All stacks must have the same number of rounds"
    nrounds = nrounds[0]

    fig, axes = plt.subplots(1, nimg, figsize=(10 * nimg, 10))
    if nimg == 1:
        axes = [axes]
    fig.patch.set_facecolor("black")

    imgs = []
    for iax, s in enumerate(stack):
        im = axes[iax].imshow(round_to_rgb(s, 0, extent, channel_colors, vmax, vmin))
        imgs.append(im)
        axes[iax].axis("off")
        if axes_titles is not None:
            axes[iax].text(
                0.01,
                0.95,
                axes_titles[iax],
                color="white",
                transform=axes[iax].transAxes,
                horizontalalignment="left",
                verticalalignment="top",
                fontsize=20,
            )
    add_bases_legend(channel_colors, transform=axes[-1].transAxes)
    fig.tight_layout()

    def animate(iround):
        for iax, im in enumerate(imgs):
            im.set_data(round_to_rgb(stack[iax], iround, extent, channel_colors, vmax))

    anim = FuncAnimation(fig, animate, frames=nrounds, interval=200)
    anim.save(savefname, writer=FFMpegWriter(fps=2))


def plot_spot_called_base(spots, ax, iround, base_color=None, **kwargs):
    """Plot called base for each spot.

    This will write a single base, as colored letter, for each spot at the given round.

    Args:
        spots (DataFrame): table of spots.
        ax (matplotlib.axes.Axes): axis to plot on.
        iround (int): round to plot.
        base_color (dict, optional): dictionary of base colors. Defaults to None.
        kwargs: additional keyword arguments for plt.text.

    Returns:
        None

    """
    bases = np.hstack([BASES, "N"])
    if base_color is None:
        channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5])
        base_color = {b: c for b, c in zip(bases, channel_colors)}

    default_kwargs = dict(
        fontweight="bold",
        fontsize=6,
        verticalalignment="center",
        horizontalalignment="center",
    )
    default_kwargs.update(kwargs)
    for i, spot in spots.iterrows():
        base = spot["bases"][iround]

        ax.text(
            spot["x"],
            spot["y"],
            base,
            color=base_color[base],
            **default_kwargs,
        )
