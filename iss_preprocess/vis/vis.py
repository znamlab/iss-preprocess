import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.cluster import hierarchy
import iss_preprocess as iss


def plot_spot_sign_image(spot_image):
    """
    Plot the average spot sign image.
    
    Args:
        spot_image: X x Y array of average spot sign values.
    
    """
    plt.figure(figsize=(5, 5), facecolor="white")
    plt.pcolormesh(spot_image, cmap="bwr", vmin=-1, vmax=1, edgecolors="white", linewidths=1)
    image_size = spot_image.shape[0]
    ticks_labels = np.arange(image_size) - int(image_size / 2)
    plt.xticks(np.arange(image_size) + 0.5, ticks_labels)
    plt.yticks(np.arange(image_size) + 0.5, ticks_labels)
    plt.colorbar()
    plt.gca().set_aspect('equal')    


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
    plt.figure(figsize=(10, 20))
    for igene, gene in enumerate(gene_names):
        plt.subplot(10, 9, igene + 1)
        plt.imshow(np.reshape(gene_dict[:, igene], (nrounds, nchannels)), cmap="gray")
        plt.title(gene)
        plt.xticks(np.arange(4), BASES)
    plt.tight_layout()


def add_bases_legend(extent, channel_colors):
    """
    Add legend for base colors to a plot.

    Args:
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]]
        channel_colors (list): list of colors for each channel.

    """
    xrange = extent[0][1] - extent[0][0]
    yrange = extent[1][1] - extent[1][0]
    for i, (color, base) in enumerate(zip(channel_colors, iss.call.BASES)):
        plt.text(
            xrange * 0.6 + i * xrange * 0.1,
            yrange * 0.95,
            base,
            color=color,
            fontweight="bold",
            fontsize=32,
        )


def round_to_rgb(stack, iround, extent, channel_colors, vmax):
    """
    Convert a single sequencing round to RGB image.
    
    Args:
        stack (ndarray): X x Y x C x R stack
        iround (int): sequencing round to visualize
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]]
        channel_colors (list): list of colors for each channel.
        vmax (float): maximum value for each channel.
        
    Returns:
        RGB image.
    """
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
        vmax (float): maximum value for each channel.
        extent (list): extent of plot. [[xmin, xmax], [ymin, ymax]]
        channel_colors (list): list of colors for each channel. 
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
    add_bases_legend(extent, channel_colors)
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
    add_bases_legend(extent, channel_colors)

    plt.axis("off")

    def animate(iround):
        im.set_data(round_to_rgb(stack, iround, extent, channel_colors, vmax))

    anim = FuncAnimation(fig, animate, frames=nrounds, interval=200)
    plt.show()
    anim.save(savefname, writer=FFMpegWriter(fps=2))

