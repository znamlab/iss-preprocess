"""Generic plotting utilities

They are not even specific to ISS data, but can be used for any kind of data
visualization. We might consider moving some out of this repo.
"""

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def get_stack_part(stack, xlim, ylim):
    ylim = sorted(ylim)
    xlim = sorted(xlim)
    return stack[ylim[0] : ylim[1], xlim[0] : xlim[1]]


def get_spot_part(df, xlim, ylim, return_mask=False, xcol="x", ycol="y"):
    """Get the part of the dataframe that is within the xlim and ylim

    Args:
        df (pd.DataFrame): the dataframe with the x and y columns
        xlim (list): the x limits
        ylim (list): the y limits
        return_mask (bool, optional): if True, return the mask that was used to filter
            the dataframe
        xcol (str, optional): Name of the x column. Defaults to 'x'.
        ycol (str, optional): Name of the y column. Defaults to 'y'.

    Returns:
        pd.DataFrame: the filtered dataframe
        np.array: the mask that was used to filter the dataframe, only if return_mask is
            True
    """
    ylim = sorted(ylim)
    xlim = sorted(xlim)
    mask = (
        (df[xcol] >= xlim[0])
        & (df[xcol] < xlim[1])
        & (df[ycol] >= ylim[0])
        & (df[ycol] < ylim[1])
    )
    if return_mask:
        return df[mask], mask
    return df[mask]


def plot_bc_over_mask(
    ax,
    ma,
    bc,
    mask_assignment,
    xlim,
    ylim,
    nc=20,
    show_bg_barcodes=False,
    mask_alpha=0.5,
    mask_centers=None,
    line2mask=None,
    bg_marker_size=20,
    too_far_marker_size=5,
    assigned_marker_size=30,
    mask_center_marker_size=2,
    too_far_edge_width=0.2,
    assigned_edge_width=0.2,
):
    # get the list of colors from tab20
    colors = matplotlib.cm.get_cmap("tab20", nc).colors
    ax.imshow(
        get_stack_part(ma, xlim, ylim) % nc,
        cmap="tab20",
        vmin=0,
        vmax=nc - 1,
        interpolation="none",
        zorder=-100,
        alpha=mask_alpha,
    )
    # centroids = get_spot_part()
    sp_col = (mask_assignment % nc).astype(int)
    too_far = mask_assignment == -2
    background = mask_assignment == -1
    assigned = mask_assignment >= 0
    barcodes = list(bc.corrected_bases.unique())
    bc_color = np.array([barcodes.index(b) for b in bc.corrected_bases]).astype(int)
    ax.scatter(
        bc.x.values[too_far] - xlim[0],
        bc.y.values[too_far] - ylim[0],
        color="w",
        edgecolors="k",
        linewidths=too_far_edge_width,
        s=too_far_marker_size,
        alpha=1,
    )
    if show_bg_barcodes:
        ec = [colors[i] for i in bc_color[background] % nc]
        alpha = 1
        s = bg_marker_size
    else:
        ec = "none"
        alpha = 0.5
        s = 5
    ax.scatter(
        bc.x.values[background] - xlim[0],
        bc.y.values[background] - ylim[0],
        color="k",
        edgecolors=ec,
        s=s,
        alpha=alpha,
        marker="o",
    )

    ec = [colors[i] for i in bc_color[assigned] % nc]
    fc = [colors[i] for i in sp_col[assigned]]
    ax.scatter(
        bc.x.values[assigned] - xlim[0],
        bc.y.values[assigned] - ylim[0],
        facecolors=fc,
        edgecolors=ec,
        linewidths=assigned_edge_width,
        s=assigned_marker_size,
        alpha=1,
        marker="o",
    )
    if line2mask is not None:
        assert mask_centers is not None
        ax.scatter(
            mask_centers.x - xlim[0],
            mask_centers.y - ylim[0],
            color="k",
            marker="+",
            alpha=0.3,
            zorder=-50,
            s=mask_center_marker_size,
        )
        for i, (sp_id, series) in enumerate(bc.iterrows()):
            target = mask_assignment[i]
            if target < 0:
                continue

            target = mask_centers.loc[target]
            ax.plot(
                [series.x - xlim[0], target.x - xlim[0]],
                [series.y - ylim[0], target.y - ylim[0]],
                color="k",
                alpha=0.5,
                linewidth=0.2,
                zorder=-70,
            )
    ax.axis("off")


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
