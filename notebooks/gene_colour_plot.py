from __future__ import annotations

import colorsys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from iss_preprocess.io import (
    get_roi_dimensions,
    load_micromanager_metadata,
)

__all__ = ["plot_genes_colour"]


def plot_genes_colour(
    gene_data: pd.DataFrame,
    data_path: str,
    roi: int = 1,
    gene_colors: Optional[Dict[str, Any]] = None,
    cmap_name: str = "tab20",
    markersize: float = 0.1,
    alpha: float = 1.0,
    add_legend: bool = True,
    return_fig_ax: bool = False,
):
    """
    Plot selected genes as colored spots on a single axes (no rotation, no titles,
    no grid, and no axis ticks/labels).

    Args:
        gene_data: DataFrame with at least columns ['Gene', 'x', 'y'].
                   Filter externally as needed (e.g., by spot_score).
        data_path: Base path used to infer field/tile dimensions.
        roi: ROI index (1-based) for dimension inference.
        gene_colors: Optional mapping {gene_name: color}. If provided, only these
                     genes are plotted using the specified colors.
        cmap_name: Matplotlib colormap used when gene_colors is None. Default 'tab20'.
        markersize: Marker size.
        alpha: Marker alpha.
        return_fig_ax: If True, returns (fig, ax) instead of calling plt.show().

    Returns:
        Optional[Tuple[Figure, Axes]] if return_fig_ax is True.
    """
    plt.rcParams["figure.facecolor"] = "white"

    # Infer image dimensions from metadata (no rotation)
    roi_dims = get_roi_dimensions(data_path)[roi - 1]
    nx = roi_dims[1] + 1
    ny = roi_dims[2] + 1

    metadata = load_micromanager_metadata(data_path, "genes_round_1_1")
    x_dim = int(metadata["FrameKey-0-0-0"]["ROI"].split("-")[2])
    y_dim = int(metadata["FrameKey-0-0-0"]["ROI"].split("-")[3])
    dim_x = ((nx - 1) * x_dim * 0.9) + x_dim
    dim_y = ((ny - 1) * y_dim * 0.9) + y_dim

    # Determine which genes to plot and their colors
    if gene_colors:
        available = set(gene_data["Gene"].unique())
        genes = [g for g in gene_colors.keys() if g in available]
        colors = [gene_colors[g] for g in genes]
    else:
        genes = list(np.sort(gene_data["Gene"].unique()))
        cmap = plt.get_cmap(cmap_name, max(len(genes), 1))
        colors = [cmap(i % cmap.N) for i in range(len(genes))]

    # Single-axes figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot each gene's spots without any rotation
    items_plotted: list[tuple[str, Any]] = []
    for g, color in zip(genes, colors):
        gd = gene_data[gene_data["Gene"] == g]
        if gd.empty:
            continue
        x = gd["x"].to_numpy()
        y = gd["y"].to_numpy()
        ax.plot(x, y, "o", c=color, markersize=markersize, alpha=alpha, linestyle="")
        items_plotted.append((g, color))

    # Styling: equal aspect, no titles, no grid, no ticks/labels
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, dim_x)
    ax.set_ylim(0, dim_y)
    # Keep image-like orientation (origin at top-left)
    ax.invert_yaxis()

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Add legend to the right (sorted by rainbow color order)
    if add_legend and items_plotted:

        def hue_key(item: tuple[str, Any]) -> tuple[int, float]:
            _, c = item
            try:
                r, g, b = mcolors.to_rgb(c)
                h, s, _ = colorsys.rgb_to_hsv(r, g, b)
            except ValueError:
                # Unknown color; push to end
                return (1, 1.0)
            # Achromatic colors (low saturation) pushed after chromatic ones
            achromatic = 1 if s < 0.1 else 0
            # Rainbow order: sort by hue ascending (red -> ... -> violet)
            adjusted_h = (h - 0.0) % 1.0
            return (achromatic, adjusted_h)

        sorted_items = sorted(items_plotted, key=hue_key)

        legend_handles: list[Line2D] = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=c,
                linestyle="",
                markersize=6,
                label=g,
            )
            for g, c in sorted_items
        ]

        ax.legend(
            handles=legend_handles,
            title="Genes",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )

    # Reserve space on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    if return_fig_ax:
        return fig, ax

    plt.show()
