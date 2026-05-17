"""Sindbis soma diagnostics.

Two pipeline stages live here so the soma workflow has one diagnostic module:

- shared (mouse-level) cluster-means / bleedthrough build:
  :func:`plot_shared_soma_reference_diagnostics` and friends. Produced by
  :func:`iss_preprocess.pipeline.somata.build_shared_soma_cluster_means` (writes
  PNGs/CSVs into ``processed/{mouse}/diagnostics/shared_soma_reference/``)
  and by the Sindbis soma notebook (rendered inline by calling with
  ``save_dir=None``).

- post-basecalling per-ROI filtered-soma overlays: per-ROI fan-out pattern
  used by :mod:`iss_preprocess.diagnostics.diag_stitching`. One slurm job per
  ROI from :func:`plot_filtered_soma_overlays`, each running
  :func:`plot_filtered_soma_overlay_roi`. ROIs with zero surviving somata are
  still rendered (background only) so the output set is uniform.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.measure import block_reduce
from znamutils import slurm_it

from ..io import get_processed_path, load_ops
from ..io.load import get_roi_dimensions
from ..pipeline.stitch import stitch_tiles
from ..vis.vis import plot_clusters, to_rgb

__all__ = [
    "summarize_cluster_sizes",
    "plot_bleedthrough_heatmaps",
    "plot_cluster_sizes",
    "plot_chamber_contribution",
    "plot_shared_soma_reference_diagnostics",
    "plot_filtered_soma_overlay_roi",
    "plot_filtered_soma_overlays",
    "plot_filtered_soma_overlay_tile",
    "plot_filtered_soma_overlay_tiles",
    "plot_cellpose_segmentation_tile",
    "plot_cellpose_segmentation_reference_tiles",
    "preview_cellpose_segmentation_on_tile",
]

DEFAULT_CHANNEL_COLORS = (
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
)


# -------------------------------------------------------------------------
# Shared (mouse-level) reference build diagnostics
# -------------------------------------------------------------------------


def summarize_cluster_sizes(cluster_inds):
    """Per-round table of cluster assignments, dropping ``-1`` rejects.

    ``cluster_inds`` follows the convention from ``scaled_k_means``:
    rejected points get ``-1``. We exclude them from the per-cluster counts
    and report them in a separate ``n_unassigned`` column.
    """
    rows = []
    for round_idx, inds in enumerate(cluster_inds):
        inds = np.asarray(inds)
        kept = inds[inds >= 0]
        nclusters = int(kept.max()) + 1 if kept.size else 0
        nclusters = max(nclusters, 4)
        counts = np.bincount(kept, minlength=nclusters)[:nclusters]
        row = {"round": round_idx + 1}
        for i, v in enumerate(counts):
            row[f"cluster_{i}"] = int(v)
        row["n_unassigned"] = int((inds < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bleedthrough_heatmaps(cluster_means, save_dir=None):
    """One heatmap per round of the (cluster x channel) cluster-means matrix.

    Returns the list of per-round figures plus a combined faceted figure.
    """
    figs = []
    for iround, mean in enumerate(cluster_means):
        fig, ax = plt.subplots(figsize=(3.5, 3))
        im = ax.imshow(mean, cmap="magma", vmin=0)
        for (i, j), v in np.ndenumerate(mean):
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="w" if v < 0.5 * mean.max() else "k",
                fontsize=8,
            )
        ax.set_xlabel("channel")
        ax.set_ylabel("cluster")
        ax.set_title(f"bleedthrough — round {iround + 1}")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        if save_dir is not None:
            fig.savefig(save_dir / f"bleedthrough_round_{iround + 1}.png", dpi=120)
        figs.append(fig)

    nrounds = len(cluster_means)
    if nrounds:
        ncols = min(nrounds, 4)
        nrows = int(np.ceil(nrounds / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3 * ncols, 2.6 * nrows), squeeze=False
        )
        for iround, mean in enumerate(cluster_means):
            ax = axes.flat[iround]
            im = ax.imshow(mean, cmap="magma", vmin=0)
            ax.set_title(f"r{iround + 1}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.flat[nrounds:]:
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.02)
        fig.suptitle("bleedthrough across rounds", y=0.99)
        if save_dir is not None:
            fig.savefig(save_dir / "bleedthrough_all_rounds.png", dpi=120)
        figs.append(fig)
    return figs


def plot_cluster_sizes(cluster_inds, save_dir=None):
    """Grouped bar plot of cluster assignments per round, plus n_unassigned."""
    table = summarize_cluster_sizes(cluster_inds)
    cluster_cols = [c for c in table.columns if c.startswith("cluster_")]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(table)), 3.5))
    width = 0.18
    x = np.arange(len(table))
    for i, col in enumerate(cluster_cols):
        ax.bar(x + i * width, table[col], width=width, label=col)
    ax.bar(
        x + len(cluster_cols) * width,
        table["n_unassigned"],
        width=width,
        label="n_unassigned",
        color="0.5",
    )
    ax.set_xticks(x + (len(cluster_cols) * width) / 2)
    ax.set_xticklabels([f"r{r}" for r in table["round"]])
    ax.set_ylabel("count")
    ax.set_title("Cluster assignment counts per round")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "cluster_sizes.png", dpi=120)
        table.to_csv(save_dir / "cluster_sizes.csv", index=False)
    return fig, table


def plot_chamber_contribution(
    traces_df,
    filtered_traces_df,
    chamber_to_tiles,
    save_dir=None,
):
    """Per-chamber contribution: tile count, total traces, traces past std filter."""
    from ..pipeline.somata import _chamber_label

    rows = []
    for data_path, tiles in chamber_to_tiles.items():
        chamber = _chamber_label(data_path)
        rows.append(
            {
                "chamber": chamber,
                "chamber_data_path": str(data_path),
                "n_reference_tiles": len(tiles),
                "n_total_traces": int((traces_df.get("chamber") == chamber).sum())
                if not traces_df.empty
                else 0,
                "n_passed_std_threshold": int(
                    (filtered_traces_df.get("chamber") == chamber).sum()
                )
                if not filtered_traces_df.empty
                else 0,
            }
        )
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(table) + 3), 3.5))
    width = 0.27
    x = np.arange(len(table))
    ax.bar(x - width, table["n_reference_tiles"], width=width, label="ref tiles")
    ax.bar(x, table["n_total_traces"], width=width, label="all traces")
    ax.bar(
        x + width,
        table["n_passed_std_threshold"],
        width=width,
        label="passed std filter",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(table["chamber"], rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Per-chamber contribution to shared soma reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(save_dir / "chamber_contribution.png", dpi=120)
        table.to_csv(save_dir / "chamber_contribution.csv", index=False)
    return fig, table


def plot_shared_soma_reference_diagnostics(
    result,
    chamber_to_tiles,
    save_dir=None,
):
    """Run all four diagnostic groups for a shared-build result.

    Args:
        result (dict): output of ``compute_soma_reference_from_traces`` /
            ``build_shared_soma_cluster_means``. Must contain
            ``cluster_means``, ``spot_colors``, ``cluster_inds``,
            ``all_traces`` (or ``traces_df``), ``filtered_traces``.
        chamber_to_tiles (dict): mapping from chamber data_path to its
            contributed reference-tile list. Used for per-chamber counts.
        save_dir (Path | None): when not None, write PNGs/CSVs here. The
            directory is created if it does not exist.

    Returns:
        dict: ``{"cluster_scatter": [Figures], "bleedthrough": [Figures],
                  "cluster_sizes": (Figure, DataFrame),
                  "chamber_contribution": (Figure, DataFrame)}``.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    cluster_scatter_figs = plot_clusters(
        result["cluster_means"], result["spot_colors"], result["cluster_inds"]
    )
    if save_dir is not None:
        for fig in cluster_scatter_figs:
            label = fig.get_label() or "clusters_unlabeled"
            fig.savefig(save_dir / f"{label}.png", dpi=120)

    bleedthrough_figs = plot_bleedthrough_heatmaps(
        result["cluster_means"], save_dir=save_dir
    )
    cluster_sizes_fig, cluster_sizes_table = plot_cluster_sizes(
        result["cluster_inds"], save_dir=save_dir
    )
    contribution_fig, contribution_table = plot_chamber_contribution(
        result.get("all_traces", result.get("traces_df", pd.DataFrame())),
        result.get("filtered_traces", pd.DataFrame()),
        chamber_to_tiles,
        save_dir=save_dir,
    )
    return {
        "cluster_scatter": cluster_scatter_figs,
        "bleedthrough": bleedthrough_figs,
        "cluster_sizes": (cluster_sizes_fig, cluster_sizes_table),
        "chamber_contribution": (contribution_fig, contribution_table),
    }


# -------------------------------------------------------------------------
# Post-basecalling per-ROI filtered-soma overlays
# -------------------------------------------------------------------------


def _stitch_rgb_overview(
    data_path,
    roi,
    prefix=None,
    suffix="max",
    downsample_factor=4,
    quantile=99.9,
    vmin=0.05,
    colors=DEFAULT_CHANNEL_COLORS,
):
    """Stitch all channels of one ROI in one shot and return an RGB overview.

    ``stitch_tiles`` is called with ``ich=None`` so all 4 channels come back
    in a single (H, W, C) array. Illumination correction is left off
    (``correct_illumination=False``) — channel-wise normalisation happens in
    :func:`to_rgb` via the per-channel ``quantile`` and ``vmin``. Returns the
    RGB image plus the downsample factor actually applied (so callers can
    scale soma coords).
    """
    ops = load_ops(data_path)
    if prefix is None:
        prefix = ops["reference_prefix"]

    stitched = stitch_tiles(
        data_path,
        prefix=prefix,
        roi=int(roi),
        suffix=suffix,
        ich=None,
        correct_illumination=False,
        register_channels=False,
        allow_quick_estimate=True,
    )
    if downsample_factor and downsample_factor > 1:
        stitched = block_reduce(
            stitched,
            (downsample_factor, downsample_factor, 1),
            np.max,
        )
    stitched = stitched[:, :, np.argsort(ops["camera_order"])]
    vmax = [
        max(float(np.percentile(stitched[:, :, ch], quantile)), 1e-6)
        for ch in range(stitched.shape[2])
    ]
    rgb = to_rgb(stitched, colors=list(colors), vmax=vmax, vmin=vmin)
    return rgb, int(downsample_factor or 1)


def _figure_dir(data_path):
    return (
        get_processed_path(data_path)
        / "figures"
        / "filtered_soma_overlays"
    )


@slurm_it(conda_env="iss-preprocess", slurm_options={"mem": "64GB", "time": "1:00:00"})
def plot_filtered_soma_overlay_roi(
    data_path,
    roi,
    barcode_prefix="barcode_round",
    downsample_factor=4,
    quantile=99.9,
    vmin=0.05,
    survivor_color="red",
    survivor_size=18,
    survivor_linewidth=0.8,
    save_dir=None,
    show=False,
):
    """Render one ROI overlay: stitched RGB + filtered-soma dots.

    ROIs without any surviving somata are still drawn (background only).

    Args:
        data_path (str): chamber data path.
        roi (int): ROI to render.
        barcode_prefix (str): barcode prefix used to locate the filtered soma
            table (``barcode_round`` by default).
        downsample_factor (int): max-pool downsample applied to the stitched
            stack before to_rgb (matches diag_stitching convention). Soma
            coords are scaled by the same factor.
        save_dir (str | Path | None): output directory. Defaults to
            ``processed/{chamber}/figures/filtered_soma_overlays/``.
        show (bool): also call ``plt.show()`` (notebook use). On slurm
            this should stay False.

    Returns:
        pathlib.Path: location of the saved PNG.
    """
    from ..pipeline.somata import load_stitched_soma_calls  # avoid circular import

    if save_dir is None:
        save_dir = _figure_dir(data_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb, applied_downsample = _stitch_rgb_overview(
        data_path,
        roi=roi,
        downsample_factor=downsample_factor,
        quantile=quantile,
        vmin=vmin,
    )

    try:
        filtered_df = load_stitched_soma_calls(
            data_path, barcode_prefix=barcode_prefix, filtered=True
        )
        survivors_df = filtered_df[filtered_df["roi"] == int(roi)]
    except FileNotFoundError:
        survivors_df = None
    n_survivors = 0 if survivors_df is None else len(survivors_df)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    if n_survivors:
        ax.scatter(
            survivors_df["x"] / applied_downsample,
            survivors_df["y"] / applied_downsample,
            facecolors="none",
            edgecolors=survivor_color,
            s=survivor_size,
            linewidths=survivor_linewidth,
        )
    ax.set_title(
        f"{data_path} | ROI {int(roi)} | {n_survivors} survivors "
        f"| ds={applied_downsample}"
    )
    ax.axis("off")
    fig.tight_layout()

    out_path = save_dir / f"roi_{int(roi):03d}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_filtered_soma_overlays(
    data_path,
    barcode_prefix="barcode_round",
    downsample_factor=4,
    quantile=99.9,
    vmin=0.05,
    use_slurm=True,
    slurm_folder=None,
    use_rois=None,
):
    """Submit one ``plot_filtered_soma_overlay_roi`` job per ROI in the chamber.

    Mirrors the ``plot_overview_images`` fan-out pattern in
    :mod:`iss_preprocess.diagnostics.diag_stitching`.

    Returns:
        list: per-ROI return values (slurm job IDs when ``use_slurm=True``,
        otherwise the saved PNG paths).
    """
    if use_rois is None:
        roi_dims = get_roi_dimensions(data_path)
        use_rois = [int(r) for r in roi_dims[:, 0]]
    if slurm_folder is None:
        slurm_folder = (
            Path.home()
            / "slurm_logs"
            / data_path
            / "filtered_soma_overlays"
        )
    slurm_folder = Path(slurm_folder)
    slurm_folder.mkdir(parents=True, exist_ok=True)

    results = []
    for roi in use_rois:
        results.append(
            plot_filtered_soma_overlay_roi(
                data_path,
                roi=int(roi),
                barcode_prefix=barcode_prefix,
                downsample_factor=downsample_factor,
                quantile=quantile,
                vmin=vmin,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=f"filtered_soma_overlay_roi_{int(roi)}",
            )
        )
    return results


# -------------------------------------------------------------------------
# Cellpose segmentation QC (run inline in the notebook on reference tiles)
# -------------------------------------------------------------------------


def _load_segmentation_tile_image(
    data_path,
    tile_coors,
    segmentation_prefix,
    channels,
    use_raw_stack,
):
    """Mirror the image-loading dispatch from run_cpsam_2d_segmentation /
    run_cellpose_segmentation so the diagnostic helpers see the same pixels
    the segmenter saw.
    """
    from ..io.load import load_tile_by_coors
    from ..pipeline.register import load_and_register_tile

    ops = load_ops(data_path)
    if use_raw_stack:
        from ..pipeline.segment import get_stack_for_cellpose

        return get_stack_for_cellpose(
            data_path,
            prefix=segmentation_prefix,
            tile_coors=tile_coors,
            use_raw_stack=True,
        )
    if len(channels) == 1:
        img = load_tile_by_coors(
            data_path,
            tile_coors=tile_coors,
            suffix=ops.get("projection_for_segmentation", "max"),
            prefix=segmentation_prefix,
            correct_illumination=True,
        )
        return img[..., channels]
    img, _ = load_and_register_tile(data_path, tile_coors, segmentation_prefix)
    if channels:
        img = img[..., channels]
    return img


def _plot_image_with_mask_boundaries(
    img,
    masks,
    title,
    quantile,
    boundary_color,
    show,
):
    """Shared renderer: one panel per image channel, mask boundaries overlaid."""
    from skimage.segmentation import find_boundaries

    boundaries = find_boundaries(masks)
    n_cells = int(masks.max())
    n_channels = img.shape[-1] if img.ndim == 3 else 1
    fig, axes = plt.subplots(1, n_channels, figsize=(8 * n_channels, 8), squeeze=False)
    for ch in range(n_channels):
        plane = img[..., ch] if img.ndim == 3 else img
        vmax = max(float(np.percentile(plane, quantile)), 1e-6)
        ax = axes[0, ch]
        ax.imshow(plane, cmap="gray", vmin=0, vmax=vmax)
        ax.imshow(boundaries, cmap="Reds", alpha=boundaries.astype(float))
        ax.set_title(f"{title} | ch{ch} | {n_cells} masks", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def preview_cellpose_segmentation_on_tile(
    data_path,
    tile_coors,
    use_raw_stack=False,
    quantile=99.9,
    boundary_color="orange",
    show=True,
    **cellpose_overrides,
):
    """Run cellpose on one tile with overridable kwargs and render the result.

    Loads the tile image via the same dispatch the segmenter uses, then calls
    :func:`iss_preprocess.segment.cells.cellpose_segmentation` with ops
    defaults merged with ``cellpose_overrides``. No mask is written to disk —
    intended for iterative hyperparameter tuning in the notebook before
    committing to a chamber-wide run.

    Recognised overrides (any unknown kwargs are forwarded to
    ``CellposeModel.eval``): ``diameter``, ``flow_threshold``,
    ``cellprob_threshold``, ``min_pix``, ``dilate_pix``, ``rescale``,
    ``model_type``, ``pretrained_model``, ``normalize``.

    Returns:
        tuple: ``(masks, figure, effective_params)``.
    """
    from ..pipeline.somata import validate_soma_workflow_config
    from ..segment.cells import cellpose_segmentation

    cfg = validate_soma_workflow_config(data_path)
    segmentation_prefix = cfg["segmentation_acquisition"]
    tile_coors = (int(tile_coors[0]), int(tile_coors[1]), int(tile_coors[2]))
    ops = load_ops(data_path)
    channels = list(ops.get("cellpose_channels") or [])

    img = _load_segmentation_tile_image(
        data_path, tile_coors, segmentation_prefix, channels, use_raw_stack
    )
    if img.ndim == 4 and not use_raw_stack:
        img = img.max(axis=-1)

    normalisation = ops.get("cellpose_normalise") or {}
    if isinstance(normalisation, dict) and "lowhigh" in normalisation:
        normalisation = {**normalisation, "lowhigh": np.array(normalisation["lowhigh"])}

    params = {
        "flow_threshold": ops.get("cellpose_flow_threshold", 0.4),
        "cellprob_threshold": ops.get("cellpose_cellprob_threshold", 0.0),
        "min_pix": ops.get("cellpose_min_pix", 0),
        "dilate_pix": ops.get("cellpose_dilate_pix", 0),
        "diameter": ops.get("cellpose_diameter"),
        "rescale": ops.get("cellpose_rescale"),
        "model_type": ops.get("cellpose_model_type", "cyto3"),
        "pretrained_model": ops.get("pretrained_model")
        or ops.get("cellpose_pretrained_model"),
        "normalize": normalisation,
    }
    params.update(cellpose_overrides)
    if params.get("pretrained_model"):
        from ..io import get_processed_path

        params["pretrained_model"] = get_processed_path(params["pretrained_model"])

    masks = cellpose_segmentation(img, use_gpu=False, do_3D=False, **params)
    n_cells = int(masks.max())
    print(f"{data_path} | tile {tile_coors} | {n_cells} masks")
    fig = _plot_image_with_mask_boundaries(
        img,
        masks,
        title=f"{data_path} | tile {tile_coors} | preview",
        quantile=quantile,
        boundary_color=boundary_color,
        show=show,
    )
    return masks, fig, params


def plot_cellpose_segmentation_tile(
    data_path,
    tile_coors,
    use_raw_stack=False,
    quantile=99.9,
    boundary_color="orange",
    show=True,
):
    """Overlay per-tile cellpose mask boundaries on the segmentation-acquisition
    image (NeuN by default). Intended for quick visual evaluation of cellpose
    output on a small set of reference tiles before running atlas / basecalling.

    Reads the same per-tile mask file produced by
    :func:`iss_preprocess.pipeline.somata.segment_soma_masks`
    (``processed/cells/{segmentation_prefix}_masks_{roi}_{tilex}_{tiley}.npy``
    when ``use_raw_stack=False``, ``processed/cells/raw_masks/...`` otherwise).

    Args:
        data_path (str): chamber data path.
        tile_coors (tuple): ``(roi, tilex, tiley)``.
        use_raw_stack (bool): must match the value passed to
            ``segment_soma_masks``. Sindbis notebook uses False (2D path).
        quantile (float): per-channel intensity clip for display.
        boundary_color: mask-boundary colour.
        show (bool): call ``plt.show()`` after rendering.

    Returns:
        matplotlib.figure.Figure
    """
    from ..io import get_processed_path
    from ..pipeline.somata import validate_soma_workflow_config

    cfg = validate_soma_workflow_config(data_path)
    segmentation_prefix = cfg["segmentation_acquisition"]
    tile_coors = (int(tile_coors[0]), int(tile_coors[1]), int(tile_coors[2]))
    ops = load_ops(data_path)
    channels = list(ops.get("cellpose_channels") or [])

    cells_dir = get_processed_path(data_path) / "cells"
    if use_raw_stack:
        cells_dir = cells_dir / "raw_masks"
    mask_path = (
        cells_dir
        / f"{segmentation_prefix}_masks_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npy"
    )
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Cellpose mask not found for {data_path} {tile_coors}: {mask_path}. "
            "Run segment_soma_masks first."
        )
    masks = np.load(mask_path)
    if masks.ndim == 3:
        # raw_stack 3D masks: collapse to 2D for display
        masks = masks.max(axis=-1)

    img = _load_segmentation_tile_image(
        data_path, tile_coors, segmentation_prefix, channels, use_raw_stack
    )
    if img.ndim == 4:
        img = img.max(axis=-1)
    return _plot_image_with_mask_boundaries(
        img,
        masks,
        title=f"{data_path} | tile {tile_coors} | {segmentation_prefix}",
        quantile=quantile,
        boundary_color=boundary_color,
        show=show,
    )


def plot_cellpose_segmentation_reference_tiles(
    data_path,
    tiles=None,
    n_tiles=None,
    use_raw_stack=False,
    quantile=99.9,
    show=True,
):
    """Render cellpose-segmentation overlays for a small set of reference tiles.

    Defaults to ``ops["barcode_soma_reference_tiles"]`` (optionally capped at
    ``n_tiles``). Pass ``tiles`` explicitly to override.

    Returns:
        list[matplotlib.figure.Figure]
    """
    if tiles is None:
        ops = load_ops(data_path)
        tiles = ops.get("barcode_soma_reference_tiles") or []
    if n_tiles is not None:
        tiles = list(tiles)[:n_tiles]
    figs = []
    for tile_coors in tiles:
        figs.append(
            plot_cellpose_segmentation_tile(
                data_path,
                tile_coors=tuple(tile_coors),
                use_raw_stack=use_raw_stack,
                quantile=quantile,
                show=show,
            )
        )
    return figs


def _tile_figure_dir(data_path):
    return (
        get_processed_path(data_path)
        / "figures"
        / "filtered_soma_overlay_tiles"
    )


def _load_tile_rgb_for_round(
    data_path,
    tile_coors,
    barcode_prefix,
    barcode_round_to_show,
    barcode_rounds,
    quantile,
    vmin,
    colors,
):
    """Single-round per-tile RGB (registered, channel-reordered, percentile-normed)."""
    from ..pipeline.register import load_and_register_sequencing_tile

    ops = load_ops(data_path)
    stack, _ = load_and_register_sequencing_tile(
        data_path,
        tile_coors,
        prefix=barcode_prefix,
        suffix="max",
        filter_r=None,
        correct_channels=False,
        corrected_shifts="best",
        correct_illumination=True,
        nrounds=barcode_rounds,
        specific_rounds=[barcode_round_to_show],
        bad_pixels_per_round=False,
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), :]
    plane = stack[:, :, :, 0]
    vmax = [
        max(float(np.percentile(plane[:, :, ch], quantile)), 1e-6)
        for ch in range(plane.shape[2])
    ]
    return to_rgb(plane, colors=list(colors), vmax=vmax, vmin=vmin)


@slurm_it(conda_env="iss-preprocess", slurm_options={"mem": "16GB", "time": "0:30:00"})
def plot_filtered_soma_overlay_tile(
    data_path,
    roi,
    tilex,
    tiley,
    barcode_prefix="barcode_round",
    barcode_round_to_show=2,
    quantile=99.9,
    vmin=0.05,
    survivor_color="orange",
    survivor_size=20,
    survivor_linewidth=2.0,
    save_dir=None,
    show=False,
):
    """Render one per-tile overlay: barcode-round RGB + atlas mask boundaries
    + survivor dots in local (``x_in_tile``, ``y_in_tile``) coordinates.

    Tiles without surviving somata are skipped (no PNG written). The
    orchestrator only submits jobs for tiles with at least one survivor, so
    in normal use this guard never fires.

    Returns:
        pathlib.Path | None: location of the saved PNG, or None if the tile
        had no survivors.
    """
    from skimage.segmentation import find_boundaries

    from ..pipeline.somata import (  # avoid circular import
        load_soma_atlas_tile,
        load_stitched_soma_calls,
        validate_soma_workflow_config,
    )

    tile_coors = (int(roi), int(tilex), int(tiley))
    cfg = validate_soma_workflow_config(data_path)

    try:
        filtered_df = load_stitched_soma_calls(
            data_path, barcode_prefix=barcode_prefix, filtered=True
        )
    except FileNotFoundError:
        return None
    survivors_df = filtered_df[
        (filtered_df["roi"] == int(roi))
        & (filtered_df["tilex"] == int(tilex))
        & (filtered_df["tiley"] == int(tiley))
    ]
    if survivors_df.empty:
        return None

    if save_dir is None:
        save_dir = _tile_figure_dir(data_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb = _load_tile_rgb_for_round(
        data_path,
        tile_coors,
        barcode_prefix=barcode_prefix,
        barcode_round_to_show=barcode_round_to_show,
        barcode_rounds=cfg["barcode_rounds"],
        quantile=quantile,
        vmin=vmin,
        colors=DEFAULT_CHANNEL_COLORS,
    )
    masks = load_soma_atlas_tile(
        data_path,
        tile_coors=tile_coors,
        expected_reference_prefix=cfg["reference_prefix"],
        expected_corrected_shifts=cfg["corrected_shifts"],
    )
    boundaries = find_boundaries(masks)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.imshow(boundaries, cmap="Reds", alpha=boundaries.astype(float))
    ax.scatter(
        survivors_df["x_in_tile"],
        survivors_df["y_in_tile"],
        facecolors="none",
        edgecolors=survivor_color,
        s=survivor_size,
        linewidths=survivor_linewidth,
    )
    ax.set_title(
        f"{data_path} | tile {tile_coors} | round {barcode_round_to_show} | "
        f"{len(survivors_df)} survivors"
    )
    ax.axis("off")
    fig.tight_layout()

    out_path = save_dir / f"tile_{int(roi):03d}_{int(tilex):03d}_{int(tiley):03d}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_filtered_soma_overlay_tiles(
    data_path,
    barcode_prefix="barcode_round",
    barcode_round_to_show=2,
    quantile=99.9,
    vmin=0.05,
    use_slurm=True,
    slurm_folder=None,
    tile_coors_list=None,
):
    """Submit one ``plot_filtered_soma_overlay_tile`` job per tile with survivors.

    Mirrors the per-ROI fan-out from :func:`plot_filtered_soma_overlays`. The
    set of tiles is derived from the chamber's filtered soma-call table; tiles
    without survivors are not plotted (no uniformity guarantee here — there
    are far too many empty tiles to be useful).

    Args:
        tile_coors_list (list[tuple] | None): optional explicit list of
            ``(roi, tilex, tiley)`` triples to render. Overrides the
            auto-derived list from filtered calls.

    Returns:
        list: per-tile return values (slurm job IDs when ``use_slurm=True``,
        otherwise the saved PNG paths or ``None`` for skipped tiles).
    """
    from ..pipeline.somata import load_stitched_soma_calls  # avoid circular import

    if tile_coors_list is None:
        try:
            filtered_df = load_stitched_soma_calls(
                data_path, barcode_prefix=barcode_prefix, filtered=True
            )
        except FileNotFoundError:
            print(f"{data_path}: no filtered soma table; nothing to plot")
            return []
        if filtered_df.empty:
            print(f"{data_path}: filtered soma table is empty; nothing to plot")
            return []
        tile_coors_list = [
            (int(roi), int(tilex), int(tiley))
            for (roi, tilex, tiley), _ in filtered_df.groupby(
                ["roi", "tilex", "tiley"]
            ).size().sort_values(ascending=False).items()
        ]

    if slurm_folder is None:
        slurm_folder = (
            Path.home()
            / "slurm_logs"
            / data_path
            / "filtered_soma_overlay_tiles"
        )
    slurm_folder = Path(slurm_folder)
    slurm_folder.mkdir(parents=True, exist_ok=True)

    results = []
    for roi, tilex, tiley in tile_coors_list:
        results.append(
            plot_filtered_soma_overlay_tile(
                data_path,
                roi=int(roi),
                tilex=int(tilex),
                tiley=int(tiley),
                barcode_prefix=barcode_prefix,
                barcode_round_to_show=barcode_round_to_show,
                quantile=quantile,
                vmin=vmin,
                use_slurm=use_slurm,
                slurm_folder=slurm_folder,
                scripts_name=(
                    f"filtered_soma_overlay_tile_{int(roi)}_"
                    f"{int(tilex)}_{int(tiley)}"
                ),
            )
        )
    return results
