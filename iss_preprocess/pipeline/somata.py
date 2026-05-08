from itertools import product

import numpy as np
import pandas as pd

from ..call import extract_traces_somata, get_cluster_means
from ..io import get_processed_path, load_ops
from ..io.load import get_roi_dimensions
from ..image.utils import highpass_stack
from .align_spots_and_cells import stitch_cell_dataframes
from .core import batch_process_tiles
from .register import load_register_and_fill_tile, load_register_and_fill_tile_mask
from .segment import (
    find_edge_touching_masks,
    remove_all_duplicate_masks,
    segment_all_tiles,
)
from .sequencing import setup_soma_barcode_calling

DEFAULT_SOMA_BARCODE_PREFIX = "barcode_round"
REQUIRED_SOMA_OPS_KEYS = (
    "segmentation_acquisition",
    "reference_prefix",
    "corrected_shifts",
    "barcode_rounds",
    "barcode_soma_reference_tiles",
    "soma_std_threshold_clustering",
    "somata_cluster_score_thresh",
    "initial_cluster_means",
)

__all__ = [
    "DEFAULT_SOMA_BARCODE_PREFIX",
    "REQUIRED_SOMA_OPS_KEYS",
    "apply_soma_qc_filters",
    "basecall_somata_tiles",
    "deduplicate_soma_masks",
    "enumerate_soma_candidate_tiles",
    "extract_soma_reference_traces",
    "get_soma_output_paths",
    "get_soma_workflow_config",
    "load_soma_reference_tile",
    "load_stitched_soma_calls",
    "missing_soma_workflow_keys",
    "compute_soma_reference_from_tiles",
    "compute_soma_reference_from_traces",
    "save_filtered_soma_calls",
    "segment_soma_masks",
    "summarize_soma_qc_threshold_grid",
    "summarize_soma_reference_tiles",
    "setup_soma_calling_reference",
    "stitch_soma_calls",
    "validate_soma_workflow_config",
]


def get_soma_workflow_config(data_path):
    """Return the subset of ops used by the Sindbis soma workflow."""
    ops = load_ops(data_path)
    return {key: ops.get(key) for key in REQUIRED_SOMA_OPS_KEYS}


def missing_soma_workflow_keys(data_path):
    """Return ops keys that are missing or null for the Sindbis soma workflow."""
    ops = load_ops(data_path)
    return [key for key in REQUIRED_SOMA_OPS_KEYS if key not in ops or ops[key] is None]


def validate_soma_workflow_config(data_path):
    """Raise a helpful error when required Sindbis soma ops keys are missing."""
    missing = missing_soma_workflow_keys(data_path)
    if missing:
        missing_fmt = ", ".join(missing)
        raise KeyError(
            "Missing Sindbis soma ops keys: "
            f"{missing_fmt}. Add them to the dataset ops before running the soma workflow."
        )
    return get_soma_workflow_config(data_path)


def get_soma_output_paths(data_path, barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX):
    """Collect the main files produced by the Sindbis soma workflow."""
    processed_path = get_processed_path(data_path)
    cells_dir = processed_path / "cells"
    stitched_dir = cells_dir / f"{barcode_prefix}_cells"
    return {
        "processed_path": processed_path,
        "cells_dir": cells_dir,
        "stitched_dir": stitched_dir,
        "somata_traces": processed_path / "somata_traces_df.pkl",
        "somata_cluster_means": processed_path / "somata_barcode_cluster_means.npy",
        "somata_reference_barcodes": processed_path / "somata_reference_barcodes.npz",
        "stitched_calls": stitched_dir / f"{barcode_prefix}_df_corrected.pkl",
        "filtered_calls": stitched_dir / f"{barcode_prefix}_df_filtered.pkl",
    }


def enumerate_soma_candidate_tiles(data_path, use_rois=None):
    """Enumerate chamber tiles that can be considered as soma reference candidates."""
    roi_dims = get_roi_dimensions(data_path)
    if use_rois is None:
        use_rois = roi_dims[:, 0]
    tiles = []
    for roi in use_rois:
        _, nx, ny = roi_dims[roi_dims[:, 0] == roi][0]
        tiles.extend((int(roi), int(tx), int(ty)) for tx in range(nx + 1) for ty in range(ny + 1))
    return tiles


def segment_soma_masks(
    data_path,
    segmentation_prefix=None,
    use_gpu=False,
    use_rois=None,
    tile_list=None,
    rerun_cellpose=False,
    use_slurm=True,
    use_raw_stack=False,
):
    """Run the tile-wise soma segmentation stage."""
    cfg = validate_soma_workflow_config(data_path)
    if segmentation_prefix is None:
        segmentation_prefix = cfg["segmentation_acquisition"]
    return segment_all_tiles(
        data_path,
        prefix=segmentation_prefix,
        use_raw_stack=use_raw_stack,
        use_gpu=use_gpu,
        use_rois=use_rois,
        tile_list=tile_list,
        rerun_cellpose=rerun_cellpose,
        use_slurm=use_slurm,
    )


def deduplicate_soma_masks(
    data_path,
    segmentation_prefix=None,
    upper_overlap_thresh=0.3,
    **kwargs,
):
    """Remove duplicate soma masks across tile overlaps."""
    cfg = validate_soma_workflow_config(data_path)
    if segmentation_prefix is None:
        segmentation_prefix = cfg["segmentation_acquisition"]
    return remove_all_duplicate_masks(
        data_path,
        segmentation_prefix,
        upper_overlap_thresh=upper_overlap_thresh,
        **kwargs,
    )


def extract_soma_reference_traces(
    data_path,
    tile_coors,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    highpass_cutoff=15.0,
    highpass_order=2,
    highpass_pad=100,
    edge_mask_border=10,
):
    """Extract soma traces from one tile in local barcode reference-tile coordinates."""
    cfg = validate_soma_workflow_config(data_path)
    stack, bad_pixels, masks = load_soma_reference_tile(
        data_path,
        tile_coors=tile_coors,
        segmentation_prefix=segmentation_prefix,
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
        barcode_prefix=barcode_prefix,
        specific_rounds=None,
        zero_fill_output=True,
    )
    ops = load_ops(data_path)
    stack = stack.copy()
    stack[bad_pixels, ...] = 0
    stack = stack[:, :, np.argsort(ops["camera_order"]), :]
    filtered_stack = highpass_stack(
        stack,
        cutoff=highpass_cutoff,
        order=highpass_order,
        pad=highpass_pad,
    )
    masks = masks.copy()
    masks, edge_touching_labels = find_edge_touching_masks(
        masks, border_width=edge_mask_border
    )
    traces_df = extract_traces_somata(filtered_stack, masks)
    tile_id = "_".join(map(str, tile_coors))
    traces_df["tile_of_origin"] = tile_id
    traces_df["roi"] = int(tile_coors[0])
    traces_df["tilex"] = int(tile_coors[1])
    traces_df["tiley"] = int(tile_coors[2])
    traces_df["bad_pixel_fraction"] = float(np.mean(bad_pixels))
    traces_df["edge_touching_labels_removed"] = len(edge_touching_labels)
    return traces_df


def summarize_soma_reference_tiles(
    data_path,
    tile_list=None,
    use_rois=None,
    std_threshold=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
):
    """Summarize candidate tiles for soma bleedthrough / cluster-mean estimation."""
    cfg = validate_soma_workflow_config(data_path)
    if std_threshold is None:
        std_threshold = cfg["soma_std_threshold_clustering"]
    if tile_list is None:
        tile_list = enumerate_soma_candidate_tiles(data_path, use_rois=use_rois)

    rows = []
    for tile_coors in tile_list:
        traces_df = extract_soma_reference_traces(
            data_path,
            tile_coors=tile_coors,
            barcode_prefix=barcode_prefix,
        )
        if len(traces_df):
            trace_mean_abs = traces_df["trace"].apply(lambda tr: float(np.mean(np.abs(tr))))
            rows.append(
                {
                    "tile_of_origin": traces_df["tile_of_origin"].iloc[0],
                    "roi": int(tile_coors[0]),
                    "tilex": int(tile_coors[1]),
                    "tiley": int(tile_coors[2]),
                    "mask_count": int(len(traces_df)),
                    "n_std_ge_threshold": int((traces_df["std"] >= std_threshold).sum()),
                    "std_threshold": float(std_threshold),
                    "std_median": float(np.nanmedian(traces_df["std"])),
                    "std_p90": float(np.nanpercentile(traces_df["std"], 90)),
                    "std_p95": float(np.nanpercentile(traces_df["std"], 95)),
                    "area_median": float(np.nanmedian(traces_df["area"])),
                    "area_sum": float(np.nansum(traces_df["area"])),
                    "trace_mean_abs_median": float(np.nanmedian(trace_mean_abs)),
                    "trace_mean_abs_p90": float(np.nanpercentile(trace_mean_abs, 90)),
                    "bad_pixel_fraction": float(traces_df["bad_pixel_fraction"].iloc[0]),
                    "edge_touching_labels_removed": int(
                        traces_df["edge_touching_labels_removed"].iloc[0]
                    ),
                }
            )
        else:
            rows.append(
                {
                    "tile_of_origin": "_".join(map(str, tile_coors)),
                    "roi": int(tile_coors[0]),
                    "tilex": int(tile_coors[1]),
                    "tiley": int(tile_coors[2]),
                    "mask_count": 0,
                    "n_std_ge_threshold": 0,
                    "std_threshold": float(std_threshold),
                    "std_median": np.nan,
                    "std_p90": np.nan,
                    "std_p95": np.nan,
                    "area_median": np.nan,
                    "area_sum": 0.0,
                    "trace_mean_abs_median": np.nan,
                    "trace_mean_abs_p90": np.nan,
                    "bad_pixel_fraction": np.nan,
                    "edge_touching_labels_removed": np.nan,
                }
            )

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(
            by=["n_std_ge_threshold", "std_p90", "mask_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary


def load_soma_reference_tile(
    data_path,
    tile_coors,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    suffix="max",
    specific_rounds=(2,),
    correct_channels="round1_only",
    correct_illumination=True,
    edge=10,
    mid=5,
    zero_fill_output=False,
):
    """Load one barcode tile and its soma masks in the local reference-tile frame."""
    cfg = validate_soma_workflow_config(data_path)
    if segmentation_prefix is None:
        segmentation_prefix = cfg["segmentation_acquisition"]
    if reference_prefix is None:
        reference_prefix = cfg["reference_prefix"]
    if corrected_shifts is None:
        corrected_shifts = cfg["corrected_shifts"]

    masks = load_register_and_fill_tile_mask(
        data_path,
        tile_coors,
        prefix=segmentation_prefix,
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
    )
    stack, bad_pixels = load_register_and_fill_tile(
        data_path,
        tile_coors,
        prefix=barcode_prefix,
        suffix=suffix,
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
        correct_channels=correct_channels,
        correct_illumination=correct_illumination,
        filter_r=None,
        specific_rounds=specific_rounds,
        nrounds=cfg["barcode_rounds"],
        edge=edge,
        mid=mid,
        zero_fill_output=zero_fill_output,
    )
    return stack, bad_pixels, masks


def compute_soma_reference_from_tiles(
    data_path,
    reference_tiles,
    std_threshold=None,
    cluster_score_thresh=None,
    initial_cluster_mean=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    save=False,
):
    """Compute soma cluster means from user-selected tiles and thresholds."""
    cfg = validate_soma_workflow_config(data_path)
    all_tile_dfs = []
    for tile_coors in reference_tiles:
        all_tile_dfs.append(
            extract_soma_reference_traces(
                data_path,
                tile_coors=tile_coors,
                barcode_prefix=barcode_prefix,
            )
        )

    traces_df = (
        pd.concat(all_tile_dfs, axis=0, ignore_index=True) if all_tile_dfs else pd.DataFrame()
    )
    result = compute_soma_reference_from_traces(
        traces_df,
        std_threshold=std_threshold,
        cluster_score_thresh=cluster_score_thresh,
        initial_cluster_mean=cfg["initial_cluster_means"]
        if initial_cluster_mean is None
        else initial_cluster_mean,
    )
    result["reference_tiles"] = [tuple(tile) for tile in reference_tiles]

    if save:
        outputs = get_soma_output_paths(data_path, barcode_prefix=barcode_prefix)
        np.save(outputs["somata_cluster_means"], result["cluster_means"])
        np.savez(
            outputs["somata_reference_barcodes"],
            spot_colors=result["spot_colors"],
            cluster_inds=result["cluster_inds"],
        )
        traces_df.to_pickle(outputs["somata_traces"])
        result["saved_paths"] = {
            "somata_cluster_means": outputs["somata_cluster_means"],
            "somata_reference_barcodes": outputs["somata_reference_barcodes"],
            "somata_traces": outputs["somata_traces"],
        }

    return result


def compute_soma_reference_from_traces(
    traces_df,
    std_threshold,
    cluster_score_thresh,
    initial_cluster_mean,
):
    """Compute soma cluster means from an already extracted trace table."""
    if initial_cluster_mean is None:
        raise ValueError("initial_cluster_mean must be provided")
    initial_cluster_mean = np.asarray(initial_cluster_mean)
    filtered_df = traces_df.loc[traces_df["std"] > std_threshold].copy()
    if filtered_df.empty:
        raise ValueError(
            "No soma traces passed the selected std threshold. "
            "Pick different reference tiles or lower std_threshold."
        )
    cluster_means, spot_colors, cluster_inds = get_cluster_means(
        filtered_df,
        score_thresh=cluster_score_thresh,
        initial_cluster_mean=initial_cluster_mean,
    )
    return {
        "std_threshold": float(std_threshold),
        "cluster_score_thresh": float(cluster_score_thresh),
        "all_traces": traces_df,
        "filtered_traces": filtered_df,
        "cluster_means": cluster_means,
        "spot_colors": spot_colors,
        "cluster_inds": cluster_inds,
    }


def setup_soma_calling_reference(data_path, reload=False, **kwargs):
    """Build soma-specific cluster means / bleedthrough reference."""
    validate_soma_workflow_config(data_path)
    return setup_soma_barcode_calling(data_path, reload=reload, **kwargs)


def basecall_somata_tiles(data_path):
    """Submit soma basecalling jobs for all tiles in a chamber."""
    validate_soma_workflow_config(data_path)
    return batch_process_tiles(data_path, "basecall_somata_tile")


def stitch_soma_calls(
    data_path,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    ref_prefix=None,
    **kwargs,
):
    """Stitch per-tile soma calls into ROI-global reference coordinates."""
    validate_soma_workflow_config(data_path)
    return stitch_cell_dataframes(
        data_path,
        prefix=barcode_prefix,
        ref_prefix=ref_prefix,
        sindbis=True,
        **kwargs,
    )


def load_stitched_soma_calls(
    data_path,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    filtered=False,
):
    """Load the stitched soma-call dataframe."""
    paths = get_soma_output_paths(data_path, barcode_prefix=barcode_prefix)
    target = paths["filtered_calls"] if filtered else paths["stitched_calls"]
    return pd.read_pickle(target)


def apply_soma_qc_filters(df, thresholds):
    """Apply a dict of lower-bound QC thresholds to a soma dataframe."""
    filtered = df.copy()
    for column, threshold in thresholds.items():
        filtered = filtered[filtered[column] > threshold]
    return filtered.copy()


def summarize_soma_qc_threshold_grid(df, threshold_grid):
    """Evaluate several QC threshold combinations on a stitched soma dataframe."""
    if not threshold_grid:
        raise ValueError("threshold_grid must be a non-empty dict of column -> list")

    grid_keys = list(threshold_grid)
    rows = []
    for combo in product(*(threshold_grid[key] for key in grid_keys)):
        thresholds = dict(zip(grid_keys, combo))
        filtered = apply_soma_qc_filters(df, thresholds)
        barcode_counts = filtered["bases"].value_counts() if "bases" in filtered else pd.Series(dtype=int)
        n_duplicate_rows = int(barcode_counts[barcode_counts > 1].sum()) if len(barcode_counts) else 0
        rows.append(
            {
                **thresholds,
                "n_somata": int(len(filtered)),
                "n_unique_barcodes": int(filtered["bases"].nunique()) if "bases" in filtered else 0,
                "n_duplicate_barcodes": int((barcode_counts > 1).sum()) if len(barcode_counts) else 0,
                "duplicate_row_fraction": float(n_duplicate_rows / len(filtered))
                if len(filtered)
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["n_somata", "n_unique_barcodes"], ascending=[False, False]
    ).reset_index(drop=True)


def save_filtered_soma_calls(
    data_path,
    filtered_df,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    suffix="filtered",
):
    """Save a filtered soma-call dataframe next to the stitched table."""
    target = get_soma_output_paths(data_path, barcode_prefix=barcode_prefix)[
        "stitched_dir"
    ]
    target.mkdir(exist_ok=True, parents=True)
    fname = target / f"{barcode_prefix}_df_{suffix}.pkl"
    filtered_df.to_pickle(fname)
    return fname
