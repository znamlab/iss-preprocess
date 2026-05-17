import json
from itertools import product
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import watershed
from skimage.transform import AffineTransform, warp
from znamutils import slurm_it

from ..call.call import extract_traces_somata, get_cluster_means
from ..io import get_mouse_path, get_processed_path, load_ops
from ..io.load import get_roi_dimensions, load_mask_by_coors
from ..image.utils import highpass_stack
from .align_spots_and_cells import stitch_cell_dataframes
from .core import batch_process_tiles
from .register import (
    _get_mask_registration_matrix,
    load_register_and_fill_tile,
    load_register_and_fill_tile_mask,
)
from .segment import (
    segment_all_tiles,
)
from .stitch import calculate_tile_positions

SOMA_ATLAS_DTYPE = np.uint32
SOMA_ATLAS_OWNER_DTYPE = np.dtype(
    [
        ("gid", "u4"),
        ("owner_tx", "i4"),
        ("owner_ty", "i4"),
        ("area", "u4"),
        ("centroid_y", "f4"),
        ("centroid_x", "f4"),
    ]
)
SOMA_ATLAS_PROVENANCE_DTYPE = np.dtype(
    [
        ("gid", "u4"),
        ("source_roi", "i4"),
        ("source_tx", "i4"),
        ("source_ty", "i4"),
        ("source_label", "u4"),
        ("source_area", "u4"),
    ]
)

DEFAULT_SOMA_BARCODE_PREFIX = "barcode_round"
SOMA_TRACE_CACHE_META_ATTR = "soma_trace_cache_meta"
SOMA_TRACE_CACHE_SCHEMA_VERSION = 1
DEFAULT_SOMA_TRACE_HIGHPASS_CUTOFF = 15.0
DEFAULT_SOMA_TRACE_HIGHPASS_ORDER = 2
DEFAULT_SOMA_TRACE_HIGHPASS_PAD = 100
DEFAULT_SOMA_STD_THRESHOLD = 1.0
SHARED_SOMA_OPS_FLAG = "use_shared_soma_cluster_means"
_SHARED_OPTIONAL_SOMA_OPS_KEYS = (
    "barcode_soma_reference_tiles",
    "soma_std_threshold_clustering",
    "somata_cluster_score_thresh",
)
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
    "DEFAULT_SOMA_STD_THRESHOLD",
    "REQUIRED_SOMA_OPS_KEYS",
    "SHARED_SOMA_OPS_FLAG",
    "SOMA_ATLAS_DTYPE",
    "SOMA_ATLAS_OWNER_DTYPE",
    "SOMA_ATLAS_PROVENANCE_DTYPE",
    "apply_soma_qc_filters",
    "basecall_somata_tiles",
    "build_shared_soma_cluster_means",
    "build_soma_atlas",
    "build_soma_atlases",
    "deduplicate_soma_masks",
    "collect_soma_trace_tile_stats",
    "enumerate_soma_candidate_tiles",
    "extract_soma_reference_traces",
    "extract_soma_trace_tile",
    "extract_soma_trace_tiles",
    "get_soma_atlas_paths",
    "get_soma_output_paths",
    "get_soma_trace_cache_paths",
    "get_soma_workflow_config",
    "load_soma_atlas_tile",
    "load_soma_reference_tile",
    "load_soma_trace_tile",
    "load_soma_traces_for_chambers",
    "load_soma_traces_for_tiles",
    "load_stitched_soma_calls",
    "missing_soma_workflow_keys",
    "compute_soma_reference_from_tiles",
    "compute_soma_reference_from_traces",
    "save_filtered_soma_calls",
    "segment_soma_masks",
    "summarize_soma_qc_threshold_grid",
    "summarize_soma_reference_tiles",
    "summarize_soma_reference_tiles_across_chambers",
    "setup_soma_calling_reference",
    "stitch_soma_calls",
    "validate_soma_workflow_config",
]


def _required_soma_ops_keys(ops):
    """The required soma ops keys for this chamber, given its ops.

    When ``use_shared_soma_cluster_means: true`` is set, the threshold and
    reference-tile keys become optional on this chamber: the shared build
    enforces "exactly one chamber declares each threshold" mouse-wide, so
    every other chamber in the pool must omit them.
    """
    if ops.get(SHARED_SOMA_OPS_FLAG, False):
        return tuple(
            key
            for key in REQUIRED_SOMA_OPS_KEYS
            if key not in _SHARED_OPTIONAL_SOMA_OPS_KEYS
        )
    return REQUIRED_SOMA_OPS_KEYS


def get_soma_workflow_config(data_path):
    """Return the subset of ops used by the Sindbis soma workflow."""
    ops = load_ops(data_path)
    cfg = {key: ops.get(key) for key in REQUIRED_SOMA_OPS_KEYS}
    cfg[SHARED_SOMA_OPS_FLAG] = bool(ops.get(SHARED_SOMA_OPS_FLAG, False))
    return cfg


def missing_soma_workflow_keys(data_path):
    """Return ops keys that are missing or null for the Sindbis soma workflow."""
    ops = load_ops(data_path)
    required = _required_soma_ops_keys(ops)
    return [key for key in required if key not in ops or ops[key] is None]


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
    """Collect the main files produced by the Sindbis soma workflow.

    Includes both the per-chamber outputs and the mouse-level (shared) outputs
    used by the shared-bleedthrough workflow (see
    :func:`build_shared_soma_cluster_means`).
    """
    processed_path = get_processed_path(data_path)
    mouse_path = get_mouse_path(data_path)
    cells_dir = processed_path / "cells"
    stitched_dir = cells_dir / f"{barcode_prefix}_cells"
    trace_cache_dir = processed_path / "somata_traces"
    return {
        "processed_path": processed_path,
        "mouse_path": mouse_path,
        "cells_dir": cells_dir,
        "stitched_dir": stitched_dir,
        "somata_trace_cache_dir": trace_cache_dir,
        "somata_trace_tile_stats": trace_cache_dir
        / f"{barcode_prefix}_soma_trace_tile_stats.pkl",
        "somata_traces": processed_path / "somata_traces_df.pkl",
        "somata_cluster_means": processed_path / "somata_barcode_cluster_means.npy",
        "somata_reference_barcodes": processed_path / "somata_reference_barcodes.npz",
        "shared_somata_traces": mouse_path / "somata_traces_df.pkl",
        "shared_somata_cluster_means": mouse_path
        / "somata_barcode_cluster_means.npy",
        "shared_somata_reference_barcodes": mouse_path
        / "somata_reference_barcodes.npz",
        "shared_somata_diagnostics_dir": mouse_path
        / "diagnostics"
        / "shared_soma_reference",
        "stitched_calls": stitched_dir / f"{barcode_prefix}_df_corrected.pkl",
        "filtered_calls": stitched_dir / f"{barcode_prefix}_df_filtered.pkl",
    }


def get_soma_trace_cache_paths(
    data_path,
    tile_coors=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
):
    """Return per-tile soma trace cache paths and the aggregate stats path."""
    root = Path(output_root) if output_root is not None else get_processed_path(data_path)
    traces_dir = root / "somata_traces"
    paths = {
        "traces_dir": traces_dir,
        "stats_summary": traces_dir / f"{barcode_prefix}_soma_trace_tile_stats.pkl",
    }
    if tile_coors is not None:
        roi, tx, ty = (int(v) for v in tile_coors)
        stem = f"{barcode_prefix}_soma"
        paths.update(
            {
                "trace": traces_dir / f"{stem}_traces_{roi}_{tx}_{ty}.pkl",
                "stats": traces_dir / f"{stem}_trace_stats_{roi}_{tx}_{ty}.json",
            }
        )
    return paths


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
        trim_edge_masks=False,
    )


def deduplicate_soma_masks(
    data_path,
    segmentation_prefix=None,
    upper_overlap_thresh=0.3,
    **kwargs,
):
    """Deprecated no-op: soma duplicate handling now happens in the atlas build."""
    cfg = validate_soma_workflow_config(data_path)
    if segmentation_prefix is None:
        segmentation_prefix = cfg["segmentation_acquisition"]
    warnings.warn(
        "deduplicate_soma_masks is deprecated and no longer edits soma masks. "
        "Build the soma atlas from raw masks with build_soma_atlas/build_soma_atlases; "
        "tile-border duplicates are merged there.",
        DeprecationWarning,
        stacklevel=2,
    )
    return []


def _tile_id(tile_coors):
    return "_".join(map(str, tile_coors))


def _empty_soma_trace_table(tile_coors):
    traces_df = pd.DataFrame(
        columns=["label", "centroid-0", "centroid-1", "area", "trace", "std"]
    )
    traces_df["tile_of_origin"] = _tile_id(tile_coors)
    traces_df["roi"] = int(tile_coors[0])
    traces_df["tilex"] = int(tile_coors[1])
    traces_df["tiley"] = int(tile_coors[2])
    traces_df["bad_pixel_fraction"] = np.nan
    traces_df["edge_touching_labels_removed"] = 0
    return traces_df


def _json_ready(value):
    """Convert numpy/pandas scalars in cache metadata to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _current_soma_trace_cache_meta(
    data_path,
    tile_coors,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    reference_prefix=None,
    corrected_shifts=None,
    highpass_cutoff=DEFAULT_SOMA_TRACE_HIGHPASS_CUTOFF,
    highpass_order=DEFAULT_SOMA_TRACE_HIGHPASS_ORDER,
    highpass_pad=DEFAULT_SOMA_TRACE_HIGHPASS_PAD,
    atlas_output_root=None,
):
    """Build the metadata fingerprint for a trace cache entry."""
    cfg = validate_soma_workflow_config(data_path)
    ops = load_ops(data_path)
    if reference_prefix is None:
        reference_prefix = cfg["reference_prefix"]
    if corrected_shifts is None:
        corrected_shifts = cfg["corrected_shifts"]

    roi = int(tile_coors[0])
    atlas_paths = get_soma_atlas_paths(data_path, roi, output_root=atlas_output_root)
    if not atlas_paths["meta"].exists():
        raise FileNotFoundError(
            f"Soma atlas metadata for ROI {roi} not found at {atlas_paths['meta']}. "
            "Run build_soma_atlas / build_soma_atlases before extracting traces."
        )
    atlas_meta = np.load(atlas_paths["meta"], allow_pickle=False)

    meta = {
        "schema_version": SOMA_TRACE_CACHE_SCHEMA_VERSION,
        "data_path": str(data_path),
        "tile_coors": [int(v) for v in tile_coors],
        "tile_of_origin": _tile_id(tile_coors),
        "barcode_prefix": str(barcode_prefix),
        "reference_prefix": str(reference_prefix),
        "corrected_shifts": str(corrected_shifts),
        "barcode_rounds": int(cfg["barcode_rounds"]),
        "camera_order": [int(v) for v in ops["camera_order"]],
        "highpass": {
            "cutoff": float(highpass_cutoff),
            "order": int(highpass_order),
            "pad": int(highpass_pad),
        },
        "atlas": {
            "roi": roi,
            "reference_prefix": str(_meta_scalar(atlas_meta, "reference_prefix")),
            "corrected_shifts": str(_meta_scalar(atlas_meta, "corrected_shifts")),
            "segmentation_prefix": str(_meta_scalar(atlas_meta, "segmentation_prefix")),
            "mask_suffix": str(_meta_scalar(atlas_meta, "mask_suffix")),
        },
    }
    return _json_ready(meta)


def _flatten_meta(meta, prefix=""):
    flat = {}
    if isinstance(meta, dict):
        for key, value in meta.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_meta(value, child_prefix))
    else:
        flat[prefix] = meta
    return flat


def _validate_soma_trace_cache_meta(cached_meta, current_meta, cache_path):
    """Raise when a trace cache entry was produced for another data state."""
    if cached_meta is None:
        raise ValueError(
            f"Soma trace cache {cache_path} has no {SOMA_TRACE_CACHE_META_ATTR!r} "
            "metadata. Re-run extract_soma_trace_tiles."
        )
    cached_flat = _flatten_meta(_json_ready(cached_meta))
    current_flat = _flatten_meta(_json_ready(current_meta))
    mismatches = [
        key
        for key in sorted(set(cached_flat) | set(current_flat))
        if cached_flat.get(key) != current_flat.get(key)
    ]
    if mismatches:
        shown = ", ".join(mismatches[:8])
        if len(mismatches) > 8:
            shown += ", ..."
        raise ValueError(
            f"Soma trace cache {cache_path} is stale or incompatible "
            f"({shown}). Re-run extract_soma_trace_tiles."
        )


def _soma_trace_stats_row(traces_df, tile_coors, std_threshold):
    """Summarize a cached trace table using the reference-tile ranking columns."""
    tile_coors = tuple(int(v) for v in tile_coors)
    if len(traces_df):
        trace_mean_abs = traces_df["trace"].apply(lambda tr: float(np.mean(np.abs(tr))))
        return {
            "tile_of_origin": _tile_id(tile_coors),
            "roi": tile_coors[0],
            "tilex": tile_coors[1],
            "tiley": tile_coors[2],
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
    return {
        "tile_of_origin": _tile_id(tile_coors),
        "roi": tile_coors[0],
        "tilex": tile_coors[1],
        "tiley": tile_coors[2],
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
        "edge_touching_labels_removed": 0,
    }


def _write_soma_trace_stats(stats_path, stats_row, meta):
    payload = {"meta": _json_ready(meta), "stats": _json_ready(stats_row)}
    tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    with tmp_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(stats_path)


def _save_soma_trace_cache(traces_df, paths, stats_row):
    paths["traces_dir"].mkdir(parents=True, exist_ok=True)
    trace_tmp = paths["trace"].with_suffix(paths["trace"].suffix + ".tmp")
    traces_df.to_pickle(trace_tmp)
    trace_tmp.replace(paths["trace"])
    _write_soma_trace_stats(
        paths["stats"],
        stats_row=stats_row,
        meta=traces_df.attrs[SOMA_TRACE_CACHE_META_ATTR],
    )


def _load_soma_trace_stats(
    data_path,
    tile_coors,
    std_threshold,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
    validate=True,
):
    paths = get_soma_trace_cache_paths(
        data_path,
        tile_coors=tile_coors,
        barcode_prefix=barcode_prefix,
        output_root=output_root,
    )
    if not paths["stats"].exists():
        raise FileNotFoundError(
            f"Soma trace stats for tile {tuple(tile_coors)} not found at "
            f"{paths['stats']}. Run extract_soma_trace_tiles and "
            "collect_soma_trace_tile_stats first."
        )
    with paths["stats"].open() as fh:
        payload = json.load(fh)
    if validate:
        current_meta = _current_soma_trace_cache_meta(
            data_path,
            tile_coors=tile_coors,
            barcode_prefix=barcode_prefix,
            atlas_output_root=atlas_output_root,
        )
        _validate_soma_trace_cache_meta(payload.get("meta"), current_meta, paths["stats"])
    stats_row = payload["stats"]
    if float(stats_row["std_threshold"]) != float(std_threshold):
        traces_df = load_soma_trace_tile(
            data_path,
            tile_coors=tile_coors,
            validate=False,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
        )
        stats_row = _soma_trace_stats_row(traces_df, tile_coors, std_threshold)
    return stats_row


def load_soma_trace_tile(
    data_path,
    tile_coors,
    validate=True,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
):
    """Load one cached soma trace table, optionally validating its metadata."""
    paths = get_soma_trace_cache_paths(
        data_path,
        tile_coors=tile_coors,
        barcode_prefix=barcode_prefix,
        output_root=output_root,
    )
    if not paths["trace"].exists():
        raise FileNotFoundError(
            f"Soma trace cache for tile {tuple(tile_coors)} not found at "
            f"{paths['trace']}. Run extract_soma_trace_tiles first."
        )
    traces_df = pd.read_pickle(paths["trace"])
    if validate:
        current_meta = _current_soma_trace_cache_meta(
            data_path,
            tile_coors=tile_coors,
            barcode_prefix=barcode_prefix,
            atlas_output_root=atlas_output_root,
        )
        _validate_soma_trace_cache_meta(
            traces_df.attrs.get(SOMA_TRACE_CACHE_META_ATTR),
            current_meta,
            paths["trace"],
        )
    return traces_df


def load_soma_traces_for_tiles(
    data_path,
    tile_list,
    validate=True,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
):
    """Load and concatenate cached soma traces for selected tiles."""
    trace_tables = [
        load_soma_trace_tile(
            data_path,
            tile_coors=tile,
            validate=validate,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
        )
        for tile in tile_list
    ]
    if not trace_tables:
        return pd.DataFrame()
    return pd.concat(trace_tables, axis=0, ignore_index=True)


def _chamber_label(data_path):
    """Short, human-readable chamber identifier (e.g. ``chamber_01``)."""
    return Path(data_path).name


def load_soma_traces_for_chambers(
    chamber_to_tiles,
    validate=True,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
):
    """Load and concatenate cached soma traces from tiles in multiple chambers.

    Tags each row with its chamber so downstream provenance survives the
    per-chamber ROI-int collisions (ROI numbering is per-chamber, not unique
    mouse-wide). ``tile_of_origin`` is rewritten as ``"<chamber_label>:<tile_id>"``
    and a new ``chamber`` column is added.

    Args:
        chamber_to_tiles (dict): mapping ``{data_path: [(roi, tilex, tiley), ...]}``.
        validate (bool): whether to validate each tile's cache metadata.
        barcode_prefix (str): trace-cache prefix.

    Returns:
        pd.DataFrame: concatenated traces with chamber-tagged provenance.
    """
    frames = []
    for data_path, tile_list in chamber_to_tiles.items():
        if not tile_list:
            continue
        df = load_soma_traces_for_tiles(
            data_path,
            tile_list=tile_list,
            validate=validate,
            barcode_prefix=barcode_prefix,
        )
        if df.empty:
            continue
        chamber = _chamber_label(data_path)
        df = df.copy()
        df["chamber"] = chamber
        df["chamber_data_path"] = str(data_path)
        df["tile_of_origin"] = chamber + ":" + df["tile_of_origin"].astype(str)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "3:00:00", "mem": "16GB"})
def extract_soma_trace_tile(
    data_path,
    tile_coors,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    highpass_cutoff=DEFAULT_SOMA_TRACE_HIGHPASS_CUTOFF,
    highpass_order=DEFAULT_SOMA_TRACE_HIGHPASS_ORDER,
    highpass_pad=DEFAULT_SOMA_TRACE_HIGHPASS_PAD,
    edge_mask_border=10,
    atlas_output_root=None,
    output_root=None,
    save=True,
    force=False,
):
    """Extract and cache soma traces from one tile.

    Loads the per-tile soma mask from the canonical ROI atlas (must be built
    first via `build_soma_atlas` / `build_soma_atlases`) and the registered
    barcode stack from `load_register_and_fill_tile`.
    """
    cfg = validate_soma_workflow_config(data_path)
    if reference_prefix is None:
        reference_prefix = cfg["reference_prefix"]
    if corrected_shifts is None:
        corrected_shifts = cfg["corrected_shifts"]
    sidecar_std_threshold = (
        cfg["soma_std_threshold_clustering"] or DEFAULT_SOMA_STD_THRESHOLD
    )
    paths = get_soma_trace_cache_paths(
        data_path,
        tile_coors=tile_coors,
        barcode_prefix=barcode_prefix,
        output_root=output_root,
    )
    if save and paths["trace"].exists() and not force:
        traces_df = load_soma_trace_tile(
            data_path,
            tile_coors=tile_coors,
            validate=True,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
        )
        if not paths["stats"].exists():
            stats_row = _soma_trace_stats_row(
                traces_df,
                tile_coors,
                std_threshold=sidecar_std_threshold,
            )
            _write_soma_trace_stats(
                paths["stats"],
                stats_row=stats_row,
                meta=traces_df.attrs[SOMA_TRACE_CACHE_META_ATTR],
            )
        return traces_df

    cache_meta = _current_soma_trace_cache_meta(
        data_path,
        tile_coors=tile_coors,
        barcode_prefix=barcode_prefix,
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
        highpass_cutoff=highpass_cutoff,
        highpass_order=highpass_order,
        highpass_pad=highpass_pad,
        atlas_output_root=atlas_output_root,
    )
    masks = load_soma_atlas_tile(
        data_path,
        tile_coors=tile_coors,
        output_root=atlas_output_root,
        expected_reference_prefix=reference_prefix,
        expected_corrected_shifts=corrected_shifts,
    )
    if not np.any(masks):
        traces_df = _empty_soma_trace_table(tile_coors)
        traces_df.attrs[SOMA_TRACE_CACHE_META_ATTR] = cache_meta
        stats_row = _soma_trace_stats_row(
            traces_df,
            tile_coors,
            std_threshold=sidecar_std_threshold,
        )
        if save:
            _save_soma_trace_cache(traces_df, paths, stats_row)
        return traces_df

    stack, bad_pixels = load_register_and_fill_tile(
        data_path,
        tile_coors,
        prefix=barcode_prefix,
        suffix="max",
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
        correct_channels="round1_only",
        correct_illumination=True,
        filter_r=None,
        specific_rounds=None,
        nrounds=cfg["barcode_rounds"],
        edge=10,
        mid=5,
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
    traces_df = extract_traces_somata(filtered_stack, masks)
    tile_id = "_".join(map(str, tile_coors))
    traces_df["tile_of_origin"] = tile_id
    traces_df["roi"] = int(tile_coors[0])
    traces_df["tilex"] = int(tile_coors[1])
    traces_df["tiley"] = int(tile_coors[2])
    traces_df["bad_pixel_fraction"] = float(np.mean(bad_pixels))
    traces_df["edge_touching_labels_removed"] = 0
    traces_df.attrs[SOMA_TRACE_CACHE_META_ATTR] = cache_meta
    stats_row = _soma_trace_stats_row(
        traces_df,
        tile_coors,
        std_threshold=sidecar_std_threshold,
    )
    if save:
        _save_soma_trace_cache(traces_df, paths, stats_row)
    return traces_df


def extract_soma_reference_traces(
    data_path,
    tile_coors,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    highpass_cutoff=DEFAULT_SOMA_TRACE_HIGHPASS_CUTOFF,
    highpass_order=DEFAULT_SOMA_TRACE_HIGHPASS_ORDER,
    highpass_pad=DEFAULT_SOMA_TRACE_HIGHPASS_PAD,
    edge_mask_border=10,
    atlas_output_root=None,
):
    """Compatibility wrapper for one-off, unsaved soma trace extraction."""
    warnings.warn(
        "extract_soma_reference_traces performs live extraction and does not use "
        "the soma trace cache. Prefer extract_soma_trace_tiles followed by "
        "load_soma_traces_for_tiles.",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_soma_trace_tile(
        data_path,
        tile_coors=tile_coors,
        segmentation_prefix=segmentation_prefix,
        reference_prefix=reference_prefix,
        corrected_shifts=corrected_shifts,
        barcode_prefix=barcode_prefix,
        highpass_cutoff=highpass_cutoff,
        highpass_order=highpass_order,
        highpass_pad=highpass_pad,
        edge_mask_border=edge_mask_border,
        atlas_output_root=atlas_output_root,
        save=False,
        force=True,
    )


def extract_soma_trace_tiles(
    data_path,
    use_rois=None,
    tile_list=None,
    use_slurm=True,
    force=False,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
):
    """Extract/cache soma traces for a chamber, optionally via Slurm."""
    validate_soma_workflow_config(data_path)
    if tile_list is None:
        if (
            use_slurm
            and output_root is None
            and atlas_output_root is None
            and barcode_prefix == DEFAULT_SOMA_BARCODE_PREFIX
        ):
            roi_dims = get_roi_dimensions(data_path)
            if use_rois is not None:
                roi_dims = roi_dims[np.isin(roi_dims[:, 0], use_rois)]
            additional_args = ",FORCE=1" if force else ""
            return batch_process_tiles(
                data_path,
                "extract_soma_trace_tile",
                roi_dims=roi_dims,
                additional_args=additional_args,
            )
        tile_list = enumerate_soma_candidate_tiles(data_path, use_rois=use_rois)

    outputs = {}
    slurm_folder = Path.home() / "slurm_logs" / data_path / "extract_soma_trace_tile"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    for tile in tile_list:
        tile_tuple = tuple(int(v) for v in tile)
        outputs[tile_tuple] = extract_soma_trace_tile(
            data_path,
            tile_coors=tile_tuple,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
            force=force,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"extract_soma_trace_tile_{tile_tuple[0]}_{tile_tuple[1]}_{tile_tuple[2]}",
        )
    return outputs


def collect_soma_trace_tile_stats(
    data_path,
    tile_list=None,
    validate=True,
    save=True,
    std_threshold=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
):
    """Collect per-tile trace-cache stats into one summary dataframe."""
    cfg = validate_soma_workflow_config(data_path)
    if std_threshold is None:
        std_threshold = (
            cfg["soma_std_threshold_clustering"] or DEFAULT_SOMA_STD_THRESHOLD
        )
    if tile_list is None:
        tile_list = enumerate_soma_candidate_tiles(data_path)

    rows = []
    for tile in tile_list:
        tile_tuple = tuple(int(v) for v in tile)
        traces_df = load_soma_trace_tile(
            data_path,
            tile_coors=tile_tuple,
            validate=validate,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
        )
        row = _soma_trace_stats_row(traces_df, tile_tuple, std_threshold)
        rows.append(row)
        if save:
            paths = get_soma_trace_cache_paths(
                data_path,
                tile_coors=tile_tuple,
                barcode_prefix=barcode_prefix,
                output_root=output_root,
            )
            _write_soma_trace_stats(
                paths["stats"],
                stats_row=row,
                meta=traces_df.attrs[SOMA_TRACE_CACHE_META_ATTR],
            )

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(
            by=["n_std_ge_threshold", "std_p90", "mask_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    if save:
        paths = get_soma_trace_cache_paths(
            data_path,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
        )
        paths["traces_dir"].mkdir(parents=True, exist_ok=True)
        summary.attrs["soma_trace_stats_meta"] = {
            "schema_version": SOMA_TRACE_CACHE_SCHEMA_VERSION,
            "data_path": str(data_path),
            "barcode_prefix": str(barcode_prefix),
            "std_threshold": float(std_threshold),
        }
        tmp_path = paths["stats_summary"].with_suffix(
            paths["stats_summary"].suffix + ".tmp"
        )
        summary.to_pickle(tmp_path)
        tmp_path.replace(paths["stats_summary"])
    return summary


def summarize_soma_reference_tiles(
    data_path,
    tile_list=None,
    use_rois=None,
    std_threshold=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    output_root=None,
    atlas_output_root=None,
):
    """Summarize candidate reference tiles from cached per-tile stats only."""
    cfg = validate_soma_workflow_config(data_path)
    if std_threshold is None:
        std_threshold = (
            cfg["soma_std_threshold_clustering"] or DEFAULT_SOMA_STD_THRESHOLD
        )
    if tile_list is None:
        tile_list = enumerate_soma_candidate_tiles(data_path, use_rois=use_rois)

    rows = [
        _load_soma_trace_stats(
            data_path,
            tile_coors=tile_coors,
            std_threshold=std_threshold,
            barcode_prefix=barcode_prefix,
            output_root=output_root,
            atlas_output_root=atlas_output_root,
            validate=True,
        )
        for tile_coors in tile_list
    ]

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(
            by=["n_std_ge_threshold", "std_p90", "mask_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return summary


def summarize_soma_reference_tiles_across_chambers(
    chamber_data_paths,
    std_threshold,
    use_rois=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
):
    """Pool per-chamber reference-tile rankings into one mouse-wide table.

    Each chamber's cached per-tile stats are loaded via the existing
    :func:`summarize_soma_reference_tiles` (which recounts at ``std_threshold``
    when the cached value differs); rows are tagged with ``chamber`` and
    ``chamber_data_path`` columns; the concatenated table is re-sorted on the
    standard ranking keys so the bigger chamber naturally wins more top-N
    slots.

    Args:
        chamber_data_paths (Iterable[str]): chambers to pool.
        std_threshold (float): threshold to evaluate ``n_std_ge_threshold``
            against, applied uniformly across chambers.
        use_rois (Iterable[int] | None): optional ROI filter passed to each
            chamber's tile enumeration.
        barcode_prefix (str): trace-cache prefix.

    Returns:
        pd.DataFrame: pooled, mouse-wide ranked summary.
    """
    frames = []
    for data_path in chamber_data_paths:
        chamber_summary = summarize_soma_reference_tiles(
            data_path,
            use_rois=use_rois,
            std_threshold=std_threshold,
            barcode_prefix=barcode_prefix,
        )
        if chamber_summary.empty:
            continue
        chamber_summary = chamber_summary.copy()
        chamber_summary["chamber"] = _chamber_label(data_path)
        chamber_summary["chamber_data_path"] = str(data_path)
        frames.append(chamber_summary)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, axis=0, ignore_index=True)
    return pooled.sort_values(
        by=["n_std_ge_threshold", "std_p90", "mask_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


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
    """Load one barcode tile and its soma masks in the local reference-tile frame.

    .. deprecated::
        Replaced by `load_soma_atlas_tile`, which slices a precomputed canonical
        ROI-global soma atlas instead of re-warping per-tile on the fly. Internal
        callers (`extract_soma_reference_traces`, `summarize_soma_reference_tiles`,
        `basecall_somata_tile`) are migrated. Kept here only in case a scratch
        notebook still imports the name.
    """
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
    atlas_output_root=None,
    trace_output_root=None,
    use_trace_cache=True,
    save=False,
):
    """Compute soma cluster means from user-selected tiles and thresholds."""
    cfg = validate_soma_workflow_config(data_path)
    if std_threshold is None:
        std_threshold = (
            cfg["soma_std_threshold_clustering"] or DEFAULT_SOMA_STD_THRESHOLD
        )
    if cluster_score_thresh is None:
        cluster_score_thresh = cfg["somata_cluster_score_thresh"]

    if use_trace_cache:
        traces_df = load_soma_traces_for_tiles(
            data_path,
            tile_list=reference_tiles,
            validate=True,
            barcode_prefix=barcode_prefix,
            output_root=trace_output_root,
            atlas_output_root=atlas_output_root,
        )
    else:
        all_tile_dfs = []
        for tile_coors in reference_tiles:
            all_tile_dfs.append(
                extract_soma_reference_traces(
                    data_path,
                    tile_coors=tile_coors,
                    barcode_prefix=barcode_prefix,
                    atlas_output_root=atlas_output_root,
                )
            )
        traces_df = (
            pd.concat(all_tile_dfs, axis=0, ignore_index=True)
            if all_tile_dfs
            else pd.DataFrame()
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


def _resolve_shared_soma_threshold(chamber_data_paths, ops_key, label):
    """Pick the unique value of ``ops_key`` declared across the chambers.

    Enforces the rule "exactly one chamber may declare this setting in
    shared mode": multiple declarations or zero declarations raise
    ``ValueError`` listing the offenders / requirement.
    """
    declared = []
    for data_path in chamber_data_paths:
        ops = load_ops(data_path)
        value = ops.get(ops_key)
        if value is not None:
            declared.append((data_path, value))
    if len(declared) == 0:
        raise ValueError(
            f"No chamber in the shared pool declares ops[{ops_key!r}]. "
            "Exactly one chamber must own this value when "
            f"{SHARED_SOMA_OPS_FLAG} is True (label: {label})."
        )
    if len(declared) > 1:
        names = ", ".join(str(dp) for dp, _ in declared)
        raise ValueError(
            f"Multiple chambers declare ops[{ops_key!r}] in the shared pool "
            f"({names}). Exactly one chamber must own this value when "
            f"{SHARED_SOMA_OPS_FLAG} is True (label: {label})."
        )
    return declared[0][1]


def _resolve_shared_initial_cluster_means(chamber_data_paths):
    """All chambers in the pool must agree on ``initial_cluster_means``."""
    values = []
    for data_path in chamber_data_paths:
        ops = load_ops(data_path)
        value = ops.get("initial_cluster_means")
        if value is None:
            raise ValueError(
                f"Chamber {data_path} is missing ops['initial_cluster_means']; "
                "this key is required in every chamber even in shared mode."
            )
        values.append((data_path, np.asarray(value, dtype=float)))
    base_dp, base = values[0]
    for dp, arr in values[1:]:
        if arr.shape != base.shape or not np.allclose(arr, base):
            raise ValueError(
                "Chambers disagree on ops['initial_cluster_means']: "
                f"{base_dp} vs {dp}. Reconcile before running the shared build."
            )
    return base


def _validate_shared_chamber_compatibility(chamber_data_paths):
    """All chambers in the pool must share the trace-cache–defining ops fields."""
    keys = ("reference_prefix", "corrected_shifts", "barcode_rounds", "camera_order")
    base = None
    base_dp = None
    for data_path in chamber_data_paths:
        ops = load_ops(data_path)
        snapshot = {k: ops.get(k) for k in keys}
        if base is None:
            base = snapshot
            base_dp = data_path
            continue
        mismatches = [
            k for k in keys if _json_ready(snapshot[k]) != _json_ready(base[k])
        ]
        if mismatches:
            raise ValueError(
                "Chambers disagree on trace-cache-defining ops fields "
                f"({', '.join(mismatches)}): {base_dp} vs {data_path}. "
                "Reconcile before running the shared build."
            )


@slurm_it(conda_env="iss-preprocess")
def build_shared_soma_cluster_means(
    chamber_data_paths,
    chamber_to_tiles=None,
    std_threshold=None,
    cluster_score_thresh=None,
    initial_cluster_mean=None,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    save=True,
    save_diagnostics=True,
):
    """Build one soma cluster-means / bleedthrough matrix from reference tiles
    pooled across multiple chambers of one mouse.

    The shared file is written at the mouse level
    (``processed/{project}/{mouse}/somata_barcode_cluster_means.npy``) and is
    consumed by ``basecall_somata_tile`` for every chamber whose ops set
    ``use_shared_soma_cluster_means: true``.

    Args:
        chamber_data_paths (list[str]): chambers in the pool. All must share
            the same mouse and the same trace-cache-defining ops fields
            (``reference_prefix``, ``corrected_shifts``, ``barcode_rounds``,
            ``camera_order``) and the same ``initial_cluster_means``.
        chamber_to_tiles (dict | None): explicit
            ``{data_path: [tile_coors, ...]}`` mapping. When ``None``, each
            chamber's ``ops['barcode_soma_reference_tiles']`` is used (chambers
            without that key contribute no tiles).
        std_threshold (float | None): explicit override. When ``None``, the
            unique declared value is resolved from the chambers' ops via the
            "exactly one chamber declares this" rule.
        cluster_score_thresh (float | None): same resolution rule as
            ``std_threshold``.
        initial_cluster_mean (np.ndarray | None): explicit override.
            When ``None``, all chambers must agree on
            ``ops['initial_cluster_means']``.
        barcode_prefix (str): trace-cache prefix.
        save (bool): write outputs to mouse-level paths.
        save_diagnostics (bool): write the diagnostics PNG/CSV bundle.

    Returns:
        dict: ``compute_soma_reference_from_traces`` result extended with
        ``reference_tiles_by_chamber``, ``chamber_data_paths``,
        ``saved_paths``, ``diagnostics_dir``.
    """
    chamber_data_paths = list(chamber_data_paths)
    if not chamber_data_paths:
        raise ValueError("chamber_data_paths must contain at least one chamber.")

    mouse_paths = {get_mouse_path(dp) for dp in chamber_data_paths}
    if len(mouse_paths) != 1:
        raise ValueError(
            "All chambers must belong to the same mouse; got "
            f"{sorted(str(p) for p in mouse_paths)}."
        )

    for data_path in chamber_data_paths:
        validate_soma_workflow_config(data_path)
    _validate_shared_chamber_compatibility(chamber_data_paths)

    if std_threshold is None:
        std_threshold = _resolve_shared_soma_threshold(
            chamber_data_paths, "soma_std_threshold_clustering", "std_threshold"
        )
    if cluster_score_thresh is None:
        cluster_score_thresh = _resolve_shared_soma_threshold(
            chamber_data_paths,
            "somata_cluster_score_thresh",
            "cluster_score_thresh",
        )
    if initial_cluster_mean is None:
        initial_cluster_mean = _resolve_shared_initial_cluster_means(
            chamber_data_paths
        )

    if chamber_to_tiles is None:
        chamber_to_tiles = {}
        for data_path in chamber_data_paths:
            ops = load_ops(data_path)
            tiles = ops.get("barcode_soma_reference_tiles") or []
            chamber_to_tiles[data_path] = [tuple(t) for t in tiles]
    else:
        chamber_to_tiles = {
            dp: [tuple(t) for t in tiles] for dp, tiles in chamber_to_tiles.items()
        }
    total_tiles = sum(len(v) for v in chamber_to_tiles.values())
    if total_tiles == 0:
        raise ValueError(
            "No reference tiles supplied across the pool. Set ops "
            "['barcode_soma_reference_tiles'] on at least one chamber or pass "
            "chamber_to_tiles explicitly."
        )

    traces_df = load_soma_traces_for_chambers(
        chamber_to_tiles,
        validate=True,
        barcode_prefix=barcode_prefix,
    )
    if traces_df.empty:
        raise ValueError(
            "Pooled trace dataframe is empty; check that per-tile soma trace "
            "caches exist for the supplied tiles."
        )

    result = compute_soma_reference_from_traces(
        traces_df,
        std_threshold=std_threshold,
        cluster_score_thresh=cluster_score_thresh,
        initial_cluster_mean=initial_cluster_mean,
    )
    result["reference_tiles_by_chamber"] = chamber_to_tiles
    result["chamber_data_paths"] = chamber_data_paths

    diagnostics_dir = None
    if save:
        outputs = get_soma_output_paths(
            chamber_data_paths[0], barcode_prefix=barcode_prefix
        )
        outputs["mouse_path"].mkdir(parents=True, exist_ok=True)
        np.save(outputs["shared_somata_cluster_means"], result["cluster_means"])
        np.savez(
            outputs["shared_somata_reference_barcodes"],
            spot_colors=result["spot_colors"],
            cluster_inds=np.array(result["cluster_inds"], dtype=object),
        )
        traces_df.to_pickle(outputs["shared_somata_traces"])
        result["saved_paths"] = {
            "shared_somata_cluster_means": outputs["shared_somata_cluster_means"],
            "shared_somata_reference_barcodes": outputs[
                "shared_somata_reference_barcodes"
            ],
            "shared_somata_traces": outputs["shared_somata_traces"],
        }
        diagnostics_dir = outputs["shared_somata_diagnostics_dir"]

    if save_diagnostics:
        from ..diagnostics.diag_somata import (
            plot_shared_soma_reference_diagnostics,
        )

        if diagnostics_dir is None:
            outputs = get_soma_output_paths(
                chamber_data_paths[0], barcode_prefix=barcode_prefix
            )
            diagnostics_dir = outputs["shared_somata_diagnostics_dir"]
        plot_shared_soma_reference_diagnostics(
            result, chamber_to_tiles, save_dir=diagnostics_dir
        )
    result["diagnostics_dir"] = diagnostics_dir
    return result


@slurm_it(conda_env="iss-preprocess")
def setup_soma_calling_reference(
    data_path,
    reload=False,
    force_redo=False,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    atlas_output_root=None,
    trace_output_root=None,
):
    """Build soma-specific cluster means / bleedthrough reference from cached traces.

    `reload` and `force_redo` are accepted for compatibility with older notebook
    and CLI calls. Existing `somata_traces_df.pkl` files are never read here;
    reference traces are loaded from the validated per-tile trace cache.

    Skips with a warning when the chamber's ops set
    ``use_shared_soma_cluster_means: true`` — those chambers consume the
    mouse-level shared file produced by ``build_shared_soma_cluster_means``.
    """
    ops = load_ops(data_path)
    if ops.get(SHARED_SOMA_OPS_FLAG, False):
        warnings.warn(
            f"{data_path} is configured for shared soma cluster means "
            f"({SHARED_SOMA_OPS_FLAG}=True). Skipping per-chamber setup; "
            "run `iss-call setup-shared-soma-barcodes` instead.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    cfg = validate_soma_workflow_config(data_path)
    if reload:
        warnings.warn(
            "setup_soma_calling_reference no longer reloads somata_traces_df.pkl; "
            "reference traces will be loaded from the validated per-tile cache.",
            RuntimeWarning,
            stacklevel=2,
        )
    return compute_soma_reference_from_tiles(
        data_path,
        reference_tiles=cfg["barcode_soma_reference_tiles"],
        std_threshold=cfg["soma_std_threshold_clustering"],
        cluster_score_thresh=cfg["somata_cluster_score_thresh"],
        initial_cluster_mean=cfg["initial_cluster_means"],
        barcode_prefix=barcode_prefix,
        atlas_output_root=atlas_output_root,
        trace_output_root=trace_output_root,
        save=True,
    )


def basecall_somata_tiles(data_path):
    """Submit soma basecalling jobs for all tiles in a chamber."""
    validate_soma_workflow_config(data_path)
    return batch_process_tiles(data_path, "basecall_somata_tile")


def stitch_soma_calls(
    data_path,
    barcode_prefix=DEFAULT_SOMA_BARCODE_PREFIX,
    ref_prefix=None,
    output_root=None,
    **kwargs,
):
    """Stitch per-tile soma calls into ROI-global reference coordinates."""
    validate_soma_workflow_config(data_path)
    return stitch_cell_dataframes(
        data_path,
        prefix=barcode_prefix,
        ref_prefix=ref_prefix,
        sindbis=True,
        output_root=output_root,
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


# ---------------------------------------------------------------------------
# Canonical soma atlas: build once per ROI in barcode-reference coordinates,
# crop back to per-tile masks for basecalling.
# ---------------------------------------------------------------------------


def get_soma_atlas_paths(data_path, roi, output_root=None):
    """Return atlas npy + meta npz paths for a given ROI."""
    root = Path(output_root) if output_root is not None else get_processed_path(data_path)
    cells_dir = root / "cells"
    return {
        "cells_dir": cells_dir,
        "atlas": cells_dir / f"soma_atlas_{int(roi)}.npy",
        "meta": cells_dir / f"soma_atlas_{int(roi)}_meta.npz",
    }


def _load_roi_tile_origins(data_path, reference_prefix, roi):
    """Compute per-tile origins (ntx, nty, 2) in ROI-global barcode-ref coords."""
    processed_path = get_processed_path(data_path)
    shifts_path = (
        processed_path
        / "reg"
        / f"{reference_prefix}_within"
        / f"{reference_prefix}_{int(roi)}_shifts.npz"
    )
    if not shifts_path.exists():
        raise FileNotFoundError(
            f"Reference within-acquisition shifts not found at {shifts_path}; "
            "run register_within_acquisition for the reference prefix first."
        )
    shifts = np.load(shifts_path)
    roi_dims = get_roi_dimensions(data_path)
    ntiles = roi_dims[roi_dims[:, 0] == int(roi), 1:][0] + 1
    ops = load_ops(data_path)
    tile_origins, _ = calculate_tile_positions(
        shifts["shift_right"],
        shifts["shift_down"],
        shifts["tile_shape"],
        ntiles=ntiles,
        x_direction=ops["x_tile_direction"],
        y_direction=ops["y_tile_direction"],
    )
    tile_origins = np.round(tile_origins).astype(int)
    tile_shape = np.asarray(shifts["tile_shape"], dtype=int)
    return tile_origins, tile_shape, ntiles


def _as_homogeneous_matrix(matrix):
    """Return a 3x3 float affine matrix."""
    matrix = np.asarray(matrix, dtype=float).copy()
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 2x3 or 3x3 affine matrix, got {matrix.shape}")
    return matrix


def _compose_mask_to_global_matrix(tform_local, tile_origin_yx):
    """Compose source-mask -> local-reference-tile -> ROI-global reference coords."""
    matrix = np.asarray(tform_local, dtype=float).copy()
    matrix = _as_homogeneous_matrix(matrix)
    y0, x0 = tile_origin_yx
    matrix[0, 2] += float(x0)
    matrix[1, 2] += float(y0)
    return matrix


def _matrix_in_canvas_space(global_matrix, canvas_global_origin_yx):
    """Shift an ROI-global affine into atlas array coordinates."""
    matrix = _as_homogeneous_matrix(global_matrix)
    origin_y, origin_x = canvas_global_origin_yx
    matrix[0, 2] -= float(origin_x)
    matrix[1, 2] -= float(origin_y)
    return matrix


def _transformed_tile_bounds(source_shape, global_matrix):
    """Return `(y0, x0, y1, x1)` bounds of a source tile after global transform."""
    height, width = (int(source_shape[0]), int(source_shape[1]))
    corners_xy = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width), 0.0, 1.0],
            [0.0, float(height), 1.0],
            [float(width), float(height), 1.0],
        ]
    )
    transformed = (_as_homogeneous_matrix(global_matrix) @ corners_xy.T).T
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    return (
        int(np.floor(np.min(ys))),
        int(np.floor(np.min(xs))),
        int(np.ceil(np.max(ys))),
        int(np.ceil(np.max(xs))),
    )


def _canvas_from_bounds(bounds, padding=4):
    """Compute canvas origin and shape from iterable `(y0, x0, y1, x1)` bounds."""
    bounds = np.asarray(bounds, dtype=int)
    if bounds.size == 0:
        raise ValueError("Cannot build a canvas without any bounds")
    min_y = int(np.min(bounds[:, 0])) - int(padding)
    min_x = int(np.min(bounds[:, 1])) - int(padding)
    max_y = int(np.max(bounds[:, 2])) + int(padding)
    max_x = int(np.max(bounds[:, 3])) + int(padding)
    canvas_global_origin_yx = np.asarray([min_y, min_x], dtype=int)
    canvas_shape = np.asarray([max_y - min_y, max_x - min_x], dtype=int)
    return canvas_global_origin_yx, canvas_shape


def _reference_tile_bounds(tile_origins, tile_shape):
    """Return nominal barcode-reference tile rectangles in ROI-global coords."""
    origins = np.asarray(tile_origins, dtype=int).reshape(-1, 2)
    height, width = (int(tile_shape[0]), int(tile_shape[1]))
    return np.column_stack(
        [
            origins[:, 0],
            origins[:, 1],
            origins[:, 0] + height,
            origins[:, 1] + width,
        ]
    )


def _warp_tile_to_local_bbox(
    source_mask,
    global_matrix,
    canvas_global_origin_yx,
    canvas_shape,
):
    """Warp one source mask into an array sized to its transformed bbox.

    Computes the integer bounding box of the source tile after `global_matrix`,
    clips it to the canvas, and warps into that local array only. Avoids the
    canvas-sized allocation that nearest-neighbour `skimage.warp` would otherwise
    produce.

    Args:
        source_mask: 2-D label array in source-tile coordinates.
        global_matrix: 3x3 affine from source pixels to ROI-global pixels.
        canvas_global_origin_yx: (y, x) of the canvas origin in ROI-global coords.
        canvas_shape: (H, W) of the full canvas.

    Returns:
        (warped_local, y0_canvas, x0_canvas) where `warped_local` is the warped
        mask shaped to the clipped bbox and `(y0_canvas, x0_canvas)` is its
        top-left in canvas coordinates.
    """
    y0g, x0g, y1g, x1g = _transformed_tile_bounds(source_mask.shape, global_matrix)
    oy, ox = int(canvas_global_origin_yx[0]), int(canvas_global_origin_yx[1])
    H_canvas, W_canvas = int(canvas_shape[0]), int(canvas_shape[1])
    y0 = max(0, y0g - oy)
    x0 = max(0, x0g - ox)
    y1 = min(H_canvas, y1g - oy)
    x1 = min(W_canvas, x1g - ox)
    if y1 <= y0 or x1 <= x0:
        return np.zeros((0, 0), dtype=source_mask.dtype), y0, x0
    # Compose source -> canvas, then shift to source -> local-bbox.
    matrix = _matrix_in_canvas_space(global_matrix, canvas_global_origin_yx).copy()
    matrix[0, 2] -= float(x0)
    matrix[1, 2] -= float(y0)
    warped_local = warp(
        source_mask,
        AffineTransform(matrix=matrix).inverse,
        order=0,
        preserve_range=True,
        output_shape=(y1 - y0, x1 - x0),
    ).astype(source_mask.dtype, copy=False)
    return warped_local, y0, x0


def _candidates_from_local_warp(source_id, warped_local, y0_canvas, x0_canvas):
    """Emit per-label candidate dicts from a local-bbox warped tile.

    Each candidate's `bbox` and `centroid_yx` are returned in canvas
    coordinates (offset by `(y0_canvas, x0_canvas)`); `local_mask` stays in
    bbox-local coordinates.

    Args:
        source_id: 3-tuple `(roi, tx, ty)` tagging the source tile.
        warped_local: 2-D label array from `_warp_tile_to_local_bbox`.
        y0_canvas, x0_canvas: top-left of `warped_local` in canvas coords.

    Returns:
        List of candidate dicts (see `_assemble_soma_atlas_arrays` for the
        consumed keys).
    """
    if warped_local.size == 0 or not np.any(warped_local):
        return []
    candidates = []
    for prop in regionprops(warped_local.astype(np.int64, copy=False)):
        ly0, lx0, ly1, lx1 = prop.bbox
        local_mask = warped_local[ly0:ly1, lx0:lx1] == prop.label
        if not local_mask.any():
            continue
        candidates.append(
            {
                "source_id": source_id,
                "source_label": int(prop.label),
                "bbox": (
                    ly0 + y0_canvas,
                    lx0 + x0_canvas,
                    ly1 + y0_canvas,
                    lx1 + x0_canvas,
                ),
                "local_mask": local_mask.copy(),
                "area": int(prop.area),
                "centroid_yx": np.asarray(
                    (
                        float(prop.centroid[0]) + y0_canvas,
                        float(prop.centroid[1]) + x0_canvas,
                    ),
                    dtype=float,
                ),
            }
        )
    return candidates


def _bbox_overlap_pairs(candidates):
    """Return upper-triangular index pairs (i, j) with i<j whose bboxes overlap."""
    if not candidates:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    bboxes = np.array([c["bbox"] for c in candidates], dtype=int)  # (N, 4): y0,x0,y1,x1
    y0, x0, y1, x1 = bboxes.T
    n = len(candidates)
    # Vectorised pairwise overlap test. Memory: O(N^2) bool ~ N=5000 -> 25 MB.
    overlap = (
        (y1[:, None] > y0[None, :])
        & (y0[:, None] < y1[None, :])
        & (x1[:, None] > x0[None, :])
        & (x0[:, None] < x1[None, :])
    )
    triu = np.triu(overlap, k=1)
    i_idx, j_idx = np.where(triu)
    return i_idx, j_idx


def _intersection_area(cand_a, cand_b):
    ay0, ax0, ay1, ax1 = cand_a["bbox"]
    by0, bx0, by1, bx1 = cand_b["bbox"]
    oy0 = max(ay0, by0)
    ox0 = max(ax0, bx0)
    oy1 = min(ay1, by1)
    ox1 = min(ax1, bx1)
    if oy1 <= oy0 or ox1 <= ox0:
        return 0
    sub_a = cand_a["local_mask"][oy0 - ay0 : oy1 - ay0, ox0 - ax0 : ox1 - ax0]
    sub_b = cand_b["local_mask"][oy0 - by0 : oy1 - by0, ox0 - bx0 : ox1 - bx0]
    return int(np.logical_and(sub_a, sub_b).sum())


class _UnionFind:
    """Tiny union-find with path compression."""

    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i):
        root = int(i)
        while self.parent[root] != root:
            root = int(self.parent[root])
        # Path compression
        cur = int(i)
        while self.parent[cur] != root:
            nxt = int(self.parent[cur])
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if ri < rj:
            self.parent[rj] = ri
        else:
            self.parent[ri] = rj

    def labels(self):
        roots = np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)
        # Renumber to dense 0..K-1 in sorted order of root id (deterministic).
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        return inverse, len(unique_roots)


def _watershed_contested_region(
    atlas, region_local_mask, y0, x0, component_centroids_by_gid
):
    """Resolve one connected contested region by distance-transform watershed.

    `region_local_mask` is the boolean mask of the region, sized to its
    bounding box. `(y0, x0)` is the top-left of that bbox in canvas coords.
    Centroids in `component_centroids_by_gid` are in canvas coords. Mutates
    `atlas` in place inside `[y0:y0+H, x0:x0+W]` and returns it. EDT and
    watershed run only on the local bbox.
    """
    if not component_centroids_by_gid:
        return atlas
    H_local, W_local = region_local_mask.shape
    if H_local == 0 or W_local == 0 or not region_local_mask.any():
        return atlas
    sorted_gids = sorted(component_centroids_by_gid)
    markers = np.zeros((H_local, W_local), dtype=np.int32)
    ys_local, xs_local = np.where(region_local_mask)
    for marker_id, gid in enumerate(sorted_gids, start=1):
        cy, cx = component_centroids_by_gid[gid]
        my = max(0, min(H_local - 1, int(round(cy)) - y0))
        mx = max(0, min(W_local - 1, int(round(cx)) - x0))
        if not region_local_mask[my, mx]:
            d2 = (ys_local - (cy - y0)) ** 2 + (xs_local - (cx - x0)) ** 2
            best = int(np.argmin(d2))
            my, mx = int(ys_local[best]), int(xs_local[best])
        markers[my, mx] = marker_id
    edt = distance_transform_edt(region_local_mask)
    ws = watershed(-edt, markers=markers, mask=region_local_mask)
    atlas_view = atlas[y0 : y0 + H_local, x0 : x0 + W_local]
    for marker_id, gid in enumerate(sorted_gids, start=1):
        atlas_view[(ws == marker_id) & region_local_mask] = gid
    return atlas


def _assemble_soma_atlas_arrays(
    candidates,
    tile_origins,
    tile_shape,
    canvas_shape,
    merge_containment_threshold=0.5,
):
    """Pure-numpy core: candidate dicts -> (atlas, owner_table, provenance).

    Args:
        candidates: list of dicts as returned by `_candidates_from_local_warp`,
            each describing one labelled region in the canvas frame.
        tile_origins: (ntx, nty, 2) integer (y, x) origins of barcode-ref tiles.
        tile_shape: (H_tile, W_tile) — both ints.
        canvas_shape: (H, W).
        merge_containment_threshold: union two source labels if
            max(intersection/area_A, intersection/area_B) >= threshold.

    Returns:
        atlas (uint32 H x W), owner_table (structured), provenance (structured).
    """
    n = len(candidates)
    if n == 0:
        atlas = np.zeros(canvas_shape, dtype=SOMA_ATLAS_DTYPE)
        owner_table = np.empty(0, dtype=SOMA_ATLAS_OWNER_DTYPE)
        provenance = np.empty(0, dtype=SOMA_ATLAS_PROVENANCE_DTYPE)
        return atlas, owner_table, provenance

    # 1. Pairwise containment-based merge.
    uf = _UnionFind(n)
    i_idx, j_idx = _bbox_overlap_pairs(candidates)
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        inter = _intersection_area(candidates[i], candidates[j])
        if inter == 0:
            continue
        area_i = candidates[i]["area"]
        area_j = candidates[j]["area"]
        if max(inter / area_i, inter / area_j) >= merge_containment_threshold:
            uf.union(i, j)
    component_of, n_components = uf.labels()
    gids = component_of.astype(SOMA_ATLAS_DTYPE) + 1  # 1-indexed

    # 2. First pass: paint pixels. For each candidate, write its gid into empty
    # pixels; mark pixels already claimed by a different gid as contested.
    atlas = np.zeros(canvas_shape, dtype=SOMA_ATLAS_DTYPE)
    contested = np.zeros(canvas_shape, dtype=bool)
    # Flat parallel arrays of competing-gid entries: one entry per
    # (contested pixel, competing gid). Deduplicated per-region after labelling.
    comp_y_chunks = []
    comp_x_chunks = []
    comp_gid_chunks = []

    for cand_idx, cand in enumerate(candidates):
        y0, x0, y1, x1 = cand["bbox"]
        gid = int(gids[cand_idx])
        local = cand["local_mask"]
        atlas_view = atlas[y0:y1, x0:x1]
        existing = atlas_view[local]  # advanced indexing -> copy
        empty_mask = existing == 0
        diff_mask = (existing != 0) & (existing != gid)
        if empty_mask.any():
            existing[empty_mask] = gid
            atlas_view[local] = existing
        if diff_mask.any():
            local_idx = np.argwhere(local)
            diff_idx = local_idx[diff_mask]
            py = (y0 + diff_idx[:, 0]).astype(np.int64)
            px = (x0 + diff_idx[:, 1]).astype(np.int64)
            existing_gids = atlas[py, px].astype(np.int64)
            contested[py, px] = True
            comp_y_chunks.append(py)
            comp_x_chunks.append(px)
            comp_gid_chunks.append(existing_gids)
            comp_y_chunks.append(py)
            comp_x_chunks.append(px)
            comp_gid_chunks.append(np.full(py.shape, gid, dtype=np.int64))

    # 3. Watershed-resolve each connected contested region.
    if contested.any():
        from scipy.ndimage import find_objects
        from scipy.ndimage import label as cc_label

        # Precompute per-gid centroid statistics over the painted atlas once.
        # Uncontested-core pixels (atlas == gid & ~contested) are the preferred
        # seed; overall atlas==gid pixels are the fallback. Both are fixed
        # after step 2 (the watershed loop only writes contested pixels, which
        # by construction are not in the uncontested core).
        nz_ys, nz_xs = np.where(atlas > 0)
        nz_gids = atlas[nz_ys, nz_xs]
        n_gid_bins = int(nz_gids.max()) + 1 if nz_gids.size else 1
        in_core = ~contested[nz_ys, nz_xs]
        core_count = np.bincount(nz_gids, weights=in_core.astype(np.float64), minlength=n_gid_bins)
        core_sum_y = np.bincount(
            nz_gids, weights=nz_ys.astype(np.float64) * in_core, minlength=n_gid_bins
        )
        core_sum_x = np.bincount(
            nz_gids, weights=nz_xs.astype(np.float64) * in_core, minlength=n_gid_bins
        )
        overall_count = np.bincount(nz_gids, minlength=n_gid_bins)
        overall_sum_y = np.bincount(
            nz_gids, weights=nz_ys.astype(np.float64), minlength=n_gid_bins
        )
        overall_sum_x = np.bincount(
            nz_gids, weights=nz_xs.astype(np.float64), minlength=n_gid_bins
        )

        cc, n_cc = cc_label(contested)
        objects = find_objects(cc)  # per-region (slice_y, slice_x) bboxes

        # Vectorised per-region competing-gid sets: assign each flat entry to
        # its connected component, sort, and slice. Replaces a Python loop
        # over every contested pixel.
        if comp_y_chunks:
            comp_y = np.concatenate(comp_y_chunks)
            comp_x = np.concatenate(comp_x_chunks)
            comp_gid = np.concatenate(comp_gid_chunks)
        else:
            comp_y = np.empty(0, dtype=np.int64)
            comp_x = np.empty(0, dtype=np.int64)
            comp_gid = np.empty(0, dtype=np.int64)
        region_of_entry = (
            cc[comp_y, comp_x] if comp_y.size else np.empty(0, dtype=cc.dtype)
        )
        order = np.argsort(region_of_entry, kind="stable")
        region_sorted = region_of_entry[order]
        gid_sorted = comp_gid[order]
        boundaries = np.searchsorted(
            region_sorted, np.arange(1, n_cc + 2), side="left"
        )

        for r_idx in range(n_cc):
            region_id = r_idx + 1
            slc = objects[r_idx]
            if slc is None:
                continue
            s = int(boundaries[r_idx])
            e = int(boundaries[r_idx + 1])
            if e <= s:
                continue
            unique_gids = np.unique(gid_sorted[s:e])
            gids_here = {int(g) for g in unique_gids.tolist() if g != 0}
            if not gids_here:
                continue
            sy, sx = slc[0], slc[1]
            y_off = int(sy.start)
            x_off = int(sx.start)
            region_local_mask = cc[sy, sx] == region_id
            ys_local, xs_local = np.where(region_local_mask)
            if ys_local.size == 0:
                continue
            fallback_cy = float(ys_local.mean()) + y_off
            fallback_cx = float(xs_local.mean()) + x_off
            centroids = {}
            for gid in gids_here:
                if gid < n_gid_bins and core_count[gid] > 0:
                    cy = float(core_sum_y[gid] / core_count[gid])
                    cx = float(core_sum_x[gid] / core_count[gid])
                elif gid < n_gid_bins and overall_count[gid] > 0:
                    cy = float(overall_sum_y[gid] / overall_count[gid])
                    cx = float(overall_sum_x[gid] / overall_count[gid])
                else:
                    cy = fallback_cy
                    cx = fallback_cx
                centroids[gid] = (cy, cx)
            atlas = _watershed_contested_region(
                atlas, region_local_mask, y_off, x_off, centroids
            )

    # 4. Owner-tile precomputation: for each gid, consider barcode-ref tiles that
    #    actually contain gid pixels. Pick the closest tile centre first, then the
    #    tile with more gid pixels, then lex `(tx, ty)`.
    H_tile, W_tile = int(tile_shape[0]), int(tile_shape[1])
    tile_centers_y = tile_origins[:, :, 0] + H_tile / 2.0
    tile_centers_x = tile_origins[:, :, 1] + W_tile / 2.0
    owner_rows = []
    for prop in regionprops(atlas.astype(np.int64, copy=False)):
        gid = int(prop.label)
        cy, cx = float(prop.centroid[0]), float(prop.centroid[1])
        gy0, gx0, gy1, gx1 = (int(v) for v in prop.bbox)
        bbox_overlaps_tile = (
            (tile_origins[:, :, 0] + H_tile > gy0)
            & (tile_origins[:, :, 0] < gy1)
            & (tile_origins[:, :, 1] + W_tile > gx0)
            & (tile_origins[:, :, 1] < gx1)
        )
        tx_grid, ty_grid = np.where(bbox_overlaps_tile)
        gid_local = atlas[gy0:gy1, gx0:gx1] == gid
        pixel_counts = []
        kept_tx = []
        kept_ty = []
        for cand_tx, cand_ty in zip(tx_grid.tolist(), ty_grid.tolist()):
            ty0 = int(tile_origins[cand_tx, cand_ty, 0])
            tx0 = int(tile_origins[cand_tx, cand_ty, 1])
            oy0 = max(gy0, ty0)
            ox0 = max(gx0, tx0)
            oy1 = min(gy1, ty0 + H_tile)
            ox1 = min(gx1, tx0 + W_tile)
            if oy1 <= oy0 or ox1 <= ox0:
                continue
            count = int(gid_local[oy0 - gy0 : oy1 - gy0, ox0 - gx0 : ox1 - gx0].sum())
            if count == 0:
                continue
            kept_tx.append(int(cand_tx))
            kept_ty.append(int(cand_ty))
            pixel_counts.append(count)
        if pixel_counts:
            tx_grid = np.asarray(kept_tx, dtype=int)
            ty_grid = np.asarray(kept_ty, dtype=int)
            pixel_counts = np.asarray(pixel_counts, dtype=int)
            d2_to_tile_center = (tile_centers_y[tx_grid, ty_grid] - cy) ** 2 + (
                tile_centers_x[tx_grid, ty_grid] - cx
            ) ** 2
            order = np.lexsort((ty_grid, tx_grid, -pixel_counts, d2_to_tile_center))
            owner_tx = int(tx_grid[order[0]])
            owner_ty = int(ty_grid[order[0]])
        else:
            # Soma is outside every barcode-reference tile rectangle, so no
            # registered barcode crop can contain it.
            owner_tx = -1
            owner_ty = -1
        owner_rows.append(
            (gid, owner_tx, owner_ty, int(prop.area), cy, cx)
        )
    owner_table = np.array(owner_rows, dtype=SOMA_ATLAS_OWNER_DTYPE)

    # 5. Provenance: gid -> list of source candidates that fed it.
    provenance_rows = []
    for cand_idx, cand in enumerate(candidates):
        gid = int(gids[cand_idx])
        src_roi, src_tx, src_ty = cand["source_id"]
        provenance_rows.append(
            (
                gid,
                int(src_roi),
                int(src_tx),
                int(src_ty),
                int(cand["source_label"]),
                int(cand["area"]),
            )
        )
    provenance = np.array(provenance_rows, dtype=SOMA_ATLAS_PROVENANCE_DTYPE)

    return atlas, owner_table, provenance


@slurm_it(
    conda_env="iss-preprocess",
    slurm_options={"mem": "64GB", "time": "4:00:00"},
    print_job_id=True,
)
def build_soma_atlas(
    data_path,
    roi,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    mask_suffix="",
    merge_containment_threshold=0.5,
    output_root=None,
    force=False,
):
    """Build the canonical ROI-global soma atlas for one ROI.

    Reads the per-tile raw segmentation masks by default, warps them directly
    into the barcode-reference frame, merges tile-border duplicates by
    containment (NOT symmetric IoU), arbitrates pixel overlaps via
    distance-transform watershed, and writes:

    - `<output_root>/cells/soma_atlas_{roi}.npy` (uint32 label image)
    - `<output_root>/cells/soma_atlas_{roi}_meta.npz` (tile_origins, tile_shape,
      canvas size, owner_table, provenance, build parameters)

    Args:
        data_path (str): Relative dataset path.
        roi (int): ROI index.
        segmentation_prefix (str, optional): Defaults to `ops['segmentation_acquisition']`.
        reference_prefix (str, optional): Defaults to `ops['reference_prefix']`.
        corrected_shifts (str, optional): Defaults to `ops['corrected_shifts']`.
        mask_suffix (str, optional): Mask suffix passed to `load_mask_by_coors`.
            Defaults to "" so soma atlases read raw masks.
        merge_containment_threshold (float): Union two source labels when
            `max(intersection/area_A, intersection/area_B) >= threshold`. Default 0.5.
        output_root (str | Path | None): Override the output dir; defaults to the
            canonical `processed_path`. Used by the verification flow on
            BRAC11398.3d to keep the canonical data untouched.
        force (bool): Overwrite an existing atlas npy if present. Defaults to False.

    Returns:
        dict with keys `atlas` (the path), `meta` (the path), `n_somata`.
    """
    cfg = validate_soma_workflow_config(data_path)
    if segmentation_prefix is None:
        segmentation_prefix = cfg["segmentation_acquisition"]
    if reference_prefix is None:
        reference_prefix = cfg["reference_prefix"]
    if corrected_shifts is None:
        corrected_shifts = cfg["corrected_shifts"]
    ops = load_ops(data_path)
    channels = ops["cellpose_channels"]

    paths = get_soma_atlas_paths(data_path, roi, output_root=output_root)
    paths["cells_dir"].mkdir(parents=True, exist_ok=True)
    if paths["atlas"].exists() and not force:
        raise FileExistsError(
            f"Atlas {paths['atlas']} already exists. Re-run with force=True to overwrite."
        )

    tile_origins, tile_shape, ntiles = _load_roi_tile_origins(
        data_path, reference_prefix, roi
    )
    # Pass 1: compose each tile's source->ROI-global affine and accumulate
    # transformed bounds. Source tile pixel dimensions are constant across the
    # acquisition so we read them once from the segmentation prefix's within
    # shifts file. Tile filtering happens in Pass 2: missing or empty masks
    # only contribute an extra bbox to `bounds`, which is harmless against the
    # 4 px canvas padding.
    src_shifts_path = (
        get_processed_path(data_path)
        / "reg"
        / f"{segmentation_prefix}_within"
        / f"{segmentation_prefix}_{int(roi)}_shifts.npz"
    )
    if not src_shifts_path.exists():
        raise FileNotFoundError(
            f"Source within shifts not found at {src_shifts_path}; run "
            "register_within_acquisition for the segmentation prefix first."
        )
    source_tile_shape = tuple(int(s) for s in np.load(src_shifts_path)["tile_shape"])

    bounds = [tuple(b) for b in _reference_tile_bounds(tile_origins, tile_shape)]
    tile_records = []
    n_tx, n_ty = int(ntiles[0]), int(ntiles[1])
    for tx in range(n_tx):
        for ty in range(n_ty):
            print(f"Pass 1: composing transform for tile (roi={roi}, tx={tx}, ty={ty})...")
            tile_coors = (int(roi), tx, ty)
            tform = _get_mask_registration_matrix(
                data_path=data_path,
                prefix=segmentation_prefix,
                tile_coors=tile_coors,
                corrected_shifts=corrected_shifts,
                channels=channels,
            )
            global_matrix = _compose_mask_to_global_matrix(
                tform,
                tile_origin_yx=tile_origins[tx, ty],
            )
            bounds.append(_transformed_tile_bounds(source_tile_shape, global_matrix))
            tile_records.append((tile_coors, global_matrix))

    canvas_global_origin_yx, canvas_shape = _canvas_from_bounds(bounds, padding=4)
    tile_origins_array = tile_origins - canvas_global_origin_yx

    # Pass 2: warp each tile into a bbox-sized local array (not canvas-sized),
    # reduce to compact per-region candidate dicts with canvas-coord bboxes,
    # then drop the local array. Peak extra memory ~ one tile bbox.
    candidates = []
    for tile_coors, global_matrix in tile_records:
        print(f"Pass 2: warping tile (roi={roi}, tx={tile_coors[1]}, ty={tile_coors[2]}) into local bbox and extracting candidates...")
        try:
            src = load_mask_by_coors(
                data_path,
                prefix=segmentation_prefix,
                tile_coors=tile_coors,
                suffix=mask_suffix,
            )
        except FileNotFoundError:
            continue
        if src is None or not np.any(src):
            continue
        warped_local, y0c, x0c = _warp_tile_to_local_bbox(
            src,
            global_matrix=global_matrix,
            canvas_global_origin_yx=canvas_global_origin_yx,
            canvas_shape=canvas_shape,
        )
        candidates.extend(
            _candidates_from_local_warp(tile_coors, warped_local, y0c, x0c)
        )
        del warped_local
    print(f"Pass 2: collected {len(candidates)} initial candidates across all tiles; merging...")
    atlas, owner_table, provenance = _assemble_soma_atlas_arrays(
        candidates=candidates,
        tile_origins=tile_origins_array,
        tile_shape=tile_shape,
        canvas_shape=canvas_shape,
        merge_containment_threshold=merge_containment_threshold,
    )
    print(f"Pass 3: merged into {len(owner_table)} atlas somata; saving atlas and metadata...")
    # Atomic writes: write to tmp, then rename.
    atlas_tmp = paths["atlas"].with_suffix(".npy.tmp")
    meta_tmp = paths["meta"].with_suffix(".npz.tmp")
    with atlas_tmp.open("wb") as fh:
        np.save(fh, atlas)
    with meta_tmp.open("wb") as fh:
        np.savez(
            fh,
            tile_origins=tile_origins,
            tile_origins_global=tile_origins,
            tile_origins_array=tile_origins_array,
            tile_shape=np.asarray(tile_shape, dtype=int),
            canvas_shape=np.asarray(canvas_shape, dtype=int),
            canvas_global_origin_yx=np.asarray(canvas_global_origin_yx, dtype=int),
            owner_table=owner_table,
            provenance=provenance,
            segmentation_prefix=str(segmentation_prefix),
            mask_suffix=str(mask_suffix),
            reference_prefix=str(reference_prefix),
            corrected_shifts=str(corrected_shifts),
            merge_containment_threshold=float(merge_containment_threshold),
            roi=int(roi),
        )
    atlas_tmp.replace(paths["atlas"])
    meta_tmp.replace(paths["meta"])

    print(f"Final: built soma atlas with {len(owner_table)} somata; saving...")
    return {
        "atlas": paths["atlas"],
        "meta": paths["meta"],
        "n_somata": int(len(owner_table)),
    }


def build_soma_atlases(
    data_path,
    segmentation_prefix=None,
    reference_prefix=None,
    corrected_shifts=None,
    mask_suffix="",
    merge_containment_threshold=0.5,
    output_root=None,
    force=False,
    use_slurm=True,
    slurm_folder=None,
):
    """Build a soma atlas for every ROI in a chamber.

    Submits one slurm job per ROI when `use_slurm=True`. Returns a dict keyed by
    ROI index with the value returned by each `build_soma_atlas` call (or its
    slurm job id when slurm-submitted).
    """
    validate_soma_workflow_config(data_path)
    roi_dims = get_roi_dimensions(data_path)
    if slurm_folder is None:
        slurm_folder = Path.home() / "slurm_logs" / data_path / "soma_atlas"
    slurm_folder = Path(slurm_folder)
    slurm_folder.mkdir(parents=True, exist_ok=True)

    out = {}
    for roi in roi_dims[:, 0]:
        roi_int = int(roi)
        out[roi_int] = build_soma_atlas(
            data_path,
            roi=roi_int,
            segmentation_prefix=segmentation_prefix,
            reference_prefix=reference_prefix,
            corrected_shifts=corrected_shifts,
            mask_suffix=mask_suffix,
            merge_containment_threshold=merge_containment_threshold,
            output_root=output_root,
            force=force,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"build_soma_atlas_{roi_int}",
        )
    return out


def _meta_scalar(meta, key):
    """Return a scalar npz metadata value as a Python object, or None if absent."""
    if key not in meta.files:
        return None
    value = meta[key]
    if np.shape(value) == ():
        return value.item()
    if np.size(value) == 1:
        return np.ravel(value)[0].item()
    return value


def _validate_soma_atlas_meta(
    meta,
    expected_reference_prefix=None,
    expected_corrected_shifts=None,
):
    """Validate that an atlas was built for the requested barcode-call frame."""
    checks = (
        ("reference_prefix", expected_reference_prefix),
        ("corrected_shifts", expected_corrected_shifts),
    )
    for key, expected in checks:
        if expected is None:
            continue
        actual = _meta_scalar(meta, key)
        if actual is None:
            raise ValueError(
                f"Soma atlas metadata is missing {key}; rebuild the atlas before "
                "using it for barcode calling."
            )
        if str(actual) != str(expected):
            raise ValueError(
                f"Soma atlas {key} mismatch: atlas was built with {actual!r}, "
                f"but this caller expected {expected!r}."
            )


def load_soma_atlas_tile(
    data_path,
    tile_coors,
    output_root=None,
    expected_reference_prefix=None,
    expected_corrected_shifts=None,
):
    """Return the per-tile soma mask cropped from the ROI's canonical atlas.

    The returned uint32 array has each soma appearing in exactly one tile (its
    precomputed owner). Soma labels are stable global IDs. If expected metadata
    values are provided, the atlas must have been built against the same barcode
    reference prefix and shift set used to load the sequencing stack.
    """
    roi, tx, ty = int(tile_coors[0]), int(tile_coors[1]), int(tile_coors[2])
    paths = get_soma_atlas_paths(data_path, roi, output_root=output_root)
    if not paths["atlas"].exists() or not paths["meta"].exists():
        raise FileNotFoundError(
            f"Soma atlas for ROI {roi} not found at {paths['atlas']}. "
            "Run build_soma_atlas / build_soma_atlases first."
        )
    atlas = np.load(paths["atlas"], mmap_mode="r")
    meta = np.load(paths["meta"], allow_pickle=False)
    _validate_soma_atlas_meta(
        meta,
        expected_reference_prefix=expected_reference_prefix,
        expected_corrected_shifts=expected_corrected_shifts,
    )
    if "tile_origins_array" in meta.files:
        tile_origins_array = meta["tile_origins_array"]
    else:
        tile_origins_global = meta["tile_origins"]
        if "canvas_global_origin_yx" in meta.files:
            canvas_global_origin_yx = meta["canvas_global_origin_yx"]
        else:
            canvas_global_origin_yx = np.zeros(2, dtype=int)
        tile_origins_array = tile_origins_global - canvas_global_origin_yx
    tile_shape = meta["tile_shape"]
    owner_table = meta["owner_table"]
    y0, x0 = int(tile_origins_array[tx, ty, 0]), int(tile_origins_array[tx, ty, 1])
    H_tile, W_tile = int(tile_shape[0]), int(tile_shape[1])
    crop = np.zeros((H_tile, W_tile), dtype=SOMA_ATLAS_DTYPE)
    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(int(atlas.shape[0]), y0 + H_tile)
    src_x1 = min(int(atlas.shape[1]), x0 + W_tile)
    if src_y1 > src_y0 and src_x1 > src_x0:
        dst_y0 = src_y0 - y0
        dst_x0 = src_x0 - x0
        crop[
            dst_y0 : dst_y0 + (src_y1 - src_y0),
            dst_x0 : dst_x0 + (src_x1 - src_x0),
        ] = atlas[src_y0:src_y1, src_x0:src_x1]
    keep_mask = (owner_table["owner_tx"] == tx) & (owner_table["owner_ty"] == ty)
    keep_gids = owner_table["gid"][keep_mask]
    if keep_gids.size == 0:
        return np.zeros_like(crop)
    crop[~np.isin(crop, keep_gids)] = 0
    return crop
