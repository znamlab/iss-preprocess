import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import yaml
from image_tools.registration.phase_correlation import (
    estimate_rotation_and_scale,
    phase_correlation,
)
from scipy.optimize import dual_annealing
from skimage import exposure
from skimage.filters import window
from skimage.transform import AffineTransform, warp
from tqdm.auto import tqdm
from znamutils import slurm_it

from ..io import get_pixel_size, get_processed_path
from ..io.load import find_roi_position_on_cryostat
from .stitch import stitch_tiles

ORIENTATIONS = ("identity", "rot180", "flip_ud", "flip_lr")
PAIRWISE_STATE_VERSION = 1
GLOBAL_TRANSFORM_VERSION = 1
PAIR_RECORDS_SUBDIR = "pair_records"
SLURM_SUBDIR = "tangential_volume_registration"


def _get_volume_job_logger(log_dir, log_name="volume_registration.log"):
    """Return a file logger for long-running volume registration jobs."""
    log_path = Path(log_dir) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger_name = f"{__name__}.{log_path.resolve()}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved_log_path = str(log_path.resolve())
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == Path(resolved_log_path)
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(resolved_log_path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger


def _get_mouse_processed_path(data_path):
    """Return the processed mouse folder for chamber-level or mouse-level inputs."""
    processed_path = get_processed_path(data_path)
    if processed_path.name.startswith("chamber_"):
        return processed_path.parent
    return processed_path


def get_volume_root(data_path):
    """Return the shared mouse-level folder used for tangential volume outputs."""
    return _get_mouse_processed_path(data_path) / "tangential_volume"


def discover_volume_data_paths(data_path):
    """Discover all chamber paths for the mouse that contains `data_path`."""
    rel_path = Path(data_path)
    processed_path = get_processed_path(data_path)
    mouse_processed = _get_mouse_processed_path(data_path)
    if processed_path.name.startswith("chamber_"):
        mouse_relative = rel_path.parent
    else:
        mouse_relative = rel_path

    chamber_dirs = sorted(
        p
        for p in mouse_processed.glob("chamber_*")
        if p.is_dir() and p.name.startswith("chamber_")
    )
    if not chamber_dirs and processed_path.name.startswith("chamber_"):
        return [str(rel_path)]
    return [str(mouse_relative / chamber_dir.name) for chamber_dir in chamber_dirs]


def _parse_overview_filename(fname):
    match = re.search(r"_sl(\d+)_r(\d+)\.ome\.tif$", fname.name)
    if match is None:
        raise ValueError(f"Could not parse slice/roi from {fname}")
    return int(match.group(1)), int(match.group(2))


def _load_overview_metadata(metadata_path):
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r") as fhandle:
        return json.loads(json.dumps(yaml.safe_load(fhandle)))


def discover_volume_slices(data_path):
    """Find all stitched overview slices used for tangential volume registration."""
    entries = []
    for chamber_path in discover_volume_data_paths(data_path):
        processed_path = get_processed_path(chamber_path)
        reg_folder = processed_path / "register_to_ara"
        if not reg_folder.exists():
            continue
        for overview_file in sorted(reg_folder.glob("*.ome.tif")):
            slice_number, roi = _parse_overview_filename(overview_file)
            metadata_path = overview_file.with_suffix(".yml")
            meta = _load_overview_metadata(metadata_path)
            height, width = tifffile.imread(overview_file).shape[:2]
            original_shape = meta.get("original_shape", [height, width])
            entries.append(
                {
                    "data_path": chamber_path,
                    "processed_path": str(processed_path),
                    "overview_file": str(overview_file),
                    "metadata_file": str(metadata_path),
                    "slice_number": slice_number,
                    "roi": roi,
                    "overview_shape_yx": [int(height), int(width)],
                    "original_shape_yx": [int(original_shape[0]), int(original_shape[1])],
                    "downsample_ratio": float(meta.get("downsample_ratio", 1.0)),
                    "overview_pixel_size_um": float(
                        meta.get("pixel_size", meta.get("original_pixel_size", 1.0))
                    ),
                    "original_pixel_size_um": float(
                        meta.get("original_pixel_size", meta.get("pixel_size", 1.0))
                    ),
                }
            )
    entries = sorted(entries, key=lambda entry: entry["slice_number"])
    if not entries:
        raise FileNotFoundError(
            "Could not find any overview slices in `register_to_ara`. "
            "Run the ROI overview generation first."
        )
    return entries


def _center_pad(image, target_shape, cval=0):
    target_y, target_x = target_shape
    ypix, xpix = image.shape[:2]
    if ypix > target_y or xpix > target_x:
        raise ValueError(
            f"Image shape {image.shape[:2]} is larger than target shape {target_shape}"
        )
    pad_y = target_y - ypix
    pad_x = target_x - xpix
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    padded = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=cval,
    )
    return padded, pad_top, pad_left


def _valid_pixels(image, mask=None):
    valid = np.isfinite(image)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid


def _masked_percentile(image, q, mask=None):
    valid = _valid_pixels(image, mask=mask)
    if not np.any(valid):
        return None
    return float(np.percentile(image[valid], q))


def _maybe_apply_clahe(image, mask=None, clip_limit=0.01):
    valid = _valid_pixels(image, mask=mask)
    if not np.any(valid):
        return image
    lo = float(np.min(image[valid]))
    hi = float(np.max(image[valid]))
    if hi <= lo:
        return image
    scaled = np.zeros_like(image, dtype=np.float32)
    scaled[valid] = np.clip((image[valid] - lo) / (hi - lo), 0.0, 1.0)
    clahe = exposure.equalize_adapthist(scaled, clip_limit=clip_limit)
    out = np.zeros_like(image, dtype=np.float32)
    out[valid] = clahe[valid]
    return out


def _preprocess_for_registration(
    image,
    mask=None,
    background_mode="percentile_subtract",
    background_percentile=5.0,
    clip_percentile=99.5,
    normalize_mode="contrast_stretch",
):
    """Apply symmetric preprocessing for automatic registration."""
    image = image.astype(np.float32, copy=True)
    valid = _valid_pixels(image, mask=mask)
    if not np.any(valid):
        return image

    if background_mode == "percentile_subtract":
        offset = _masked_percentile(image, background_percentile, mask=mask)
        if offset is not None:
            image[valid] = image[valid] - offset
        image[image < 0] = 0
    elif background_mode != "none":
        raise ValueError(f"Unknown background_mode: {background_mode}")

    if normalize_mode == "contrast_stretch":
        lo = _masked_percentile(image, 1.0, mask=mask)
        hi = _masked_percentile(image, 99.0, mask=mask)
        if lo is not None and hi is not None and hi > lo:
            image[valid] = np.clip((image[valid] - lo) / (hi - lo), 0.0, 1.0)
    elif normalize_mode == "clahe":
        image = _maybe_apply_clahe(image, mask=mask)
        valid = _valid_pixels(image, mask=mask)
    elif normalize_mode != "none":
        raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    if clip_percentile is not None:
        clip_value = _masked_percentile(image, clip_percentile, mask=mask)
        if clip_value is not None and clip_value > 0:
            image[valid] = np.clip(image[valid], 0, clip_value)
    return image


def _apply_translation_taper(image, apply_translation_hann=True):
    if not apply_translation_hann:
        return image
    return image * window("hann", image.shape).astype(np.float32)


def export_unregistered_volume_stack(
    data_path,
    output_name="unregistered_slices.npz",
    overwrite=False,
):
    """Export centered overview slices as a single 3D stack for manual masking."""
    entries = discover_volume_slices(data_path)
    volume_root = get_volume_root(data_path)
    volume_root.mkdir(parents=True, exist_ok=True)
    logger = _get_volume_job_logger(volume_root)
    target = volume_root / output_name
    if target.exists() and not overwrite:
        logger.info("Skipping stack export because %s already exists", target)
        return target

    target_shape = (
        max(entry["overview_shape_yx"][0] for entry in entries),
        max(entry["overview_shape_yx"][1] for entry in entries),
    )
    logger.info(
        "Exporting %d overview slices to %s with target shape %s",
        len(entries),
        target,
        target_shape,
    )
    images = np.zeros((len(entries), *target_shape), dtype=np.float32)
    default_masks = np.zeros((len(entries), *target_shape), dtype=bool)
    user_masks = np.zeros((len(entries), *target_shape), dtype=bool)
    with tqdm(
        total=len(entries) + 1,
        desc="Exporting overview slices",
        dynamic_ncols=True,
    ) as pbar:
        for i, entry in enumerate(entries):
            try:
                image = tifffile.imread(entry["overview_file"]).astype(np.float32)
                padded, pad_top, pad_left = _center_pad(image, target_shape)
                images[i] = padded
                entry["pad_top"] = int(pad_top)
                entry["pad_left"] = int(pad_left)
                logger.info(
                    "Exported slice %s roi %s from %s",
                    entry["slice_number"],
                    entry["roi"],
                    entry["overview_file"],
                )
                pbar.update(1)
            except Exception:
                logger.exception(
                    "Failed exporting slice %s roi %s from %s",
                    entry.get("slice_number"),
                    entry.get("roi"),
                    entry.get("overview_file"),
                )
                raise

        pbar.set_description("Compressing overview stack")
        logger.info("Compressing and saving unregistered volume stack to %s", target)

        manifest = {
            "target_shape_yx": [int(target_shape[0]), int(target_shape[1])],
            "entries": entries,
        }
        np.savez_compressed(
            target,
            images=images,
            default_masks=default_masks.astype(np.uint8),
            user_masks=user_masks.astype(np.uint8),
            manifest_json=json.dumps(manifest),
        )
        pbar.update(1)
    logger.info("Saved unregistered volume stack to %s", target)
    return target


def load_unregistered_volume_stack(stack_path):
    """Load the saved unregistered slice stack and its manifest."""
    stack_path = Path(stack_path)
    data = np.load(stack_path, allow_pickle=False)
    manifest = json.loads(str(data["manifest_json"].item()))
    images = data["images"].astype(np.float32)
    default_masks = data["default_masks"].astype(bool)
    if "user_masks" in data.files:
        user_masks = data["user_masks"].astype(bool)
    else:
        user_masks = default_masks.copy()
    return {
        "stack_path": str(stack_path),
        "images": images,
        "default_masks": default_masks,
        "user_masks": user_masks,
        "manifest": manifest,
    }


def _translation_matrix(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def _rotation_matrix(angle_deg, shape):
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return _translation_matrix(cx, cy) @ rotation @ _translation_matrix(-cx, -cy)


def _orientation_matrix(shape, orientation):
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    if orientation == "identity":
        return np.eye(3)
    if orientation == "rot180":
        return _rotation_matrix(180.0, shape)
    if orientation == "flip_ud":
        return np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 2.0 * cy], [0.0, 0.0, 1.0]])
    if orientation == "flip_lr":
        return np.array([[-1.0, 0.0, 2.0 * cx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unknown orientation: {orientation}")


def warp_with_affine(image, matrix, output_shape=None, order=1, cval=0.0):
    """Warp an image with a 3x3 affine matrix in x/y homogeneous coordinates."""
    if output_shape is None:
        output_shape = image.shape[:2]
    warped = warp(
        image,
        inverse_map=AffineTransform(matrix=matrix).inverse,
        output_shape=output_shape,
        order=order,
        mode="constant",
        cval=cval,
        preserve_range=True,
    )
    return warped.astype(image.dtype if np.issubdtype(image.dtype, np.floating) else np.float32)


def apply_affine_to_points(points_xy, matrix):
    """Apply a 3x3 affine matrix to a set of row-wise x/y points."""
    points_xy = np.asarray(points_xy, dtype=float)
    ones = np.ones((points_xy.shape[0], 1), dtype=float)
    hom = np.hstack([points_xy, ones])
    out = hom @ matrix.T
    out /= out[:, [2]]
    return out[:, :2]


def _candidate_angles(
    fixed_image,
    moving_image,
    rotation_dog=(15, 40),
    upsample_factor=20,
    rotation_hann=True,
):
    normal = estimate_rotation_and_scale(
        fixed_image,
        moving_image,
        dog=rotation_dog,
        estimate_scale=False,
        debug=False,
        upsample_factor=upsample_factor,
        hann=rotation_hann,
        normalization=None,
    )[0]
    flipped = estimate_rotation_and_scale(
        fixed_image,
        np.flipud(moving_image),
        dog=rotation_dog,
        estimate_scale=False,
        debug=False,
        upsample_factor=upsample_factor,
        hann=rotation_hann,
        normalization=None,
    )[0]
    return {
        "identity": -float(normal),
        "rot180": -float(normal),
        "flip_ud": -float(flipped),
        "flip_lr": -float(flipped),
    }


def _phase_correlation_score(fixed_image, moving_image, fixed_mask, moving_mask):
    shift_yx, score, _, overlap = phase_correlation(
        fixed_image,
        moving_image,
        fixed_mask=fixed_mask.astype(bool),
        moving_mask=moving_mask.astype(bool),
    )
    overlap_arr = np.asarray(overlap)
    overlap_value = float(overlap_arr.max()) if overlap_arr.ndim > 0 else float(overlap_arr)
    return np.asarray(shift_yx, dtype=float), float(score), overlap_value


def _masked_overlap_score(fixed_image, fixed_mask, moving_image, moving_mask):
    overlap = fixed_mask & moving_mask & np.isfinite(fixed_image) & np.isfinite(moving_image)
    n_overlap = int(np.sum(overlap))
    if n_overlap < 50:
        return 1e6
    fixed = fixed_image[overlap].astype(float, copy=False)
    moving = moving_image[overlap].astype(float, copy=False)
    fixed = fixed - fixed.mean()
    moving = moving - moving.mean()
    fixed_std = fixed.std()
    moving_std = moving.std()
    if fixed_std == 0 or moving_std == 0:
        return 1e6
    corr = np.mean((fixed / fixed_std) * (moving / moving_std))
    return -float(corr)


def _manual_refine(
    fixed_image,
    moving_image,
    fixed_mask,
    moving_mask,
    orientation,
    initial_dx,
    initial_dy,
    initial_angle,
    radius,
    angle_radius_deg=5.0,
    maxiter=20,
    maxfun=120,
):
    shape = fixed_image.shape
    orient = _orientation_matrix(shape, orientation)

    def objective(params):
        dx, dy, angle_deg = params
        matrix = _translation_matrix(dx, dy) @ _rotation_matrix(angle_deg, shape) @ orient
        warped_image = warp_with_affine(moving_image, matrix, output_shape=shape, order=1)
        warped_mask = warp_with_affine(
            moving_mask.astype(np.float32), matrix, output_shape=shape, order=0
        ) > 0.5
        return _masked_overlap_score(
            fixed_image,
            fixed_mask,
            warped_image,
            warped_mask,
        )

    result = dual_annealing(
        objective,
        bounds=[
            (initial_dx - radius, initial_dx + radius),
            (initial_dy - radius, initial_dy + radius),
            (initial_angle - angle_radius_deg, initial_angle + angle_radius_deg),
        ],
        x0=np.array([initial_dx, initial_dy, initial_angle], dtype=float),
        maxiter=maxiter,
        maxfun=maxfun,
        no_local_search=True,
    )
    dx, dy, angle_deg = result.x
    matrix = (
        _translation_matrix(dx, dy)
        @ _rotation_matrix(angle_deg, shape)
        @ _orientation_matrix(shape, orientation)
    )
    warped_image = warp_with_affine(moving_image, matrix, output_shape=shape, order=1)
    warped_mask = warp_with_affine(
        moving_mask.astype(np.float32), matrix, output_shape=shape, order=0
    ) > 0.5
    return {
        "matrix": matrix,
        "shift_xy": [float(dx), float(dy)],
        "angle_deg": float(angle_deg),
        "score": float(-objective(result.x)),
        "warped_image": warped_image,
        "warped_mask": warped_mask,
        "optimisation_value": float(result.fun),
        "optimisation_success": bool(getattr(result, "success", True)),
    }


def estimate_pairwise_registration(
    fixed_image,
    moving_image,
    fixed_mask=None,
    moving_mask=None,
    orientation="auto",
    manual_angle_deg=None,
    manual_shift_xy=None,
    local_search_radius=None,
    angle_radius_deg=5.0,
    rotation_dog=(15, 40),
    rotation_hann=True,
    apply_translation_hann=True,
    background_mode="percentile_subtract",
    background_percentile=5.0,
    clip_percentile=99.5,
    normalize_mode="contrast_stretch",
    upsample_factor=20,
):
    """Estimate a 2D transform from a moving slice to a fixed slice."""
    if fixed_mask is None:
        fixed_mask = fixed_image > 0
    if moving_mask is None:
        moving_mask = moving_image > 0
    fixed_mask = fixed_mask.astype(bool)
    moving_mask = moving_mask.astype(bool)
    fixed_image = _preprocess_for_registration(
        fixed_image,
        mask=fixed_mask,
        background_mode=background_mode,
        background_percentile=background_percentile,
        clip_percentile=clip_percentile,
        normalize_mode=normalize_mode,
    )
    moving_image = _preprocess_for_registration(
        moving_image,
        mask=moving_mask,
        background_mode=background_mode,
        background_percentile=background_percentile,
        clip_percentile=clip_percentile,
        normalize_mode=normalize_mode,
    )
    shape = fixed_image.shape
    if moving_image.shape != shape:
        raise ValueError("Fixed and moving images must have the same shape")

    base_angles = _candidate_angles(
        fixed_image,
        moving_image,
        rotation_dog=rotation_dog,
        upsample_factor=upsample_factor,
        rotation_hann=rotation_hann,
    )
    if orientation == "auto":
        candidate_orientations = ORIENTATIONS
    else:
        candidate_orientations = (orientation,)

    best = None
    for orientation_name in candidate_orientations:
        angle_deg = (
            float(manual_angle_deg)
            if manual_angle_deg is not None
            else float(base_angles[orientation_name])
        )
        transform_wo_shift = (
            _rotation_matrix(angle_deg, shape)
            @ _orientation_matrix(shape, orientation_name)
        )
        warped_moving = warp_with_affine(
            moving_image,
            transform_wo_shift,
            output_shape=shape,
            order=1,
        )
        warped_mask = warp_with_affine(
            moving_mask.astype(np.float32),
            transform_wo_shift,
            output_shape=shape,
            order=0,
        ) > 0.5
        shift_yx, score, overlap = _phase_correlation_score(
            _apply_translation_taper(
                fixed_image, apply_translation_hann=apply_translation_hann
            ),
            _apply_translation_taper(
                warped_moving, apply_translation_hann=apply_translation_hann
            ),
            fixed_mask,
            warped_mask,
        )
        dx = float(shift_yx[1])
        dy = float(shift_yx[0])
        if manual_shift_xy is not None:
            dx = float(manual_shift_xy[0])
            dy = float(manual_shift_xy[1])
        matrix = _translation_matrix(dx, dy) @ transform_wo_shift
        candidate = {
            "orientation": orientation_name,
            "angle_deg": float(angle_deg),
            "shift_xy": [dx, dy],
            "score": float(score),
            "overlap": float(overlap),
            "matrix": matrix,
            "warped_image": warp_with_affine(
                moving_image,
                matrix,
                output_shape=shape,
                order=1,
            ),
            "warped_mask": warp_with_affine(
                moving_mask.astype(np.float32),
                matrix,
                output_shape=shape,
                order=0,
            ) > 0.5,
            "method": "auto_phase_correlation" if manual_shift_xy is None else "manual_seed",
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not estimate pairwise registration")

    if local_search_radius is not None and float(local_search_radius) > 0:
        refined = _manual_refine(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            orientation=best["orientation"],
            initial_dx=float(best["shift_xy"][0]),
            initial_dy=float(best["shift_xy"][1]),
            initial_angle=float(best["angle_deg"]),
            radius=float(local_search_radius),
            angle_radius_deg=float(angle_radius_deg),
        )
        best.update(refined)
        best["method"] = "local_simulated_annealing"

    best["matrix"] = np.asarray(best["matrix"], dtype=float)
    return best


def _pair_key(fixed_slice, moving_slice):
    return f"{int(fixed_slice):03d}_{int(moving_slice):03d}"


def load_pairwise_registration_state(state_path):
    state_path = Path(state_path)
    if not state_path.exists():
        return {
            "version": PAIRWISE_STATE_VERSION,
            "bad_slices": [],
            "pairs": {},
        }
    with open(state_path, "r") as fhandle:
        return json.load(fhandle)


def save_pairwise_registration_state(state_path, state):
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as fhandle:
        json.dump(state, fhandle, indent=2, sort_keys=True)


def build_adjacent_slice_pairs(entries, bad_slices=None):
    """Create adjacent slice pairs after removing bad slices."""
    bad_slices = {int(s) for s in (bad_slices or [])}
    active_entries = [
        entry for entry in entries if int(entry["slice_number"]) not in bad_slices
    ]
    active_entries = sorted(active_entries, key=lambda entry: entry["slice_number"])
    return [(active_entries[i + 1], active_entries[i]) for i in range(len(active_entries) - 1)]


def _default_state_path(stack_path):
    return Path(stack_path).with_name("pairwise_registration_state.json")


def _default_pair_records_dir(state_path):
    return Path(state_path).parent / PAIR_RECORDS_SUBDIR


def _default_slurm_folder(state_path, entries=None):
    if entries:
        data_path = entries[0].get("data_path")
        if data_path:
            volume_rel = Path(data_path)
            if volume_rel.name.startswith("chamber_"):
                volume_rel = volume_rel.parent
            return Path.home() / "slurm_logs" / volume_rel / SLURM_SUBDIR
    return Path(state_path).parent / "slurm"


def _slurm_log_options(slurm_folder, scripts_name, slurm_options=None):
    options = dict(slurm_options or {})
    options.setdefault("output", str(Path(slurm_folder) / f"{scripts_name}_%j.out"))
    options.setdefault("error", str(Path(slurm_folder) / f"{scripts_name}_%j.err"))
    return options


def _pair_record_path(pair_records_dir, key):
    return Path(pair_records_dir) / f"pair_{key}.json"


def _atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fhandle:
        json.dump(obj, fhandle, indent=2, sort_keys=True)
    tmp.replace(path)


def _compute_pair_record(
    fixed_entry,
    moving_entry,
    fixed_image,
    moving_image,
    fixed_mask,
    moving_mask,
    existing,
    local_search_radius,
    angle_radius_deg,
    rotation_dog,
    rotation_hann,
    apply_translation_hann,
    background_mode,
    background_percentile,
    clip_percentile,
    normalize_mode,
):
    """Run pairwise registration for a single pair and return the serialised record."""
    fixed_slice = int(fixed_entry["slice_number"])
    moving_slice = int(moving_entry["slice_number"])
    pair_orientation = existing.get("orientation", "auto")
    manual_angle = existing.get("manual_angle_deg")
    manual_shift = existing.get("manual_shift_xy")
    pair_radius = existing.get("local_search_radius", local_search_radius)
    pair_angle_radius = existing.get("angle_radius_deg", angle_radius_deg)
    result = estimate_pairwise_registration(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        orientation=pair_orientation,
        manual_angle_deg=manual_angle,
        manual_shift_xy=manual_shift,
        local_search_radius=pair_radius,
        angle_radius_deg=pair_angle_radius,
        rotation_dog=rotation_dog,
        rotation_hann=rotation_hann,
        apply_translation_hann=apply_translation_hann,
        background_mode=background_mode,
        background_percentile=background_percentile,
        clip_percentile=clip_percentile,
        normalize_mode=normalize_mode,
    )
    status = "accepted" if existing.get("status") == "accepted" else "pending"
    return {
        "fixed_slice": fixed_slice,
        "moving_slice": moving_slice,
        "orientation": result["orientation"],
        "angle_deg": result["angle_deg"],
        "shift_xy": result["shift_xy"],
        "matrix": np.asarray(result["matrix"]).tolist(),
        "score": result["score"],
        "overlap": result.get("overlap"),
        "status": status,
        "manual_angle_deg": manual_angle,
        "manual_shift_xy": manual_shift,
        "local_search_radius": pair_radius,
        "angle_radius_deg": pair_angle_radius,
        "method": result["method"],
    }


def _failed_pair_record(
    fixed_entry,
    moving_entry,
    existing,
    local_search_radius,
    angle_radius_deg,
    errors,
):
    fixed_slice = int(fixed_entry["slice_number"])
    moving_slice = int(moving_entry["slice_number"])
    shift_xy = existing.get("shift_xy")
    if shift_xy is None:
        shift_xy = [0.0, 0.0]
    angle_deg = existing.get("angle_deg")
    if angle_deg is None:
        angle_deg = 0.0
    return {
        "fixed_slice": fixed_slice,
        "moving_slice": moving_slice,
        "orientation": existing.get("orientation", "auto"),
        "angle_deg": float(angle_deg),
        "shift_xy": [float(shift_xy[0]), float(shift_xy[1])],
        "matrix": None,
        "score": None,
        "overlap": None,
        "status": "failed",
        "manual_angle_deg": existing.get("manual_angle_deg"),
        "manual_shift_xy": existing.get("manual_shift_xy"),
        "local_search_radius": existing.get("local_search_radius", local_search_radius),
        "angle_radius_deg": existing.get("angle_radius_deg", angle_radius_deg),
        "method": "failed",
        "error": " | ".join(str(err) for err in errors),
        "attempt_count": len(errors),
    }


def _compute_pair_record_with_retry(
    fixed_entry,
    moving_entry,
    fixed_image,
    moving_image,
    fixed_mask,
    moving_mask,
    existing,
    local_search_radius,
    angle_radius_deg,
    rotation_dog,
    rotation_hann,
    apply_translation_hann,
    background_mode,
    background_percentile,
    clip_percentile,
    normalize_mode,
    logger=None,
):
    errors = []
    for attempt_idx in range(2):
        try:
            return _compute_pair_record(
                fixed_entry=fixed_entry,
                moving_entry=moving_entry,
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                existing=existing,
                local_search_radius=local_search_radius,
                angle_radius_deg=angle_radius_deg,
                rotation_dog=rotation_dog,
                rotation_hann=rotation_hann,
                apply_translation_hann=apply_translation_hann,
                background_mode=background_mode,
                background_percentile=background_percentile,
                clip_percentile=clip_percentile,
                normalize_mode=normalize_mode,
            )
        except Exception as exc:
            errors.append(exc)
            if logger is not None:
                logger.exception(
                    "Attempt %d/2 failed registering pair %s",
                    attempt_idx + 1,
                    _pair_key(
                        fixed_entry["slice_number"],
                        moving_entry["slice_number"],
                    ),
                )
    return _failed_pair_record(
        fixed_entry=fixed_entry,
        moving_entry=moving_entry,
        existing=existing,
        local_search_radius=local_search_radius,
        angle_radius_deg=angle_radius_deg,
        errors=errors,
    )


def _prepare_pairwise_state(
    stack_path,
    state_path,
    force=False,
):
    """Populate bad_slices, active_pairs, entries fields in the pairwise state.

    Returns (state, entries, active_pairs) after reading the stack and any existing
    state. Also persists the updated state to disk so workers can read overrides.
    """
    stack = load_unregistered_volume_stack(stack_path)
    entries = stack["manifest"]["entries"]
    masks = stack["user_masks"]
    state = load_pairwise_registration_state(state_path)
    state["version"] = PAIRWISE_STATE_VERSION
    state["stack_path"] = str(stack_path)
    bad_slices = {int(s) for s in state.get("bad_slices", [])}
    auto_bad_slices = set()
    for i, entry in enumerate(entries):
        if not masks[i].any():
            auto_bad_slices.add(int(entry["slice_number"]))
    bad_slices |= auto_bad_slices
    state["bad_slices"] = sorted(bad_slices)
    state["auto_bad_slices"] = sorted(auto_bad_slices)
    active_pairs = build_adjacent_slice_pairs(entries, bad_slices=bad_slices)
    state["active_pairs"] = [
        [
            _pair_key(fixed["slice_number"], moving["slice_number"]),
            int(fixed["slice_number"]),
            int(moving["slice_number"]),
        ]
        for fixed, moving in active_pairs
    ]
    state["entries"] = entries
    save_pairwise_registration_state(state_path, state)
    try:
        write_bad_slices_to_chamber_ops(
            stack_path=stack_path, state_path=state_path, warn_missing=False
        )
    except Exception:
        pass
    return state, entries, active_pairs


def _pairs_to_process(active_pairs, state, force):
    """Return the subset of active_pairs that need (re)registration."""
    out = []
    pairs = state.get("pairs", {})
    for fixed_entry, moving_entry in active_pairs:
        key = _pair_key(fixed_entry["slice_number"], moving_entry["slice_number"])
        existing = pairs.get(key, {})
        if existing and not force and existing.get("status") == "accepted":
            continue
        out.append((fixed_entry, moving_entry))
    return out


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="32G"))
def register_single_pair(
    stack_path,
    fixed_slice,
    moving_slice,
    state_path=None,
    pair_records_dir=None,
    local_search_radius=None,
    angle_radius_deg=5.0,
    rotation_dog=(15, 40),
    rotation_hann=True,
    apply_translation_hann=True,
    background_mode="percentile_subtract",
    background_percentile=5.0,
    clip_percentile=99.5,
    normalize_mode="contrast_stretch",
):
    """Register a single adjacent slice pair and write its result as pair_<key>.json.

    Designed to be called via znamutils.slurm_it so pairs can run as independent
    SLURM jobs. Reads per-pair overrides (manual_*, orientation, status) from
    `state_path` before running, and writes to `pair_records_dir/pair_<key>.json`
    atomically. Does NOT modify the shared state file - use `reduce_pairwise_state`
    to merge results back.
    """
    stack_path = Path(stack_path)
    if state_path is None:
        state_path = _default_state_path(stack_path)
    if pair_records_dir is None:
        pair_records_dir = _default_pair_records_dir(state_path)
    pair_records_dir = Path(pair_records_dir)
    fixed_slice = int(fixed_slice)
    moving_slice = int(moving_slice)
    key = _pair_key(fixed_slice, moving_slice)

    logger = _get_volume_job_logger(
        pair_records_dir, log_name=f"register_pair_{key}.log"
    )
    stack = load_unregistered_volume_stack(stack_path)
    entries = stack["manifest"]["entries"]
    images = stack["images"]
    masks = stack["user_masks"]
    index_by_slice = {int(entry["slice_number"]): i for i, entry in enumerate(entries)}
    if fixed_slice not in index_by_slice or moving_slice not in index_by_slice:
        raise KeyError(
            f"Slice {fixed_slice} or {moving_slice} not found in {stack_path}"
        )

    state = load_pairwise_registration_state(state_path)
    existing = state.get("pairs", {}).get(key, {})
    fixed_entry = entries[index_by_slice[fixed_slice]]
    moving_entry = entries[index_by_slice[moving_slice]]

    logger.info("Registering pair %s", key)
    record = _compute_pair_record_with_retry(
        fixed_entry=fixed_entry,
        moving_entry=moving_entry,
        fixed_image=images[index_by_slice[fixed_slice]],
        moving_image=images[index_by_slice[moving_slice]],
        fixed_mask=masks[index_by_slice[fixed_slice]],
        moving_mask=masks[index_by_slice[moving_slice]],
        existing=existing,
        local_search_radius=local_search_radius,
        angle_radius_deg=angle_radius_deg,
        rotation_dog=rotation_dog,
        rotation_hann=rotation_hann,
        apply_translation_hann=apply_translation_hann,
        background_mode=background_mode,
        background_percentile=background_percentile,
        clip_percentile=clip_percentile,
        normalize_mode=normalize_mode,
        logger=logger,
    )

    record["key"] = key
    out_path = _pair_record_path(pair_records_dir, key)
    _atomic_write_json(out_path, record)
    if record["status"] == "failed":
        logger.warning(
            "Pair %s failed after %d attempts: %s",
            key,
            int(record.get("attempt_count", 0)),
            record.get("error", "unknown error"),
        )
    else:
        logger.info(
            "Finished pair %s (score %.6f, method %s) -> %s",
            key,
            float(record["score"]),
            record["method"],
            out_path,
        )
    return str(out_path)


@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="8G"))
def reduce_pairwise_state(
    stack_path,
    state_path=None,
    pair_records_dir=None,
    cleanup=False,
    missing_pairs_to_retry=None,
    local_search_radius=None,
    angle_radius_deg=5.0,
    rotation_dog=(15, 40),
    rotation_hann=True,
    apply_translation_hann=True,
    background_mode="percentile_subtract",
    background_percentile=5.0,
    clip_percentile=99.5,
    normalize_mode="contrast_stretch",
):
    """Merge per-pair JSON records into the shared pairwise registration state file.

    For each `pair_<key>.json` in `pair_records_dir`, update `state["pairs"][key]`.
    Widget edits to the same key between a worker read and this reducer run would
    be clobbered - re-run the worker after curation to refresh the per-pair file.
    """
    stack_path = Path(stack_path)
    if state_path is None:
        state_path = _default_state_path(stack_path)
    state_path = Path(state_path)
    if pair_records_dir is None:
        pair_records_dir = _default_pair_records_dir(state_path)
    pair_records_dir = Path(pair_records_dir)

    logger = _get_volume_job_logger(state_path.parent)
    state = load_pairwise_registration_state(state_path)
    state.setdefault("pairs", {})

    for fixed_slice, moving_slice in missing_pairs_to_retry or []:
        key = _pair_key(fixed_slice, moving_slice)
        pair_file = _pair_record_path(pair_records_dir, key)
        if pair_file.exists():
            continue
        logger.warning("Missing pair record for %s; retrying inline before merge", key)
        register_single_pair(
            stack_path=str(stack_path),
            fixed_slice=int(fixed_slice),
            moving_slice=int(moving_slice),
            state_path=str(state_path),
            pair_records_dir=str(pair_records_dir),
            local_search_radius=local_search_radius,
            angle_radius_deg=angle_radius_deg,
            rotation_dog=rotation_dog,
            rotation_hann=rotation_hann,
            apply_translation_hann=apply_translation_hann,
            background_mode=background_mode,
            background_percentile=background_percentile,
            clip_percentile=clip_percentile,
            normalize_mode=normalize_mode,
        )

    pair_files = sorted(pair_records_dir.glob("pair_*.json"))
    logger.info(
        "Merging %d pair records from %s into %s",
        len(pair_files),
        pair_records_dir,
        state_path,
    )
    merged = 0
    for pair_file in pair_files:
        with open(pair_file, "r") as fhandle:
            record = json.load(fhandle)
        key = record.get("key") or _pair_key(
            record["fixed_slice"], record["moving_slice"]
        )
        record.pop("key", None)
        state["pairs"][key] = record
        merged += 1

    save_pairwise_registration_state(state_path, state)
    logger.info("Merged %d pair records into %s", merged, state_path)

    try:
        written = write_bad_slices_to_chamber_ops(
            state_path=state_path, warn_missing=False
        )
        if written:
            logger.info("Updated bad_slices in %d chamber ops.yml files", len(written))
    except Exception:
        logger.exception("Failed mirroring bad_slices into chamber ops.yml")

    if cleanup:
        for pair_file in pair_files:
            pair_file.unlink()
        logger.info("Removed %d merged per-pair files", len(pair_files))

    return Path(state_path)


def register_adjacent_slices(
    stack_path,
    state_path=None,
    force=False,
    local_search_radius=None,
    angle_radius_deg=5.0,
    rotation_dog=(15, 40),
    rotation_hann=True,
    apply_translation_hann=True,
    background_mode="percentile_subtract",
    background_percentile=5.0,
    clip_percentile=99.5,
    normalize_mode="contrast_stretch",
    use_slurm=False,
    slurm_folder=None,
    slurm_options=None,
    scripts_name=None,
    pair_records_dir=None,
    cleanup_pair_records=False,
):
    """Register adjacent slices using the saved unregistered stack and masks.

    When `use_slurm=False` (default) each pair is registered sequentially in this
    process, preserving the previous behavior. When `use_slurm=True`, each pair
    is submitted as an independent SLURM job writing to
    `pair_records_dir/pair_<key>.json`, followed by a reducer job that merges the
    per-pair files into the shared state. A list of job ids is returned instead
    of the state path.
    """
    if state_path is None:
        state_path = _default_state_path(stack_path)
    state_path = Path(state_path)
    logger = _get_volume_job_logger(state_path.parent)

    state, entries, active_pairs = _prepare_pairwise_state(
        stack_path=stack_path,
        state_path=state_path,
        force=force,
    )
    if state.get("auto_bad_slices"):
        logger.info(
            "Excluding %d slices with empty user masks: %s",
            len(state["auto_bad_slices"]),
            state["auto_bad_slices"],
        )
    logger.info(
        "Registering %d adjacent slice pairs from %s into %s",
        len(active_pairs),
        stack_path,
        state_path,
    )

    pairs_to_do = _pairs_to_process(active_pairs, state, force)
    logger.info(
        "%d pairs need registration (force=%s); %d accepted pairs skipped",
        len(pairs_to_do),
        force,
        len(active_pairs) - len(pairs_to_do),
    )

    if pair_records_dir is None:
        pair_records_dir = _default_pair_records_dir(state_path)
    pair_records_dir = Path(pair_records_dir)
    if use_slurm and slurm_folder is None:
        slurm_folder = _default_slurm_folder(state_path, entries=entries)
    if slurm_folder is not None:
        slurm_folder = Path(slurm_folder)

    if use_slurm:
        pair_records_dir.mkdir(parents=True, exist_ok=True)
        slurm_folder.mkdir(parents=True, exist_ok=True)
        pair_job_ids = []
        pairs_submitted = []
        for fixed_entry, moving_entry in pairs_to_do:
            fixed_slice = int(fixed_entry["slice_number"])
            moving_slice = int(moving_entry["slice_number"])
            key = _pair_key(fixed_slice, moving_slice)
            pair_script_name = (scripts_name or "register_pair") + f"_{key}"
            job_id = register_single_pair(
                stack_path=str(stack_path),
                fixed_slice=fixed_slice,
                moving_slice=moving_slice,
                state_path=str(state_path),
                pair_records_dir=str(pair_records_dir),
                local_search_radius=local_search_radius,
                angle_radius_deg=angle_radius_deg,
                rotation_dog=rotation_dog,
                rotation_hann=rotation_hann,
                apply_translation_hann=apply_translation_hann,
                background_mode=background_mode,
                background_percentile=background_percentile,
                clip_percentile=clip_percentile,
                normalize_mode=normalize_mode,
                use_slurm=True,
                slurm_folder=str(slurm_folder),
                slurm_options=_slurm_log_options(
                    slurm_folder, pair_script_name, slurm_options
                ),
                scripts_name=pair_script_name,
            )
            pair_job_ids.append(job_id)
            pairs_submitted.append([fixed_slice, moving_slice])
        reducer_script_name = (scripts_name or "reduce_pairwise") + "_reduce"
        reducer_job_id = reduce_pairwise_state(
            stack_path=str(stack_path),
            state_path=str(state_path),
            pair_records_dir=str(pair_records_dir),
            cleanup=cleanup_pair_records,
            missing_pairs_to_retry=pairs_submitted,
            local_search_radius=local_search_radius,
            angle_radius_deg=angle_radius_deg,
            rotation_dog=rotation_dog,
            rotation_hann=rotation_hann,
            apply_translation_hann=apply_translation_hann,
            background_mode=background_mode,
            background_percentile=background_percentile,
            clip_percentile=clip_percentile,
            normalize_mode=normalize_mode,
            use_slurm=True,
            dependency_type="afterany",
            slurm_folder=str(slurm_folder),
            slurm_options=_slurm_log_options(
                slurm_folder, reducer_script_name, slurm_options
            ),
            scripts_name=reducer_script_name,
            job_dependency=pair_job_ids if pair_job_ids else None,
        )
        logger.info(
            "Submitted %d pair jobs + reducer job %s",
            len(pair_job_ids),
            reducer_job_id,
        )
        return {"pair_job_ids": pair_job_ids, "reducer_job_id": reducer_job_id}

    stack = load_unregistered_volume_stack(stack_path)
    images = stack["images"]
    masks = stack["user_masks"]
    index_by_slice = {int(entry["slice_number"]): i for i, entry in enumerate(entries)}
    pair_records = dict(state.get("pairs", {}))
    for fixed_entry, moving_entry in tqdm(
        pairs_to_do,
        total=len(pairs_to_do),
        desc="Registering adjacent slices",
        dynamic_ncols=True,
    ):
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        key = _pair_key(fixed_slice, moving_slice)
        existing = state.get("pairs", {}).get(key, {})
        logger.info("Registering pair %s", key)
        record = _compute_pair_record_with_retry(
            fixed_entry=fixed_entry,
            moving_entry=moving_entry,
            fixed_image=images[index_by_slice[fixed_slice]],
            moving_image=images[index_by_slice[moving_slice]],
            fixed_mask=masks[index_by_slice[fixed_slice]],
            moving_mask=masks[index_by_slice[moving_slice]],
            existing=existing,
            local_search_radius=local_search_radius,
            angle_radius_deg=angle_radius_deg,
            rotation_dog=rotation_dog,
            rotation_hann=rotation_hann,
            apply_translation_hann=apply_translation_hann,
            background_mode=background_mode,
            background_percentile=background_percentile,
            clip_percentile=clip_percentile,
            normalize_mode=normalize_mode,
            logger=logger,
        )
        pair_records[key] = record
        if record["status"] == "failed":
            logger.warning(
                "Pair %s failed after %d attempts: %s",
                key,
                int(record.get("attempt_count", 0)),
                record.get("error", "unknown error"),
            )
        else:
            logger.info(
                "Finished pair %s with score %.6f using %s",
                key,
                float(record["score"]),
                record["method"],
            )

    state["pairs"] = pair_records
    save_pairwise_registration_state(state_path, state)
    logger.info("Saved pairwise registration state to %s", state_path)
    try:
        written = write_bad_slices_to_chamber_ops(
            state_path=state_path, warn_missing=False
        )
        if written:
            logger.info("Updated bad_slices in %d chamber ops.yml files", len(written))
    except Exception:
        logger.exception("Failed mirroring bad_slices into chamber ops.yml")
    return Path(state_path)


def _canvas_from_global_matrices(entries, global_from_padded):
    corners = []
    matrices = []
    for entry, matrix in zip(entries, global_from_padded):
        height, width = entry["overview_shape_yx"]
        pad_left = entry["pad_left"]
        pad_top = entry["pad_top"]
        pad = _translation_matrix(pad_left, pad_top)
        total = matrix @ pad
        matrices.append(total)
        local_corners = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=float,
        )
        corners.append(apply_affine_to_points(local_corners, total))
    corners = np.concatenate(corners, axis=0)
    min_xy = np.floor(corners.min(axis=0))
    max_xy = np.ceil(corners.max(axis=0))
    offset = _translation_matrix(-min_xy[0], -min_xy[1])
    canvas_shape = (int(max_xy[1] - min_xy[1]), int(max_xy[0] - min_xy[0]))
    return offset, canvas_shape, matrices


def _reference_slice_from_entries(entries, strategy="largest_area"):
    if strategy == "middle":
        return int(entries[len(entries) // 2]["slice_number"])
    if strategy == "first":
        return int(entries[0]["slice_number"])
    if strategy == "largest_area":
        best = max(
            entries,
            key=lambda entry: entry["overview_shape_yx"][0] * entry["overview_shape_yx"][1],
        )
        return int(best["slice_number"])
    raise ValueError(f"Unknown reference slice strategy: {strategy}")


def compose_global_slice_transforms(
    stack_path,
    state_path=None,
    reference_slice=None,
    reference_strategy="largest_area",
    output_name="global_slice_transforms.npz",
):
    """Concatenate adjacent pairwise transforms into a global slice transform table."""
    stack = load_unregistered_volume_stack(stack_path)
    entries = stack["manifest"]["entries"]
    if state_path is None:
        state_path = Path(stack_path).with_name("pairwise_registration_state.json")
    state = load_pairwise_registration_state(state_path)
    bad_slices = {int(s) for s in state.get("bad_slices", [])}
    active_entries = [
        entry for entry in entries if int(entry["slice_number"]) not in bad_slices
    ]
    active_entries = sorted(active_entries, key=lambda entry: entry["slice_number"])
    if not active_entries:
        raise RuntimeError("No active slices left after excluding bad slices")

    if reference_slice is None:
        reference_slice = _reference_slice_from_entries(
            active_entries,
            strategy=reference_strategy,
        )
    reference_slice = int(reference_slice)
    pair_map = {
        key: value
        for key, value in state.get("pairs", {}).items()
        if value.get("status", "accepted") == "accepted"
    }
    index_by_slice = {
        int(entry["slice_number"]): i for i, entry in enumerate(active_entries)
    }
    if reference_slice not in index_by_slice:
        raise ValueError(f"Reference slice {reference_slice} is not available")

    global_from_padded = {reference_slice: np.eye(3)}
    ordered_slices = [int(entry["slice_number"]) for entry in active_entries]
    ref_index = index_by_slice[reference_slice]

    for i in range(ref_index - 1, -1, -1):
        lower_slice = ordered_slices[i]
        higher_slice = ordered_slices[i + 1]
        key = _pair_key(higher_slice, lower_slice)
        if key not in pair_map:
            raise KeyError(
                "Missing pairwise transform for moving slice "
                f"{lower_slice} to fixed slice {higher_slice}"
            )
        matrix = np.asarray(pair_map[key]["matrix"], dtype=float)
        global_from_padded[lower_slice] = global_from_padded[higher_slice] @ matrix

    for i in range(ref_index + 1, len(ordered_slices)):
        higher_slice = ordered_slices[i]
        lower_slice = ordered_slices[i - 1]
        key = _pair_key(higher_slice, lower_slice)
        if key not in pair_map:
            raise KeyError(
                "Missing pairwise transform for moving slice "
                f"{lower_slice} to fixed slice {higher_slice}"
            )
        matrix = np.asarray(pair_map[key]["matrix"], dtype=float)
        global_from_padded[higher_slice] = (
            global_from_padded[lower_slice] @ np.linalg.inv(matrix)
        )

    matrices = np.stack(
        [global_from_padded[int(entry["slice_number"])] for entry in active_entries],
        axis=0,
    )
    offset, canvas_shape, matrices_with_pad = _canvas_from_global_matrices(
        active_entries,
        matrices,
    )
    global_from_overview = np.stack(
        [offset @ matrix for matrix in matrices_with_pad],
        axis=0,
    )
    global_from_fullres = []
    for entry, matrix in zip(active_entries, matrices):
        downsample = 1.0 / float(entry.get("downsample_ratio", 1.0))
        scale = np.array(
            [[downsample, 0.0, 0.0], [0.0, downsample, 0.0], [0.0, 0.0, 1.0]]
        )
        pad = _translation_matrix(entry["pad_left"], entry["pad_top"])
        global_from_fullres.append(offset @ matrix @ pad @ scale)
    global_from_fullres = np.stack(global_from_fullres, axis=0)

    output_path = Path(stack_path).with_name(output_name)
    np.savez_compressed(
        output_path,
        version=GLOBAL_TRANSFORM_VERSION,
        reference_slice=reference_slice,
        slice_numbers=np.asarray(ordered_slices, dtype=int),
        rois=np.asarray([entry["roi"] for entry in active_entries], dtype=int),
        data_paths=np.asarray([entry["data_path"] for entry in active_entries]),
        overview_files=np.asarray([entry["overview_file"] for entry in active_entries]),
        canvas_shape_yx=np.asarray(canvas_shape, dtype=int),
        canvas_offset_matrix=offset,
        global_from_padded=matrices,
        global_from_overview=global_from_overview,
        global_from_fullres=global_from_fullres,
        bad_slices=np.asarray(sorted(bad_slices), dtype=int),
        auto_bad_slices=np.asarray(
            sorted(int(s) for s in state.get("auto_bad_slices", [])), dtype=int
        ),
    )
    return output_path


def load_global_slice_transforms(transforms_path):
    data = np.load(transforms_path, allow_pickle=False)
    out = {
        "reference_slice": int(data["reference_slice"]),
        "slice_numbers": data["slice_numbers"].astype(int),
        "rois": data["rois"].astype(int),
        "data_paths": data["data_paths"].astype(str),
        "overview_files": data["overview_files"].astype(str),
        "canvas_shape_yx": tuple(int(v) for v in data["canvas_shape_yx"]),
        "canvas_offset_matrix": data["canvas_offset_matrix"].astype(float),
        "global_from_padded": data["global_from_padded"].astype(float),
        "global_from_overview": data["global_from_overview"].astype(float),
        "global_from_fullres": data["global_from_fullres"].astype(float),
    }
    # Added in later versions; fall back to empty arrays for older files.
    out["bad_slices"] = (
        data["bad_slices"].astype(int)
        if "bad_slices" in data.files
        else np.asarray([], dtype=int)
    )
    out["auto_bad_slices"] = (
        data["auto_bad_slices"].astype(int)
        if "auto_bad_slices" in data.files
        else np.asarray([], dtype=int)
    )
    return out


def _group_bad_slices_by_chamber(entries, bad_slices):
    """Return {chamber_data_path: sorted_bad_slice_numbers} from a manifest + bad set."""
    bad = {int(s) for s in bad_slices}
    per_chamber = {}
    for entry in entries:
        chamber = entry["data_path"]
        slice_number = int(entry["slice_number"])
        per_chamber.setdefault(chamber, set())
        if slice_number in bad:
            per_chamber[chamber].add(slice_number)
    return {chamber: sorted(slices) for chamber, slices in per_chamber.items()}


def write_bad_slices_to_chamber_ops(
    stack_path=None,
    state_path=None,
    data_path=None,
    warn_missing=True,
):
    """Mirror bad_slices per-chamber into each chamber's `ops.yml`.

    Downstream tooling (e.g. iss-qc-sindbis) can then read bad_slices via the
    standard `load_ops(chamber_path)["bad_slices"]` path without needing to
    know about the tangential_volume state file. The state JSON remains the
    authoritative source; this writer keeps ops.yml in sync.

    Only updates chambers that already have an `ops.yml` - does not create
    new ops files.
    """
    if state_path is None:
        if stack_path is None:
            if data_path is None:
                raise ValueError("Provide one of data_path, stack_path or state_path")
            stack_path = get_volume_root(data_path) / "unregistered_slices.npz"
        state_path = _default_state_path(stack_path)
    state = load_pairwise_registration_state(state_path)
    bad_slices = state.get("bad_slices", [])
    entries = state.get("entries")
    if entries is None:
        if stack_path is None:
            stack_path = state.get("stack_path")
        if stack_path is None:
            raise RuntimeError(
                "State has no entries and no stack_path to recover them from."
            )
        stack = load_unregistered_volume_stack(stack_path)
        entries = stack["manifest"]["entries"]
    grouping = _group_bad_slices_by_chamber(entries, bad_slices)

    written = []
    for chamber, chamber_bad in grouping.items():
        processed_path = get_processed_path(chamber)
        ops_fname = processed_path / "ops.yml"
        if not ops_fname.exists():
            if warn_missing:
                print(f"Skipping bad_slices write: {ops_fname} does not exist")
            continue
        with open(ops_fname, "r") as fhandle:
            existing = yaml.safe_load(fhandle) or {}
        if existing.get("bad_slices") == chamber_bad:
            continue
        existing["bad_slices"] = chamber_bad
        with open(ops_fname, "w") as fhandle:
            yaml.safe_dump(existing, fhandle, sort_keys=True)
        written.append(str(ops_fname))
    return written


def load_bad_slices(data_path=None, stack_path=None, state_path=None):
    """Return the list of bad (excluded) slice numbers for a volume registration run.

    Resolution order: explicit `state_path` -> explicit `stack_path` -> derive
    from `data_path` via `get_volume_root`. Returns a dict with `bad_slices`
    (full exclusion list, manual + auto) and `auto_bad_slices` (empty-mask
    detections), each as a sorted list of ints.

    Downstream tooling (e.g. iss-qc-sindbis) should prefer this helper over
    reading `pairwise_registration_state.json` directly so the path resolution
    stays in one place.
    """
    if state_path is None:
        if stack_path is None:
            if data_path is None:
                raise ValueError(
                    "Provide one of data_path, stack_path or state_path"
                )
            stack_path = get_volume_root(data_path) / "unregistered_slices.npz"
        state_path = _default_state_path(stack_path)
    state = load_pairwise_registration_state(state_path)
    return {
        "bad_slices": sorted(int(s) for s in state.get("bad_slices", [])),
        "auto_bad_slices": sorted(int(s) for s in state.get("auto_bad_slices", [])),
        "state_path": str(state_path),
    }


def build_registered_volume_stack(
    data_path,
    transforms_path,
    prefix,
    output_name=None,
    suffix="max",
    channels=None,
    correct_illumination=True,
    z_step_um=None,
):
    """Warp stitched ROIs into the global tangential volume frame."""
    transforms = load_global_slice_transforms(transforms_path)
    slice_numbers = transforms["slice_numbers"]
    matrices = transforms["global_from_fullres"]
    data_paths = transforms["data_paths"]
    rois = transforms["rois"]
    canvas_shape = transforms["canvas_shape_yx"]

    if output_name is None:
        output_name = f"registered_volume_{prefix}.npz"
    volume_root = get_volume_root(data_path)
    volume_root.mkdir(parents=True, exist_ok=True)
    logger = _get_volume_job_logger(volume_root)
    output_path = volume_root / output_name

    if z_step_um is None:
        z_step_um = float(
            min(find_roi_position_on_cryostat(path)[1] for path in set(data_paths.tolist()))
        )

    roi_pos_um = {}
    for chamber_path in set(data_paths.tolist()):
        positions, _ = find_roi_position_on_cryostat(chamber_path)
        roi_pos_um[chamber_path] = {int(k): float(v) for k, v in positions.items()}
    z_positions = np.asarray(
        [roi_pos_um[path][int(roi)] for path, roi in zip(data_paths, rois)],
        dtype=float,
    )
    z_indices = np.round((z_positions - z_positions.min()) / z_step_um).astype(int)

    sample = stitch_tiles(
        data_paths[0],
        prefix=prefix,
        roi=int(rois[0]),
        suffix=suffix,
        ich=channels,
        correct_illumination=correct_illumination,
        shifts_prefix=None,
        register_channels=True,
        allow_quick_estimate=False,
        filter_r=False,
    )
    if sample.ndim == 2:
        sample = sample[:, :, np.newaxis]
    nch = sample.shape[2]
    volume = np.zeros(
        (z_indices.max() + 1, canvas_shape[0], canvas_shape[1], nch),
        dtype=np.float32,
    )
    logger.info(
        "Building registered volume %s from %d slices for prefix %s",
        output_path,
        len(slice_numbers),
        prefix,
    )

    with tqdm(
        total=len(slice_numbers) + 1,
        desc=f"Building {prefix} volume",
        dynamic_ncols=True,
    ) as pbar:
        for data_path_i, roi, z_index, matrix in zip(data_paths, rois, z_indices, matrices):
            try:
                stitched = stitch_tiles(
                    data_path_i,
                    prefix=prefix,
                    roi=int(roi),
                    suffix=suffix,
                    ich=channels,
                    correct_illumination=correct_illumination,
                    shifts_prefix=None,
                    register_channels=True,
                    allow_quick_estimate=False,
                    filter_r=False,
                )
                if stitched.ndim == 2:
                    stitched = stitched[:, :, np.newaxis]
                for ch in range(stitched.shape[2]):
                    volume[z_index, :, :, ch] = warp_with_affine(
                        stitched[:, :, ch].astype(np.float32),
                        matrix,
                        output_shape=canvas_shape,
                        order=1,
                    )
                logger.info(
                    "Added roi %s from %s at z-index %s",
                    int(roi),
                    data_path_i,
                    int(z_index),
                )
                pbar.update(1)
            except Exception:
                logger.exception(
                    "Failed building roi %s from %s at z-index %s",
                    int(roi),
                    data_path_i,
                    int(z_index),
                )
                raise

        pbar.set_description(f"Compressing {prefix} volume")
        logger.info("Compressing and saving registered volume to %s", output_path)
        np.savez_compressed(
            output_path,
            volume=volume,
            slice_numbers=slice_numbers,
            z_positions_um=z_positions,
            z_indices=z_indices,
            z_step_um=float(z_step_um),
            canvas_shape_yx=np.asarray(canvas_shape, dtype=int),
            channels=np.asarray(np.arange(nch), dtype=int),
        )
        pbar.update(1)
    logger.info("Saved registered volume to %s", output_path)
    return output_path


def register_spots_to_global_volume(
    data_path,
    transforms_path,
    spots_prefix="barcode_round",
    output_name=None,
    output_unit="pixel",
):
    """Project per-ROI spot tables into the global tangential volume frame."""
    if output_unit not in {"pixel", "um"}:
        raise ValueError("`output_unit` must be either `pixel` or `um`")
    transforms = load_global_slice_transforms(transforms_path)
    matrices = transforms["global_from_fullres"]
    data_paths = transforms["data_paths"]
    rois = transforms["rois"]
    slice_numbers = transforms["slice_numbers"]
    if output_name is None:
        output_name = f"{spots_prefix}_spots_global.pkl"
    logger = _get_volume_job_logger(get_volume_root(data_path))

    pixel_size_um = get_pixel_size(data_paths[0])
    slice_positions_um = {}
    for chamber_path in set(data_paths.tolist()):
        roi_pos_um, _ = find_roi_position_on_cryostat(chamber_path)
        slice_positions_um[chamber_path] = {
            int(k): float(v) for k, v in roi_pos_um.items()
        }

    all_spots = []
    logger.info(
        "Registering spot tables for %d slices from %s",
        len(slice_numbers),
        transforms_path,
    )
    for matrix, chamber_path, roi, slice_number in tqdm(
        zip(matrices, data_paths, rois, slice_numbers),
        total=len(slice_numbers),
        desc="Registering spot tables",
        dynamic_ncols=True,
    ):
        try:
            processed_path = get_processed_path(chamber_path)
            spot_file = processed_path / f"{spots_prefix}_spots_{int(roi)}.pkl"
            if not spot_file.exists():
                logger.info("Skipping missing spot file %s", spot_file)
                continue
            spots = pd.read_pickle(spot_file).copy()
            if not len(spots):
                logger.info("Skipping empty spot file %s", spot_file)
                continue
            global_xy = apply_affine_to_points(spots[["x", "y"]].to_numpy(), matrix)
            spots["x_global"] = global_xy[:, 0]
            spots["y_global"] = global_xy[:, 1]
            if output_unit == "um":
                spots["x_global"] *= pixel_size_um
                spots["y_global"] *= pixel_size_um
            spots["slice_number"] = int(slice_number)
            spots["roi"] = int(roi)
            spots["data_path"] = chamber_path
            spots["z_global_um"] = slice_positions_um[chamber_path][int(roi)]
            all_spots.append(spots)
            logger.info("Registered %d spots from %s", len(spots), spot_file)
        except Exception:
            logger.exception(
                "Failed registering spots for roi %s from %s",
                int(roi),
                chamber_path,
            )
            raise

    if not all_spots:
        raise FileNotFoundError(
            f"Could not find any ROI-level spot files for prefix `{spots_prefix}`."
        )
    output_path = get_volume_root(data_path) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=2, desc="Saving global spot table", dynamic_ncols=True) as pbar:
        merged = pd.concat(all_spots, ignore_index=True)
        pbar.update(1)
        logger.info("Saving merged global spot table to %s", output_path)
        merged.to_pickle(output_path)
        pbar.update(1)
    logger.info("Saved merged global spot table to %s", output_path)
    return output_path
