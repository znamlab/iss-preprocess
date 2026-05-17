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
from skimage.transform import AffineTransform, resize, warp
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

    # Use ruamel.yaml round-trip mode so we preserve the existing file's
    # key ordering, comments, quoting style, and block/flow layout instead
    # of nuking everything with PyYAML's safe_dump. The bad_slices entry
    # is updated in place; new files (none exist by default) would have
    # the key appended at the end.
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedSeq

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True

    written = []
    for chamber, chamber_bad in grouping.items():
        processed_path = get_processed_path(chamber)
        ops_fname = processed_path / "ops.yml"
        if not ops_fname.exists():
            if warn_missing:
                print(f"Skipping bad_slices write: {ops_fname} does not exist")
            continue
        with open(ops_fname, "r") as fhandle:
            existing = yaml_rt.load(fhandle) or {}
        # CommentedSeq inherits from list; list-equality compares element-wise.
        if list(existing.get("bad_slices", [])) == list(chamber_bad):
            continue
        # Force flow style ([1, 2, 3]) regardless of length, matching the
        # short-list convention typical in hand-written ops.yml files.
        bad_seq = CommentedSeq(chamber_bad)
        bad_seq.fa.set_flow_style()
        existing["bad_slices"] = bad_seq
        with open(ops_fname, "w") as fhandle:
            yaml_rt.dump(existing, fhandle)
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


def _scale_matrix(scale):
    return np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])


def _infer_overview_downsample_ratio(transforms):
    ratios = []
    for overview_matrix, fullres_matrix in zip(
        transforms["global_from_overview"], transforms["global_from_fullres"]
    ):
        overview_norm = np.linalg.norm(
            np.asarray(overview_matrix, dtype=float)[:2, :2]
        )
        fullres_norm = np.linalg.norm(
            np.asarray(fullres_matrix, dtype=float)[:2, :2]
        )
        if (
            fullres_norm > 0
            and np.isfinite(overview_norm)
            and np.isfinite(fullres_norm)
        ):
            ratios.append(overview_norm / fullres_norm)
    if ratios:
        return float(np.median(ratios))
    return 1.0


def _overview_pixel_size_um(transforms, native_xy_um):
    for overview_file in transforms.get("overview_files", []):
        metadata_path = Path(str(overview_file)).with_suffix(".yml")
        try:
            metadata = _load_overview_metadata(metadata_path)
        except Exception:
            metadata = {}
        if not metadata:
            continue
        if metadata.get("pixel_size") is not None:
            return float(metadata["pixel_size"])
        if metadata.get("downsample_ratio") is not None:
            original_pixel_size = float(
                metadata.get("original_pixel_size", native_xy_um)
            )
            return original_pixel_size * float(metadata["downsample_ratio"])

    return float(native_xy_um) * _infer_overview_downsample_ratio(transforms)


def _integer_local_mean_downsample(image, factor):
    out_y = image.shape[0] // factor
    out_x = image.shape[1] // factor
    if out_y < 1 or out_x < 1:
        raise ValueError(
            f"Cannot downsample image with shape {image.shape[:2]} by factor {factor}"
        )
    cropped = image[: out_y * factor, : out_x * factor]
    return cropped.reshape(out_y, factor, out_x, factor).mean(
        axis=(1, 3),
        dtype=np.float32,
    )


def _area_resample_to_pixel_size(image, input_pixel_size_um, output_pixel_size_um):
    image = np.asarray(image, dtype=np.float32)
    input_pixel_size_um = float(input_pixel_size_um)
    output_pixel_size_um = float(output_pixel_size_um)
    if input_pixel_size_um <= 0 or output_pixel_size_um <= 0:
        raise ValueError("Pixel sizes must be positive")

    scale = input_pixel_size_um / output_pixel_size_um
    output_shape = (
        max(1, int(np.floor(image.shape[0] * scale))),
        max(1, int(np.floor(image.shape[1] * scale))),
    )
    if output_shape == image.shape[:2]:
        return image

    downsample_factor = output_pixel_size_um / input_pixel_size_um
    nearest_integer = int(round(downsample_factor))
    if downsample_factor > 1 and np.isclose(
        downsample_factor, nearest_integer, rtol=1e-6, atol=1e-6
    ):
        downsampled = _integer_local_mean_downsample(image, nearest_integer)
        if downsampled.shape == output_shape:
            return downsampled.astype(np.float32, copy=False)

    return resize(
        image,
        output_shape,
        order=1,
        mode="edge",
        cval=0,
        clip=False,
        preserve_range=True,
        anti_aliasing=scale < 1.0,
    ).astype(np.float32, copy=False)


def _prepare_volume_resampling(transforms, native_xy_um, target_voxel_size_um=None):
    overview_xy_um = _overview_pixel_size_um(transforms, native_xy_um)
    xy_voxel_size_um = (
        overview_xy_um
        if target_voxel_size_um is None
        else float(target_voxel_size_um)
    )
    if xy_voxel_size_um <= 0:
        raise ValueError("target_voxel_size_um must be positive")

    canvas_shape = transforms["canvas_shape_yx"]
    output_from_overview_scale = overview_xy_um / xy_voxel_size_um
    output_shape = (
        max(1, int(round(int(canvas_shape[0]) * output_from_overview_scale))),
        max(1, int(round(int(canvas_shape[1]) * output_from_overview_scale))),
    )
    output_from_overview = _scale_matrix(output_from_overview_scale)
    overview_from_resampled_source = _scale_matrix(
        xy_voxel_size_um / overview_xy_um
    )
    matrices = np.stack(
        [
            output_from_overview
            @ np.asarray(matrix, dtype=float)
            @ overview_from_resampled_source
            for matrix in transforms["global_from_overview"]
        ]
    )
    return matrices, output_shape, float(xy_voxel_size_um), float(overview_xy_um)


def _build_volume_slice_worker(
    data_path,
    roi,
    z_index,
    matrix,
    output_shape,
    native_xy_um,
    resampled_xy_um,
    prefix,
    suffix,
    channels,
    correct_illumination,
):
    """Stitch + warp a single slice/ROI for ``build_registered_volume_stack``.

    Module-level so loky workers can pickle it. Returns
    ``(z_index, [(channel_idx, warped_2d_float32), ...])``.
    """
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        stitched = stitch_tiles(
            data_path,
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
        warped_channels = []
        for ch in range(stitched.shape[2]):
            resampled = _area_resample_to_pixel_size(
                stitched[:, :, ch],
                input_pixel_size_um=native_xy_um,
                output_pixel_size_um=resampled_xy_um,
            )
            warped = warp_with_affine(
                resampled,
                matrix,
                output_shape=output_shape,
                order=1,
            )
            warped_channels.append((ch, warped.astype(np.float32, copy=False)))
        return int(z_index), warped_channels


def build_registered_volume_stack(
    data_path,
    transforms_path,
    prefix,
    output_name=None,
    suffix="max",
    channels=None,
    correct_illumination=True,
    z_step_um=None,
    target_voxel_size_um=None,
    n_jobs=1,
):
    """Warp stitched ROIs into the global tangential volume frame.

    Args:
        data_path: One chamber path or the parent mouse path; used to
            resolve the volume root for outputs.
        transforms_path: NPZ produced by ``compose_global_slice_transforms``.
        prefix: Acquisition prefix to stitch and warp.
        output_name: Output filename inside the volume root. Default
            includes the target voxel size when one is provided.
        suffix, channels, correct_illumination: Forwarded to ``stitch_tiles``.
        z_step_um (float, optional): Override the inferred Z spacing.
        target_voxel_size_um (float, optional): If set, the volume is
            rendered at this isotropic XY voxel size. Each stitched channel
            is first downsampled with local-mean/anti-aliased resampling,
            then warped at the smaller canvas size. If ``None``, the output
            keeps the overview pixel size used for slice registration.
        n_jobs (int): Parallel worker count. ``1`` (default) runs
            sequentially; ``>1`` dispatches per-slice work via joblib
            ``loky`` workers with BLAS oversubscription guarded by
            ``threadpool_limits(limits=1)``. Throughput is typically
            disk-bound, so values much larger than the number of
            independent disk readers see diminishing returns.
    """
    transforms = load_global_slice_transforms(transforms_path)
    slice_numbers = transforms["slice_numbers"]
    data_paths = transforms["data_paths"]
    rois = transforms["rois"]

    if output_name is None:
        suffix_token = (
            f"_iso{int(round(target_voxel_size_um))}um"
            if target_voxel_size_um is not None
            else ""
        )
        output_name = f"registered_volume_{prefix}{suffix_token}.npz"
    volume_root = get_volume_root(data_path)
    volume_root.mkdir(parents=True, exist_ok=True)
    logger = _get_volume_job_logger(volume_root)
    output_path = volume_root / output_name

    native_xy_um = float(get_pixel_size(data_paths[0], prefix=prefix))
    matrices, output_shape, xy_voxel_size_um, overview_xy_um = (
        _prepare_volume_resampling(
            transforms,
            native_xy_um=native_xy_um,
            target_voxel_size_um=target_voxel_size_um,
        )
    )

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
        (z_indices.max() + 1, output_shape[0], output_shape[1], nch),
        dtype=np.float32,
    )
    logger.info(
        "Building registered volume %s from %d slices for prefix %s "
        "(canvas %s, %d ch, n_jobs=%d, native_xy_um=%.4f, "
        "overview_xy_um=%.4f, output_xy_um=%.4f)",
        output_path,
        len(slice_numbers),
        prefix,
        output_shape,
        nch,
        int(n_jobs),
        native_xy_um,
        overview_xy_um,
        xy_voxel_size_um,
    )

    n_slices = len(slice_numbers)

    with tqdm(
        total=n_slices + 1,
        desc=f"Building {prefix} volume"
        + (f" (n_jobs={int(n_jobs)})" if int(n_jobs) > 1 else ""),
        dynamic_ncols=True,
    ) as pbar:
        if int(n_jobs) > 1:
            from joblib import Parallel, delayed

            tasks = [
                (str(dp), int(roi), int(z_idx), np.asarray(mat, dtype=float))
                for dp, roi, z_idx, mat in zip(
                    data_paths, rois, z_indices, matrices
                )
            ]
            try:
                results_iter = Parallel(
                    n_jobs=int(n_jobs),
                    backend="loky",
                    return_as="generator",
                )(
                    delayed(_build_volume_slice_worker)(
                        dp,
                        roi,
                        z_idx,
                        mat,
                        output_shape,
                        native_xy_um,
                        xy_voxel_size_um,
                        prefix,
                        suffix,
                        channels,
                        correct_illumination,
                    )
                    for dp, roi, z_idx, mat in tasks
                )
            except TypeError:
                # joblib < 1.3 has no return_as="generator"
                results_iter = iter(
                    Parallel(n_jobs=int(n_jobs), backend="loky")(
                        delayed(_build_volume_slice_worker)(
                            dp,
                            roi,
                            z_idx,
                            mat,
                            output_shape,
                            native_xy_um,
                            xy_voxel_size_um,
                            prefix,
                            suffix,
                            channels,
                            correct_illumination,
                        )
                        for dp, roi, z_idx, mat in tasks
                    )
                )

            for z_idx, warped_channels in results_iter:
                for ch, warped in warped_channels:
                    volume[z_idx, :, :, ch] = warped
                logger.info("Added slice at z-index %s", int(z_idx))
                pbar.update(1)
        else:
            for data_path_i, roi, z_index, matrix in zip(
                data_paths, rois, z_indices, matrices
            ):
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
                        resampled = _area_resample_to_pixel_size(
                            stitched[:, :, ch],
                            input_pixel_size_um=native_xy_um,
                            output_pixel_size_um=xy_voxel_size_um,
                        )
                        volume[z_index, :, :, ch] = warp_with_affine(
                            resampled,
                            matrix,
                            output_shape=output_shape,
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
            xy_voxel_size_um=float(xy_voxel_size_um),
            native_xy_voxel_size_um=float(native_xy_um),
            overview_xy_voxel_size_um=float(overview_xy_um),
            canvas_shape_yx=np.asarray(output_shape, dtype=int),
            channels=np.asarray(np.arange(nch), dtype=int),
        )
        pbar.update(1)
    logger.info("Saved registered volume to %s", output_path)
    return output_path


def register_spots_to_global_volume(
    data_path,
    transforms_path,
    spots_prefix="barcode_round",
    pixel_size_reference_round="barcode_round_1_1",
    output_name=None,
    output_unit="pixel",
    include_bad_slices=True,
):
    """Project per-ROI spot tables into the global tangential volume frame.

    Iterates over ``transforms["slice_numbers"]`` (registered slices only)
    and applies the per-slice ``global_from_fullres`` affine to each
    chamber's ``{spots_prefix}_spots_{roi}.pkl`` ``(x, y)`` columns. The
    union of all registered slices is concatenated and saved as one mouse-
    level table at ``{volume_root}/{spots_prefix}_spots_global.pkl``.

    When ``include_bad_slices=True`` (default), bad-slice ROIs are also
    loaded from the same per-ROI files and appended with NaN
    ``x_global`` / ``y_global`` / ``z_global_um`` and
    ``is_bad_slice=True``. This keeps the output table a single source of
    truth: spatial QC steps ``.dropna()`` the NaN rows out, but sequence-
    only steps (error correction, Hamming distance) still see them.

    Args:
        data_path: any chamber path; only used to resolve
            ``get_volume_root(data_path)`` for the output location.
        transforms_path: NPZ produced by
            :func:`compose_global_slice_transforms`.
        spots_prefix: prefix for the per-ROI spot files
            (``{spots_prefix}_spots_{roi}.pkl``) and the output filename.
        pixel_size_reference_round: acquisition prefix used to look up the
            camera pixel size when ``output_unit='um'``.
        output_name: override the output filename; defaults to
            ``{spots_prefix}_spots_global.pkl`` inside the volume root.
        output_unit: ``"pixel"`` (default) or ``"um"``. Applied to the
            ``x_global`` / ``y_global`` columns only; ``z_global_um`` is
            always in microns (from ``find_roi_position_on_cryostat``).
        include_bad_slices: when True (default), append bad-slice ROI rows
            with NaN globals + ``is_bad_slice=True``.

    Returns:
        pathlib.Path: location of the saved pickle.
    """
    if output_unit not in {"pixel", "um"}:
        raise ValueError("`output_unit` must be either `pixel` or `um`")
    transforms = load_global_slice_transforms(transforms_path)
    matrices = transforms["global_from_fullres"]
    data_paths = transforms["data_paths"]
    rois = transforms["rois"]
    slice_numbers = transforms["slice_numbers"]
    bad_slice_numbers = {int(s) for s in transforms.get("bad_slices", [])}
    if output_name is None:
        output_name = f"{spots_prefix}_spots_global.pkl"
    logger = _get_volume_job_logger(get_volume_root(data_path))

    # The global_from_fullres affine lands in the global canvas, which is
    # at overview-pixel resolution (not native camera pixels). To go to
    # μm we therefore scale by the overview pixel size, not the native
    # camera pixel size — multiplying by the native size silently
    # underestimates μm by `downsample_ratio` (e.g. 8× or 32×).
    native_xy_um = float(
        get_pixel_size(data_paths[0], prefix=pixel_size_reference_round)
    )
    overview_xy_um = float(_overview_pixel_size_um(transforms, native_xy_um))
    slice_positions_um = {}
    z_step_per_chamber = {}
    for chamber_path in set(data_paths.tolist()):
        roi_pos_um, z_step = find_roi_position_on_cryostat(chamber_path)
        slice_positions_um[chamber_path] = {
            int(k): float(v) for k, v in roi_pos_um.items()
        }
        z_step_per_chamber[chamber_path] = float(z_step)

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
                spots["x_global"] *= overview_xy_um
                spots["y_global"] *= overview_xy_um
            spots["slice_number"] = int(slice_number)
            spots["roi"] = int(roi)
            spots["data_path"] = chamber_path
            spots["z_global_um"] = slice_positions_um[chamber_path][int(roi)]
            spots["is_bad_slice"] = False
            all_spots.append(spots)
            logger.info("Registered %d spots from %s", len(spots), spot_file)
        except Exception:
            logger.exception(
                "Failed registering spots for roi %s from %s",
                int(roi),
                chamber_path,
            )
            raise

    if include_bad_slices and bad_slice_numbers:
        for chamber_path in sorted(set(data_paths.tolist())):
            z_step = z_step_per_chamber[chamber_path]
            for roi, pos_um in slice_positions_um[chamber_path].items():
                slice_num = int(round(pos_um / z_step))
                if slice_num not in bad_slice_numbers:
                    continue
                processed_path = get_processed_path(chamber_path)
                spot_file = processed_path / f"{spots_prefix}_spots_{int(roi)}.pkl"
                if not spot_file.exists():
                    continue
                spots = pd.read_pickle(spot_file).copy()
                if not len(spots):
                    continue
                spots["x_global"] = np.nan
                spots["y_global"] = np.nan
                spots["z_global_um"] = np.nan
                spots["slice_number"] = slice_num
                spots["roi"] = int(roi)
                spots["data_path"] = chamber_path
                spots["is_bad_slice"] = True
                all_spots.append(spots)
                logger.info(
                    "Loaded %d spots from bad-slice %s roi %d (no global coords)",
                    len(spots),
                    spot_file,
                    int(roi),
                )

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


def register_somata_to_global_volume(
    data_path,
    transforms_path,
    barcode_prefix="barcode_round",
    pixel_size_reference_round="barcode_round_1_1",
    output_name=None,
    output_unit="pixel",
    include_bad_slices=True,
):
    """Project per-chamber stitched soma-call tables into the global tangential
    volume frame.

    Mirrors :func:`register_spots_to_global_volume`. For each registered
    slice listed in the transforms file, loads the corresponding chamber's
    stitched soma table via
    :func:`iss_preprocess.pipeline.somata.load_stitched_soma_calls`
    (``filtered=False`` — soma QC belongs in iss-qc-sindbis), slices by
    ROI, applies the matching ``global_from_fullres`` affine to the
    ``(x, y)`` columns, and adds ``x_global`` / ``y_global`` /
    ``z_global_um`` / ``slice_number`` / ``data_path`` columns. The union
    across slices and chambers is saved as a single mouse-level table at
    ``{volume_root}/{barcode_prefix}_somata_global.pkl``.

    When ``include_bad_slices=True`` (default), somata from bad-slice ROIs
    are also appended with NaN ``x_global`` / ``y_global`` /
    ``z_global_um`` and ``is_bad_slice=True``. They still carry every
    non-coordinate column (``label``, ``sequence``, ``bases``, ``area``,
    QC scores, …) so sequence-only QC steps in iss-qc-sindbis (error
    correction, Hamming distance) can use them.

    Args:
        data_path: any chamber path; only used to resolve
            ``get_volume_root(data_path)`` for the output location.
        transforms_path: NPZ produced by
            :func:`compose_global_slice_transforms`.
        barcode_prefix: prefix used by
            :func:`iss_preprocess.pipeline.somata.load_stitched_soma_calls`
            and for the output filename.
        pixel_size_reference_round: acquisition prefix used to look up
            the camera pixel size when ``output_unit='um'``.
        output_name: override the output filename; defaults to
            ``{barcode_prefix}_somata_global.pkl`` inside the volume root.
        output_unit: ``"pixel"`` (default) or ``"um"``. Applied to the
            ``x_global`` / ``y_global`` columns only; ``z_global_um`` is
            always in microns.
        include_bad_slices: when True (default), append bad-slice ROI
            rows with NaN globals + ``is_bad_slice=True``.

    Returns:
        pathlib.Path: location of the saved pickle.
    """
    from .somata import load_stitched_soma_calls

    if output_unit not in {"pixel", "um"}:
        raise ValueError("`output_unit` must be either `pixel` or `um`")
    transforms = load_global_slice_transforms(transforms_path)
    matrices = transforms["global_from_fullres"]
    data_paths = transforms["data_paths"]
    rois = transforms["rois"]
    slice_numbers = transforms["slice_numbers"]
    bad_slice_numbers = {int(s) for s in transforms.get("bad_slices", [])}
    if output_name is None:
        output_name = f"{barcode_prefix}_somata_global.pkl"
    logger = _get_volume_job_logger(get_volume_root(data_path))

    # See note in `register_spots_to_global_volume`: μm conversion uses
    # the overview pixel size (canvas resolution), not the native camera
    # pixel size.
    native_xy_um = float(
        get_pixel_size(data_paths[0], prefix=pixel_size_reference_round)
    )
    overview_xy_um = float(_overview_pixel_size_um(transforms, native_xy_um))
    slice_positions_um = {}
    z_step_per_chamber = {}
    for chamber_path in set(data_paths.tolist()):
        roi_pos_um, z_step = find_roi_position_on_cryostat(chamber_path)
        slice_positions_um[chamber_path] = {
            int(k): float(v) for k, v in roi_pos_um.items()
        }
        z_step_per_chamber[chamber_path] = float(z_step)

    # Per-chamber load once, slice by ROI to apply per-slice affines.
    chamber_tables = {}

    def _load_chamber(chamber_path):
        if chamber_path not in chamber_tables:
            try:
                chamber_tables[chamber_path] = load_stitched_soma_calls(
                    chamber_path,
                    barcode_prefix=barcode_prefix,
                    filtered=False,
                )
            except FileNotFoundError:
                logger.info(
                    "Skipping chamber %s: stitched soma table not found",
                    chamber_path,
                )
                chamber_tables[chamber_path] = None
        return chamber_tables[chamber_path]

    logger.info(
        "Registering soma tables for %d slices from %s",
        len(slice_numbers),
        transforms_path,
    )

    all_somata = []
    for matrix, chamber_path, roi, slice_number in tqdm(
        zip(matrices, data_paths, rois, slice_numbers),
        total=len(slice_numbers),
        desc="Registering soma tables",
        dynamic_ncols=True,
    ):
        try:
            stitched = _load_chamber(chamber_path)
            if stitched is None or stitched.empty:
                continue
            soma_subset = stitched.loc[stitched["roi"].astype(int) == int(roi)].copy()
            if soma_subset.empty:
                continue
            global_xy = apply_affine_to_points(
                soma_subset[["x", "y"]].to_numpy(), matrix
            )
            soma_subset["x_global"] = global_xy[:, 0]
            soma_subset["y_global"] = global_xy[:, 1]
            if output_unit == "um":
                soma_subset["x_global"] *= overview_xy_um
                soma_subset["y_global"] *= overview_xy_um
            soma_subset["slice_number"] = int(slice_number)
            soma_subset["data_path"] = chamber_path
            soma_subset["z_global_um"] = slice_positions_um[chamber_path][int(roi)]
            soma_subset["is_bad_slice"] = False
            all_somata.append(soma_subset)
            logger.info(
                "Registered %d somata from %s roi %d",
                len(soma_subset),
                chamber_path,
                int(roi),
            )
        except Exception:
            logger.exception(
                "Failed registering somata for roi %s from %s",
                int(roi),
                chamber_path,
            )
            raise

    if include_bad_slices and bad_slice_numbers:
        for chamber_path in sorted(set(data_paths.tolist())):
            stitched = _load_chamber(chamber_path)
            if stitched is None or stitched.empty:
                continue
            z_step = z_step_per_chamber[chamber_path]
            for roi, pos_um in slice_positions_um[chamber_path].items():
                slice_num = int(round(pos_um / z_step))
                if slice_num not in bad_slice_numbers:
                    continue
                soma_subset = stitched.loc[stitched["roi"].astype(int) == int(roi)].copy()
                if soma_subset.empty:
                    continue
                soma_subset["x_global"] = np.nan
                soma_subset["y_global"] = np.nan
                soma_subset["z_global_um"] = np.nan
                soma_subset["slice_number"] = slice_num
                soma_subset["data_path"] = chamber_path
                soma_subset["is_bad_slice"] = True
                all_somata.append(soma_subset)
                logger.info(
                    "Loaded %d somata from bad-slice %s roi %d (no global coords)",
                    len(soma_subset),
                    chamber_path,
                    int(roi),
                )

    if not all_somata:
        raise FileNotFoundError(
            f"Could not find any stitched soma tables for prefix `{barcode_prefix}`."
        )
    output_path = get_volume_root(data_path) / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=2, desc="Saving global soma table", dynamic_ncols=True) as pbar:
        merged = pd.concat(all_somata, ignore_index=True)
        pbar.update(1)
        logger.info("Saving merged global soma table to %s", output_path)
        merged.to_pickle(output_path)
        pbar.update(1)
    logger.info("Saved merged global soma table to %s", output_path)
    return output_path


def register_tissue_masks_to_global_volume(
    data_path,
    transforms_path,
    stack_path=None,
    output_name="tissue_mask_global.npz",
    z_step_um=None,
    pixel_size_reference_round="barcode_round_1_1",
):
    """Warp the user-drawn tissue masks from
    :func:`export_unregistered_volume_stack` into the global volume frame.

    Input: ``user_masks`` plane of
    ``tangential_volume/unregistered_slices.npz``, drawn in Napari (1 =
    keep, 0 = exclude) in the same padded-overview pixel space as the
    unregistered slice images.

    Per active slice, the function applies
    ``canvas_offset_matrix @ global_from_padded[i]`` (nearest-neighbour
    warp) to project the 2D mask into the global canvas. Bad slices are
    excluded automatically — ``compose_global_slice_transforms`` drops
    them from ``transforms["slice_numbers"]``, and
    ``_prepare_pairwise_state`` flags any slice with an all-zero mask as
    a bad slice upstream.

    Output frame matches :func:`build_registered_volume_stack`'s
    native-resolution output: a 3D boolean volume in the canvas defined
    by ``transforms["canvas_shape_yx"]``, with z indexed by
    ``z_index = round((z_position_um - z_min_um) / z_step_um)``.
    ``xy_voxel_size_um`` (overview pixel size, derived via
    ``_overview_pixel_size_um``) is saved as metadata so a μm-coordinate
    consumer can convert query coords to mask indices.

    Args:
        data_path: any chamber path; only used to resolve
            ``get_volume_root(data_path)`` for the output location.
        transforms_path: NPZ produced by
            :func:`compose_global_slice_transforms`.
        stack_path: NPZ produced by
            :func:`export_unregistered_volume_stack`. Defaults to
            ``{volume_root}/unregistered_slices.npz``.
        output_name: output filename inside the volume root.
        z_step_um: override the inferred z spacing. If ``None``, taken
            from the minimum cryostat slice spacing across chambers
            (matches :func:`build_registered_volume_stack`).
        pixel_size_reference_round: acquisition prefix used to look up
            the native (per-tile) pixel size; the overview pixel size is
            then derived via ``_overview_pixel_size_um``.

    Returns:
        pathlib.Path: location of the saved NPZ. Contents:
        ``volume`` (z, y, x) bool, ``slice_numbers``, ``rois``,
        ``data_paths``, ``z_positions_um``, ``z_indices``, ``z_step_um``,
        ``xy_voxel_size_um``, ``canvas_shape_yx``.
    """
    volume_root = get_volume_root(data_path)
    if stack_path is None:
        stack_path = volume_root / "unregistered_slices.npz"
    logger = _get_volume_job_logger(volume_root)

    transforms = load_global_slice_transforms(transforms_path)
    slice_numbers = transforms["slice_numbers"]
    rois = transforms["rois"]
    data_paths = transforms["data_paths"]
    canvas_shape = transforms["canvas_shape_yx"]
    offset = transforms["canvas_offset_matrix"]
    global_from_padded = transforms["global_from_padded"]

    stack = load_unregistered_volume_stack(stack_path)
    user_masks = stack["user_masks"]  # (N_entries, H_pad, W_pad) bool
    manifest_entries = stack["manifest"]["entries"]
    entry_by_slice = {int(entry["slice_number"]): i for i, entry in enumerate(manifest_entries)}

    active_slices_set = {int(s) for s in slice_numbers}
    for entry_slice, i in entry_by_slice.items():
        if entry_slice in active_slices_set:
            continue
        if user_masks[i].any():
            logger.warning(
                "Slice %s has a user mask but is flagged as bad and will be excluded",
                entry_slice,
            )

    native_xy_um = float(
        get_pixel_size(data_paths[0], prefix=pixel_size_reference_round)
    )
    overview_xy_um = float(_overview_pixel_size_um(transforms, native_xy_um))
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
    n_z = int(z_indices.max()) + 1

    volume = np.zeros((n_z, canvas_shape[0], canvas_shape[1]), dtype=bool)
    logger.info(
        "Warping %d user masks into global canvas %s at z-depth %d",
        len(slice_numbers),
        canvas_shape,
        n_z,
    )
    for i, (slice_number, matrix_padded) in enumerate(
        tqdm(
            zip(slice_numbers, global_from_padded),
            total=len(slice_numbers),
            desc="Warping tissue masks",
            dynamic_ncols=True,
        )
    ):
        slice_int = int(slice_number)
        if slice_int not in entry_by_slice:
            logger.warning(
                "Active slice %s missing from manifest; skipping", slice_int
            )
            continue
        mask_in = user_masks[entry_by_slice[slice_int]].astype(np.uint8)
        if not mask_in.any():
            continue
        warped = warp_with_affine(
            mask_in,
            offset @ matrix_padded,
            output_shape=canvas_shape,
            order=0,
            cval=0.0,
        )
        volume[int(z_indices[i])] |= warped.astype(bool)

    output_path = volume_root / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        volume=volume,
        slice_numbers=np.asarray(slice_numbers, dtype=int),
        rois=np.asarray(rois, dtype=int),
        data_paths=np.asarray(data_paths),
        z_positions_um=z_positions,
        z_indices=z_indices,
        z_step_um=float(z_step_um),
        xy_voxel_size_um=float(overview_xy_um),
        canvas_shape_yx=np.asarray(canvas_shape, dtype=int),
    )
    logger.info("Saved global tissue mask to %s", output_path)
    return output_path
