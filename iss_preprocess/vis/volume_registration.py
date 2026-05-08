from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ..pipeline.volume_registration import (
    _orientation_matrix,
    _rotation_matrix,
    _translation_matrix,
    build_adjacent_slice_pairs,
    estimate_pairwise_registration,
    load_pairwise_registration_state,
    load_unregistered_volume_stack,
    save_pairwise_registration_state,
    warp_with_affine,
    write_bad_slices_to_chamber_ops,
)


def open_volume_stack_in_napari(stack_path, viewer=None):
    """Open an unregistered volume stack in napari for mask drawing.

    `user_masks` is added as a Labels layer so 0 is transparent and painted
    pixels (label 1) are visible. Use `save_user_masks_from_napari` to write
    the edited masks back into the NPZ.
    """
    import napari

    stack = load_unregistered_volume_stack(stack_path)
    if viewer is None:
        viewer = napari.Viewer()
    viewer.add_image(stack["images"], name="images", contrast_limits=(0, 1))
    viewer.add_labels(stack["default_masks"].astype(np.uint8), name="default_masks")
    viewer.add_labels(stack["user_masks"].astype(np.uint8), name="user_masks")
    return viewer, stack


def save_user_masks_from_napari(viewer, stack_path, layer_name="user_masks"):
    """Persist the napari `user_masks` Labels layer back into the NPZ stack."""
    import json

    stack = load_unregistered_volume_stack(stack_path)
    user_masks = (np.asarray(viewer.layers[layer_name].data) > 0).astype(np.uint8)
    np.savez_compressed(
        stack_path,
        images=stack["images"].astype(np.float32),
        default_masks=stack["default_masks"].astype(np.uint8),
        user_masks=user_masks,
        manifest_json=json.dumps(stack["manifest"]),
    )
    return stack_path


def _mask_overlay(mask, color, alpha=0.25):
    mask = np.asarray(mask, dtype=bool)
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[..., :3] = color
    rgba[..., 3] = mask.astype(float) * alpha
    return rgba


OVERLAP_FIXED_CMAP = LinearSegmentedColormap.from_list(
    "overlap_fixed", ["#000000", "#00ffd5"]
)
OVERLAP_MOVING_CMAP = LinearSegmentedColormap.from_list(
    "overlap_moving", ["#000000", "#ff4fd8"]
)


def _intensity_rgba(image, cmap, vmin=None, vmax=None, alpha_scale=1.0, mask=None):
    image = np.asarray(image, dtype=float)
    if vmin is None:
        vmin = float(np.nanmin(image))
    if vmax is None:
        vmax = float(np.nanmax(image))
    denom = max(float(vmax) - float(vmin), 1e-6)
    norm = np.clip((image - float(vmin)) / denom, 0.0, 1.0)
    rgba = cmap(norm)
    rgba[..., 3] = np.clip(norm * float(alpha_scale), 0.0, 1.0)
    if mask is not None:
        rgba[..., 3] *= np.asarray(mask, dtype=float)
    return rgba


def _render_overlay(
    fixed_image,
    warped_image,
    title,
    vmin=None,
    vmax=None,
    fixed_mask=None,
    warped_mask=None,
):
    fig = plt.figure(figsize=(10, 18))
    grid = fig.add_gridspec(3, 1)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    ]
    overlap_fixed = fixed_image if fixed_mask is None else fixed_image * fixed_mask
    overlap_warped = warped_image if warped_mask is None else warped_image * warped_mask
    axes[0].imshow(
        _intensity_rgba(
            overlap_fixed,
            OVERLAP_FIXED_CMAP,
            vmin=vmin,
            vmax=vmax,
            alpha_scale=1.35,
            mask=fixed_mask,
        )
    )
    axes[0].imshow(
        _intensity_rgba(
            overlap_warped,
            OVERLAP_MOVING_CMAP,
            vmin=vmin,
            vmax=vmax,
            alpha_scale=1.35,
            mask=warped_mask,
        )
    )
    axes[0].set_title(title)
    axes[1].imshow(fixed_image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Fixed")
    axes[2].imshow(warped_image, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title("Warped moving")
    if fixed_mask is not None:
        axes[0].imshow(_mask_overlay(fixed_mask, color=(0.2, 1.0, 1.0), alpha=0.18))
        axes[1].imshow(_mask_overlay(fixed_mask, color=(0.2, 1.0, 1.0), alpha=0.22))
        axes[1].contour(fixed_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    if warped_mask is not None:
        axes[0].imshow(_mask_overlay(warped_mask, color=(1.0, 0.5, 0.0), alpha=0.18))
        axes[2].imshow(_mask_overlay(warped_mask, color=(1.0, 0.5, 0.0), alpha=0.22))
        axes[2].contour(warped_mask.astype(float), levels=[0.5], colors=["orange"], linewidths=0.8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def _pair_bad_slice_labels(fixed_entry, moving_entry, bad_slices):
    bad = {int(s) for s in (bad_slices or [])}
    labels = []
    if int(fixed_entry["slice_number"]) in bad:
        labels.append(f"fixed bad slice {int(fixed_entry['slice_number']):03d}")
    if int(moving_entry["slice_number"]) in bad:
        labels.append(f"moving bad slice {int(moving_entry['slice_number']):03d}")
    return labels


def _pair_label_html(fixed_entry, moving_entry, bad_slices):
    fixed_text = f"{int(fixed_entry['slice_number']):03d}"
    moving_text = f"{int(moving_entry['slice_number']):03d}"
    if int(fixed_entry["slice_number"]) in {int(s) for s in (bad_slices or [])}:
        fixed_text += " (bad slice)"
    if int(moving_entry["slice_number"]) in {int(s) for s in (bad_slices or [])}:
        moving_text += " (bad slice)"
    return f"<b>Slice pair:</b> fixed {fixed_text} <- moving {moving_text}"


def save_pairwise_overlap_plots(
    stack_path,
    state_path=None,
    output_dir=None,
):
    """Save one PNG overlay per pairwise registration for quick review."""
    stack = load_unregistered_volume_stack(stack_path)
    entries = stack["manifest"]["entries"]
    images = stack["images"]
    index_by_slice = {int(entry["slice_number"]): i for i, entry in enumerate(entries)}
    if state_path is None:
        state_path = Path(stack_path).with_name("pairwise_registration_state.json")
    state = load_pairwise_registration_state(state_path)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(stack_path).parent / "pairwise_overlap_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, record in sorted(state.get("pairs", {}).items()):
        fixed_slice = int(record["fixed_slice"])
        moving_slice = int(record["moving_slice"])
        fixed_image = images[index_by_slice[fixed_slice]]
        moving_image = images[index_by_slice[moving_slice]].astype(np.float32)
        fixed_mask = stack["user_masks"][index_by_slice[fixed_slice]].astype(bool)
        moving_mask = stack["user_masks"][index_by_slice[moving_slice]].astype(bool)
        if record.get("matrix") is not None:
            matrix = np.asarray(record["matrix"], dtype=float)
            warped = warp_with_affine(
                moving_image,
                matrix,
                output_shape=fixed_image.shape,
                order=1,
            )
            warped_mask = (
                warp_with_affine(
                    moving_mask.astype(np.float32),
                    matrix,
                    output_shape=fixed_image.shape,
                    order=0,
                )
                > 0.5
            )
            mode = record.get("method", "saved")
        else:
            warped = moving_image
            warped_mask = moving_mask
            mode = "unregistered"
        bad_labels = _pair_bad_slice_labels(
            entries[index_by_slice[fixed_slice]],
            entries[index_by_slice[moving_slice]],
            state.get("bad_slices", []),
        )
        title = f"{key} | {record.get('status', 'pending')} | {mode}"
        if bad_labels:
            title += " | contains bad slice"
        fig = _render_overlay(
            fixed_image,
            warped,
            title,
            fixed_mask=fixed_mask,
            warped_mask=warped_mask,
        )
        fig.savefig(output_dir / f"{key}_{record.get('status', 'pending')}.png", dpi=150)
        plt.close(fig)
    return output_dir


def review_pairwise_registrations_widget(
    stack_path,
    state_path=None,
    default_radius=150.0,
    default_angle_radius_deg=5.0,
):
    """Create an ipywidgets UI for reviewing pairwise slice registrations."""
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:
        raise ImportError(
            "The review widget requires `ipywidgets` and an IPython notebook environment."
        ) from exc

    stack = load_unregistered_volume_stack(stack_path)
    entries = stack["manifest"]["entries"]
    images = stack["images"]
    masks = stack["user_masks"]
    index_by_slice = {int(entry["slice_number"]): i for i, entry in enumerate(entries)}

    if state_path is None:
        state_path = Path(stack_path).with_name("pairwise_registration_state.json")
    state = load_pairwise_registration_state(state_path)
    state.setdefault("bad_slices", [])
    state.setdefault("pairs", {})

    image_min = float(np.nanmin(images))
    image_max = float(np.nanmax(images))
    if not np.isfinite(image_min):
        image_min = 0.0
    if not np.isfinite(image_max):
        image_max = 1.0
    if image_max <= image_min:
        image_max = image_min + 1.0
    default_vmin = float(np.nanpercentile(images, 1))
    default_vmax = float(np.nanpercentile(images, 99.5))
    default_vmin = min(max(default_vmin, image_min), image_max)
    default_vmax = min(max(default_vmax, default_vmin + 1e-6), image_max)

    pair_label = widgets.HTML()
    status_label = widgets.HTML()
    action_label = widgets.HTML()
    transform_label = widgets.HTML()
    action_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description="",
        bar_style="info",
        layout=widgets.Layout(width="500px", visibility="hidden"),
    )
    pair_index = widgets.IntSlider(description="Pair", min=0, max=0, value=0)
    orientation = widgets.Dropdown(
        description="Orientation",
        options=[
            ("Auto", "auto"),
            ("No flip, 0 deg", "identity"),
            ("No flip, 180 deg", "rot180"),
            ("Flip, 0 deg", "flip_lr"),
            ("Flip, 180 deg", "flip_ud"),
        ],
        value="auto",
    )
    bad_slice = widgets.Dropdown(
        description="Bad slice",
        options=[("None", "none"), ("Fixed", "fixed"), ("Moving", "moving")],
        value="none",
    )
    dx = widgets.FloatText(description="dx", value=0.0)
    dy = widgets.FloatText(description="dy", value=0.0)
    angle = widgets.FloatText(description="angle", value=0.0)
    radius = widgets.FloatText(description="radius", value=float(default_radius))
    angle_radius = widgets.FloatText(
        description="ang radius",
        value=float(default_angle_radius_deg),
    )
    contrast = widgets.FloatRangeSlider(
        description="contrast",
        min=image_min,
        max=image_max,
        step=(image_max - image_min) / 200.0,
        value=(default_vmin, default_vmax),
        readout_format=".3f",
        continuous_update=False,
        layout=widgets.Layout(width="500px"),
    )
    run_auto = widgets.Button(description="Run auto", button_style="primary")
    reload_pair = widgets.Button(description="Reload pair")
    run_local = widgets.Button(description="Local refine")
    reset_transform = widgets.Button(description="Reset to PC")
    accept = widgets.Button(description="Accept")
    save = widgets.Button(description="Save")
    prev_button = widgets.Button(description="Prev")
    next_button = widgets.Button(description="Next")
    output = widgets.Output()
    ui_state = {
        "pending_record": None,
        "controls_dirty": False,
        "seeding_controls": False,
        "busy_message": "",
        "rendering": False,
        "render_requested": False,
    }

    action_buttons = [
        run_auto,
        reload_pair,
        run_local,
        reset_transform,
        accept,
        save,
        prev_button,
        next_button,
    ]

    def current_pairs():
        return build_adjacent_slice_pairs(entries, bad_slices=state.get("bad_slices", []))

    def current_key():
        pairs = current_pairs()
        if not pairs:
            return None, None, None
        pair = pairs[pair_index.value]
        fixed_slice = int(pair[0]["slice_number"])
        moving_slice = int(pair[1]["slice_number"])
        return f"{fixed_slice:03d}_{moving_slice:03d}", pair[0], pair[1]

    def clamp_pair_index():
        pairs = current_pairs()
        pair_index.max = max(len(pairs) - 1, 0)
        if pair_index.value > pair_index.max:
            pair_index.value = pair_index.max

    def load_record():
        key, fixed_entry, moving_entry = current_key()
        if key is None:
            return None, None, None, None
        return key, fixed_entry, moving_entry, state["pairs"].get(key, {})

    def _float_or_default(value, default):
        if value is None:
            return float(default)
        return float(value)

    def _seed_transform_controls_from_record(record, *, use_auto=False):
        orientation_seed = (
            record.get("auto_orientation", record.get("orientation", "auto"))
            if use_auto
            else record.get("orientation", "auto")
        )
        ui_state["seeding_controls"] = True
        orientation.value = orientation_seed
        if orientation.value not in {opt[1] for opt in orientation.options}:
            orientation.value = "auto"
        if use_auto and record.get("auto_shift_xy") is not None:
            dx.value = float(record["auto_shift_xy"][0])
            dy.value = float(record["auto_shift_xy"][1])
        elif record.get("manual_shift_xy") is not None:
            dx.value = float(record["manual_shift_xy"][0])
            dy.value = float(record["manual_shift_xy"][1])
        else:
            dx.value = float(record.get("shift_xy", [0.0, 0.0])[0])
            dy.value = float(record.get("shift_xy", [0.0, 0.0])[1])
        angle_seed = (
            record.get("auto_angle_deg")
            if use_auto and record.get("auto_angle_deg") is not None
            else record.get("manual_angle_deg")
        )
        if angle_seed is None:
            angle_seed = record.get("angle_deg")
        angle.value = _float_or_default(angle_seed, 0.0)
        radius.value = _float_or_default(
            record.get("local_search_radius"), default_radius
        )
        angle_radius.value = _float_or_default(
            record.get("angle_radius_deg"), default_angle_radius_deg
        )
        ui_state["seeding_controls"] = False
        ui_state["controls_dirty"] = False
        ui_state["pending_record"] = None

    def seed_controls_from_record(record):
        ui_state["seeding_controls"] = True
        bad_labels = _pair_bad_slice_labels(
            load_record()[1], load_record()[2], state.get("bad_slices", [])
        )
        if any(label.startswith("fixed ") for label in bad_labels):
            bad_slice.value = "fixed"
        elif any(label.startswith("moving ") for label in bad_labels):
            bad_slice.value = "moving"
        else:
            bad_slice.value = "none"
        _seed_transform_controls_from_record(record, use_auto=False)

    def save_state():
        save_pairwise_registration_state(state_path, state)

    def _current_display_record():
        _key, _fixed_entry, _moving_entry, record = load_record()
        return ui_state["pending_record"] or record

    def _format_transform_html(record):
        if not record or record.get("matrix") is None:
            return "<b>Preview transform:</b> none"
        shift_xy = record.get("shift_xy", [0.0, 0.0])
        angle_deg = record.get("angle_deg", 0.0)
        orientation_name = record.get("orientation", "auto")
        method = record.get("method", "unknown")
        return (
            "<b>Preview transform:</b> "
            f"orientation={orientation_name} &nbsp;&nbsp; "
            f"dx={float(shift_xy[0]):.2f} &nbsp;&nbsp; "
            f"dy={float(shift_xy[1]):.2f} &nbsp;&nbsp; "
            f"angle={float(angle_deg):.2f} deg &nbsp;&nbsp; "
            f"method={method}"
        )

    def _update_transform_label():
        transform_label.value = _format_transform_html(_current_display_record())

    def _set_busy(message="", *, disable=False):
        ui_state["busy_message"] = message
        action_progress.layout.visibility = "visible" if message else "hidden"
        if message:
            action_progress.value = 100 if action_progress.value == 0 else 0
        else:
            action_progress.value = 0
        for button in action_buttons:
            button.disabled = disable
        action_label.value = (
            f"<b>Action:</b> {ui_state['busy_message']}"
            if ui_state["busy_message"]
            else ""
        )
        _update_transform_label()

    def _record_from_result(key, fixed_slice, moving_slice, result, *, manual_angle, manual_shift, local_radius, angular_radius, status):
        return {
            "fixed_slice": fixed_slice,
            "moving_slice": moving_slice,
            "orientation": result["orientation"],
            "auto_orientation": result["orientation"],
            "angle_deg": float(result["angle_deg"]),
            "auto_angle_deg": float(result["angle_deg"]),
            "shift_xy": [float(result["shift_xy"][0]), float(result["shift_xy"][1])],
            "auto_shift_xy": [float(result["shift_xy"][0]), float(result["shift_xy"][1])],
            "manual_angle_deg": float(manual_angle) if manual_angle is not None else None,
            "manual_shift_xy": [float(manual_shift[0]), float(manual_shift[1])]
            if manual_shift is not None
            else None,
            "local_search_radius": float(local_radius) if local_radius is not None else None,
            "angle_radius_deg": float(angular_radius),
            "matrix": np.asarray(result["matrix"]).tolist(),
            "score": float(result["score"]),
            "status": status,
            "method": result["method"],
        }

    def _preview_record_from_controls(fixed_entry, moving_entry, saved_record):
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        key = f"{fixed_slice:03d}_{moving_slice:03d}"
        fixed_image = images[index_by_slice[fixed_slice]]
        shape = fixed_image.shape
        base_record = dict(saved_record or {})
        orientation_name = orientation.value
        if orientation_name == "auto":
            orientation_name = base_record.get("orientation", "identity")
            if orientation_name == "auto":
                orientation_name = "identity"
        matrix = (
            _translation_matrix(dx.value, dy.value)
            @ _rotation_matrix(angle.value, shape)
            @ _orientation_matrix(shape, orientation_name)
        )
        prior_status = base_record.get("status", "pending")
        next_status = "accepted" if prior_status == "accepted" else "pending"
        return key, {
            "fixed_slice": fixed_slice,
            "moving_slice": moving_slice,
            "orientation": orientation_name,
            "auto_orientation": base_record.get("auto_orientation", base_record.get("orientation")),
            "angle_deg": float(angle.value),
            "auto_angle_deg": base_record.get("auto_angle_deg", base_record.get("angle_deg")),
            "shift_xy": [float(dx.value), float(dy.value)],
            "auto_shift_xy": base_record.get("auto_shift_xy", base_record.get("shift_xy")),
            "manual_angle_deg": float(angle.value),
            "manual_shift_xy": [float(dx.value), float(dy.value)],
            "local_search_radius": base_record.get("local_search_radius", radius.value),
            "angle_radius_deg": base_record.get("angle_radius_deg", angle_radius.value),
            "matrix": np.asarray(matrix).tolist(),
            "score": base_record.get("score"),
            "status": next_status,
            "method": "widget_preview_affine",
            "error": base_record.get("error"),
        }

    def _estimate_from_controls(fixed_entry, moving_entry, *, use_local, force_auto):
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        key = f"{fixed_slice:03d}_{moving_slice:03d}"
        fixed_image = images[index_by_slice[fixed_slice]]
        moving_image = images[index_by_slice[moving_slice]]
        fixed_mask = masks[index_by_slice[fixed_slice]]
        moving_mask = masks[index_by_slice[moving_slice]]
        existing = state.get("pairs", {}).get(key, {})
        if force_auto:
            forced_orientation = "auto"
            manual_shift = None
            manual_angle = None
            local_radius = None
        else:
            forced_orientation = orientation.value
            use_manual_seed = (
                forced_orientation != "auto" or ui_state["controls_dirty"]
            )
            if use_manual_seed:
                manual_shift = [dx.value, dy.value]
                manual_angle = angle.value
            else:
                manual_shift = None
                manual_angle = None
            local_radius = radius.value if use_local else None
        angular_radius = angle_radius.value
        result = estimate_pairwise_registration(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            orientation=forced_orientation,
            manual_angle_deg=manual_angle,
            manual_shift_xy=manual_shift,
            local_search_radius=local_radius,
            angle_radius_deg=angular_radius,
        )
        prior_status = state.get("pairs", {}).get(key, {}).get("status", "pending")
        next_status = "accepted" if prior_status == "accepted" else "pending"
        return key, _record_from_result(
            key,
            fixed_slice,
            moving_slice,
            result,
            manual_angle=manual_angle,
            manual_shift=manual_shift,
            local_radius=local_radius,
            angular_radius=angular_radius,
            status=existing.get("status", next_status),
        )

    def _commit_current_record(*, accept_record=False):
        key, fixed_entry, moving_entry, saved_record = load_record()
        if key is None:
            return None
        record = ui_state["pending_record"]
        if record is None and ui_state["controls_dirty"]:
            _, record = _preview_record_from_controls(
                fixed_entry, moving_entry, saved_record
            )
        if record is None:
            record = dict(saved_record) if saved_record else None
        if record is None:
            _, record = _estimate_from_controls(
                fixed_entry, moving_entry, use_local=False, force_auto=True
            )
        record = dict(record)
        if accept_record:
            record["status"] = "accepted"
        state["pairs"][key] = record
        save_state()
        ui_state["pending_record"] = None
        ui_state["controls_dirty"] = False
        return record

    def render_current(*, seed_controls=False):
        if ui_state["rendering"]:
            ui_state["render_requested"] = True
            return
        ui_state["rendering"] = True
        clamp_pair_index()
        key, fixed_entry, moving_entry, record = load_record()
        try:
            with output:
                clear_output(wait=True)
                if key is None:
                    pair_label.value = "<b>No active pairs left.</b>"
                    status_label.value = ""
                    action_label.value = ""
                    return
                if seed_controls and record:
                    seed_controls_from_record(record)
                display_record = ui_state["pending_record"] or record
                bad_labels = _pair_bad_slice_labels(
                    fixed_entry, moving_entry, state.get("bad_slices", [])
                )
                pair_label.value = _pair_label_html(
                    fixed_entry, moving_entry, state.get("bad_slices", [])
                )
                status = (
                    display_record.get("status", "pending")
                    if display_record
                    else "pending"
                )
                score = display_record.get("score") if display_record else None
                score_value = float(score) if score is not None else np.nan
                if bad_labels:
                    status = f"{status}; contains bad slice"
                if ui_state["pending_record"] is not None:
                    status = f"{status}; preview not yet saved"
                elif ui_state["controls_dirty"]:
                    status = f"{status}; controls changed, press Reload pair"
                status_label.value = (
                    f"<b>Status:</b> {status} &nbsp;&nbsp; <b>Score:</b> {score_value:.4f}"
                    if np.isfinite(score_value)
                    else f"<b>Status:</b> {status}"
                )
                transform_label.value = _format_transform_html(display_record)
                action_label.value = (
                    f"<b>Action:</b> {ui_state['busy_message']}"
                    if ui_state["busy_message"]
                    else ""
                )
                if display_record:
                    fixed_slice = int(fixed_entry["slice_number"])
                    moving_slice = int(moving_entry["slice_number"])
                    fixed_mask = masks[index_by_slice[fixed_slice]].astype(bool)
                    moving_mask = masks[index_by_slice[moving_slice]].astype(bool)
                    if display_record.get("matrix") is not None:
                        matrix = np.asarray(display_record["matrix"], dtype=float)
                        warped = warp_with_affine(
                            images[index_by_slice[moving_slice]].astype(np.float32),
                            matrix,
                            output_shape=images[index_by_slice[fixed_slice]].shape,
                            order=1,
                        )
                        warped_mask = (
                            warp_with_affine(
                                moving_mask.astype(np.float32),
                                matrix,
                                output_shape=images[index_by_slice[fixed_slice]].shape,
                                order=0,
                            )
                            > 0.5
                        )
                        title = (
                            f"{display_record['orientation']} | "
                            f"{display_record.get('method', 'saved')}"
                        )
                    else:
                        warped = images[index_by_slice[moving_slice]].astype(np.float32)
                        warped_mask = moving_mask
                        title = "unregistered | no saved transform"
                    if bad_labels:
                        title += " | contains bad slice"
                    fig = _render_overlay(
                        images[index_by_slice[fixed_slice]],
                        warped,
                        title,
                        vmin=float(contrast.value[0]),
                        vmax=float(contrast.value[1]),
                        fixed_mask=fixed_mask,
                        warped_mask=warped_mask,
                    )
                    display(fig)
                    plt.close(fig)
                    if display_record.get("error"):
                        print(f"Last error: {display_record['error']}")
                else:
                    print(
                        "No saved result for this pair yet. Use 'Run auto' or 'Local refine'."
                    )
        finally:
            ui_state["rendering"] = False
            rerender = ui_state["render_requested"]
            ui_state["render_requested"] = False
        if rerender:
            render_current(seed_controls=False)

    def _bridge_pair_across(bad_id, current_bad_slices):
        """Return the (fixed, moving) entries that will bridge `bad_id` once excluded.

        None if `bad_id` sits at the edge of the active list and no bridge
        registration is needed.
        """
        active_slice_list = sorted(
            int(entry["slice_number"])
            for entry in entries
            if int(entry["slice_number"]) not in {int(s) for s in current_bad_slices}
            and int(entry["slice_number"]) != int(bad_id)
        )
        pre_exclusion = sorted(
            int(entry["slice_number"])
            for entry in entries
            if int(entry["slice_number"]) not in {int(s) for s in current_bad_slices}
        )
        if int(bad_id) not in pre_exclusion:
            return None
        idx = pre_exclusion.index(int(bad_id))
        if idx == 0 or idx == len(pre_exclusion) - 1:
            return None
        lower_slice = active_slice_list[idx - 1]
        higher_slice = active_slice_list[idx]
        return (
            entries[index_by_slice[higher_slice]],
            entries[index_by_slice[lower_slice]],
        )

    def on_accept(_):
        key, _, _, record = load_record()
        if key is None:
            return
        _set_busy("Saving accepted pair...", disable=True)
        committed = _commit_current_record(accept_record=True)
        if committed is None:
            _set_busy("", disable=False)
            return
        bridge_pair = None
        if bad_slice.value != "none":
            if bad_slice.value == "fixed":
                bad_id = committed["fixed_slice"]
            else:
                bad_id = committed["moving_slice"]
            bridge_pair = _bridge_pair_across(bad_id, state.get("bad_slices", []))
            bad = {int(v) for v in state.get("bad_slices", [])}
            bad.add(int(bad_id))
            state["bad_slices"] = sorted(bad)
        save_state()
        if bridge_pair is not None:
            bridge_key = (
                f"{int(bridge_pair[0]['slice_number']):03d}_"
                f"{int(bridge_pair[1]['slice_number']):03d}"
            )
            with output:
                print(f"Auto-registering bridge pair {bridge_key}...")
            _, bridge_record = _estimate_from_controls(
                bridge_pair[0],
                bridge_pair[1],
                use_local=False,
                force_auto=True,
            )
            state["pairs"][bridge_key] = bridge_record
            save_state()
        if bad_slice.value != "none":
            try:
                write_bad_slices_to_chamber_ops(
                    state_path=state_path, warn_missing=False
                )
            except Exception as exc:
                with output:
                    print(f"Failed to mirror bad_slices into chamber ops.yml: {exc}")
        _set_busy("", disable=False)
        render_current(seed_controls=True)

    def on_save(_):
        _set_busy("Saving preview...", disable=True)
        _commit_current_record(accept_record=False)
        _set_busy("", disable=False)
        render_current(seed_controls=True)

    def on_reload_pair(_):
        key, fixed_entry, moving_entry, saved_record = load_record()
        if key is None:
            return
        if not saved_record:
            return
        _set_busy("Reapplying transform preview...", disable=True)
        _, preview_record = _preview_record_from_controls(
            fixed_entry,
            moving_entry,
            saved_record,
        )
        ui_state["pending_record"] = preview_record
        ui_state["controls_dirty"] = False
        _update_transform_label()
        _set_busy("", disable=False)
        render_current(seed_controls=False)

    def on_run_auto(_):
        key, fixed_entry, moving_entry, _record = load_record()
        if key is None:
            return
        _set_busy("Running automatic registration...", disable=True)
        key, auto_record = _estimate_from_controls(
            fixed_entry,
            moving_entry,
            use_local=False,
            force_auto=True,
        )
        state["pairs"][key] = auto_record
        save_state()
        ui_state["pending_record"] = None
        ui_state["controls_dirty"] = False
        _set_busy("", disable=False)
        render_current(seed_controls=True)

    def on_run_local(_):
        key, fixed_entry, moving_entry, _record = load_record()
        if key is None:
            return
        _set_busy("Running local refine...", disable=True)
        key, local_record = _estimate_from_controls(
            fixed_entry,
            moving_entry,
            use_local=True,
            force_auto=False,
        )
        state["pairs"][key] = local_record
        save_state()
        ui_state["pending_record"] = None
        ui_state["controls_dirty"] = False
        _set_busy("", disable=False)
        render_current(seed_controls=True)

    def on_reset_transform(_):
        key, _fixed_entry, _moving_entry, saved_record = load_record()
        if key is None or not saved_record:
            return
        _set_busy("Resetting to automatic fit...", disable=True)
        _seed_transform_controls_from_record(saved_record, use_auto=True)
        _, preview_record = _preview_record_from_controls(*load_record()[1:3], saved_record)
        ui_state["pending_record"] = preview_record
        ui_state["controls_dirty"] = False
        _update_transform_label()
        _set_busy("", disable=False)
        render_current(seed_controls=False)

    def on_prev(_):
        pair_index.value = max(pair_index.value - 1, 0)

    def on_next(_):
        pair_index.value = min(pair_index.value + 1, pair_index.max)

    def on_control_change(change):
        if ui_state["seeding_controls"]:
            return
        ui_state["controls_dirty"] = True
        ui_state["pending_record"] = None

    def on_bad_slice_change(change):
        if ui_state["seeding_controls"]:
            return
        render_current(seed_controls=False)

    run_auto.on_click(on_run_auto)
    reload_pair.on_click(on_reload_pair)
    run_local.on_click(on_run_local)
    reset_transform.on_click(on_reset_transform)
    accept.on_click(on_accept)
    save.on_click(on_save)
    prev_button.on_click(on_prev)
    next_button.on_click(on_next)

    clamp_pair_index()
    render_current(seed_controls=True)

    pair_index.observe(lambda change: render_current(seed_controls=True), names="value")
    for control in (orientation, dx, dy, angle, radius, angle_radius):
        control.observe(on_control_change, names="value")
    bad_slice.observe(on_bad_slice_change, names="value")
    contrast.observe(lambda change: render_current(seed_controls=False), names="value")
    widget = widgets.VBox(
        [
            pair_label,
            status_label,
            action_label,
            action_progress,
            pair_index,
            widgets.HBox(
                [
                    prev_button,
                    next_button,
                    run_auto,
                    reload_pair,
                    run_local,
                    reset_transform,
                    accept,
                    save,
                ]
            ),
            transform_label,
            widgets.HBox([orientation, bad_slice]),
            widgets.HBox([dx, dy, angle]),
            widgets.HBox([radius, angle_radius]),
            contrast,
            output,
        ]
    )
    return widget
