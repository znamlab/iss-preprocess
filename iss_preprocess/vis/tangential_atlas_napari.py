"""napari reviewer for the tangential atlas.

The reviewer is intentionally a thin UI over the pipeline context. Slice
identity is resolved by ``slice_number`` via
``build_tangential_atlas_context``; the kept-list index is only a widget
position.
"""

from pathlib import Path

import numpy as np

__all__ = ["review_tangential_atlas_napari"]


def _record_for_slice_index(context, slice_index):
    M = len(context.slice_records)
    if (
        isinstance(slice_index, bool)
        or not isinstance(slice_index, (int, np.integer))
        or not (0 <= int(slice_index) < M)
    ):
        raise ValueError(
            f"slice_index must be in range(0, {M}), got {slice_index!r}"
        )
    return context.slice_records[int(slice_index)]


def _slice_image_for_index(context, slice_index):
    from iss_preprocess.pipeline import image_for_tangential_atlas_slice

    record = _record_for_slice_index(context, slice_index)
    return np.asarray(
        image_for_tangential_atlas_slice(context, record.slice_number),
        dtype=np.float32,
    )


def _annotation_border(annotation):
    from skimage.segmentation import find_boundaries

    return find_boundaries(np.asarray(annotation), mode="inner").astype(np.uint8)


def _slice_metadata_text(context, slice_index):
    record = _record_for_slice_index(context, slice_index)
    issues = [
        issue
        for issue in context.issues
        if issue.slice_number is None or issue.slice_number == record.slice_number
    ]
    warning_codes = [issue.code for issue in issues if issue.severity == "warning"]
    parts = [
        f"slice_number={record.slice_number}",
        f"kept_index={record.kept_index}",
        f"image_index={record.image_index}",
        f"z_um={record.z_um:g}",
        f"roi={record.roi}",
        f"chamber={record.chamber}",
        f"overview_px_um={context.metadata.overview_pixel_size_um:g}",
        f"section_um={context.metadata.section_thickness_um:g}",
        f"z_source={context.metadata.z_source}",
    ]
    if warning_codes:
        parts.append("warnings=" + ",".join(sorted(set(warning_codes))))
    return " | ".join(parts)


def _render_current_planes(context, atlas, pose, residuals, slice_index):
    from iss_preprocess.pipeline import (
        build_slice_plane_spec,
        render_annotation_plane,
        render_reference_plane,
    )

    record = _record_for_slice_index(context, slice_index)
    spec = build_slice_plane_spec(
        context.geometry,
        pose,
        residual=residuals[int(record.slice_number)],
        slice_index=int(record.kept_index),
    )
    reference = render_reference_plane(atlas, spec)
    ann_spec = build_slice_plane_spec(
        context.geometry,
        pose,
        residual=residuals[int(record.slice_number)],
        slice_index=int(record.kept_index),
    )
    ann_spec = ann_spec.__class__(
        out_shape_yx=ann_spec.out_shape_yx,
        pixel_size_um=ann_spec.pixel_size_um,
        atlas_um_from_grid_4x4=ann_spec.atlas_um_from_grid_4x4,
        order=0,
        cval=0,
    )
    annotation = render_annotation_plane(atlas, ann_spec)
    return reference, annotation, _annotation_border(annotation)


def review_tangential_atlas_napari(
    data_path,
    *,
    state_path=None,
    atlas_name: str = "allen_mouse_10um",
    anatomy_prefix: str = "hybridisation_round_1_1",
    anatomy_channel_index: int = 1,
    overview_pixel_size_um: float = None,
    section_thickness_um: float = None,
    strict_metadata: bool = False,
):
    """Open the tangential atlas napari reviewer.

    Parameters
    ----------
    data_path : str or Path
        Mouse-level processed root, or the ``unregistered_slices.npz``
        file directly.
    state_path : str or Path, optional
        Explicit state-file path. If ``None``, derive from
        ``data_path``.
    atlas_name : str
        BrainGlobe atlas identifier.
    anatomy_prefix, anatomy_channel_index :
        Reserved for M7+ — the slice rendered is currently the canvas
        from the unregistered stack at the chosen kept-list index.
    overview_pixel_size_um, section_thickness_um : float, optional
        Explicit metadata overrides passed to
        :func:`build_tangential_atlas_context`. Section thickness falls
        back visibly to 20.0 µm in that context when unresolved.
    strict_metadata : bool
        If True, context warnings are promoted to errors.

    Returns
    -------
    napari.Viewer
    """
    try:
        import napari
    except ImportError as exc:
        raise ImportError(
            "napari is required for review_tangential_atlas_napari; "
            "install with 'pip install napari'"
        ) from exc

    try:
        from magicgui import magicgui
    except ImportError as exc:
        raise ImportError(
            "magicgui is required for review_tangential_atlas_napari; "
            "it is normally a transitive dependency of napari"
        ) from exc

    from iss_preprocess.pipeline import (
        SliceResidual,
        StackPose,
        build_tangential_atlas_context,
        load_atlas,
        pose_to_matrix,
        save_tangential_atlas_state,
    )

    context = build_tangential_atlas_context(
        data_path,
        state_path=state_path,
        overview_pixel_size_um=overview_pixel_size_um,
        section_thickness_um=section_thickness_um,
        strict=strict_metadata,
    )
    geometry = context.geometry
    M = len(geometry.slice_numbers)
    if M < 1:
        raise ValueError("no kept slices in the stack")

    atlas = load_atlas(atlas_name)

    init_pose = StackPose(
        atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
    )
    residuals = {
        int(sn): SliceResidual(slice_number=int(sn))
        for sn in geometry.slice_numbers.tolist()
    }

    state = context.state
    if state is not None:
        pose_dict = state.get("pose", {})
        if "atlas_from_tangential_4x4" in pose_dict:
            init_pose = StackPose(
                atlas_from_tangential=np.asarray(
                    pose_dict["atlas_from_tangential_4x4"], dtype=np.float64
                ),
                yaw_deg=float(pose_dict.get("yaw_deg", 0.0)),
                pitch_deg=float(pose_dict.get("pitch_deg", 0.0)),
                roll_deg=float(pose_dict.get("roll_deg", 0.0)),
                depth_um=float(pose_dict.get("depth_um", 0.0)),
                tx_atlas_um=float(pose_dict.get("tx_atlas_um", 0.0)),
                ty_atlas_um=float(pose_dict.get("ty_atlas_um", 0.0)),
            )
        for k, v in state.get("residuals", {}).items():
            sn = int(k)
            if sn in residuals:
                residuals[sn] = SliceResidual(
                    slice_number=sn,
                    dx_um=float(v.get("dx_um", 0.0)),
                    dy_um=float(v.get("dy_um", 0.0)),
                    dtheta_deg=float(v.get("dtheta_deg", 0.0)),
                    dz_um=0.0,
                )

    state_holder = {
        "pose": init_pose,
        "slice_index": int(M // 2),
    }

    def _current_slice_image():
        return _slice_image_for_index(context, state_holder["slice_index"])

    def _render_planes():
        return _render_current_planes(
            context,
            atlas,
            state_holder["pose"],
            residuals,
            state_holder["slice_index"],
        )

    def _update_metadata_overlay():
        text = _slice_metadata_text(context, state_holder["slice_index"])
        if hasattr(viewer, "text_overlay"):
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = text
        for layer in (slice_layer, atlas_layer, annotation_layer, border_layer):
            layer.metadata["tangential_atlas"] = text

    viewer = napari.Viewer()
    ref_plane, annotation_plane, border_plane = _render_planes()
    slice_layer = viewer.add_image(
        _current_slice_image(), name="slice", colormap="gray"
    )
    atlas_layer = viewer.add_image(
        ref_plane, name="atlas_reference",
        colormap="magma", opacity=0.5,
    )
    annotation_layer = viewer.add_labels(
        annotation_plane.astype(np.int64), name="atlas_annotation", opacity=0.25
    )
    border_layer = viewer.add_image(
        border_plane, name="atlas_annotation_border", colormap="green", opacity=0.8
    )
    _update_metadata_overlay()

    @magicgui(
        auto_call=True,
        slice_index={"widget_type": "SpinBox", "min": 0, "max": M - 1},
        yaw_deg={"widget_type": "FloatSlider", "min": -30.0, "max": 30.0, "step": 0.5},
        pitch_deg={"widget_type": "FloatSlider", "min": -30.0, "max": 30.0, "step": 0.5},
        roll_deg={"widget_type": "FloatSlider", "min": -30.0, "max": 30.0, "step": 0.5},
        depth_um={"widget_type": "FloatSlider", "min": -1000.0, "max": 1000.0, "step": 5.0},
        tx_atlas_um={"widget_type": "FloatSlider", "min": -1000.0, "max": 1000.0, "step": 5.0},
        ty_atlas_um={"widget_type": "FloatSlider", "min": -1000.0, "max": 1000.0, "step": 5.0},
    )
    def pose_controls(
        slice_index: int = state_holder["slice_index"],
        yaw_deg: float = init_pose.yaw_deg,
        pitch_deg: float = init_pose.pitch_deg,
        roll_deg: float = init_pose.roll_deg,
        depth_um: float = init_pose.depth_um,
        tx_atlas_um: float = init_pose.tx_atlas_um,
        ty_atlas_um: float = init_pose.ty_atlas_um,
    ):
        state_holder["slice_index"] = int(slice_index)
        state_holder["pose"] = StackPose(
            atlas_from_tangential=pose_to_matrix(
                yaw_deg, pitch_deg, roll_deg,
                depth_um, tx_atlas_um, ty_atlas_um,
            ),
            yaw_deg=float(yaw_deg), pitch_deg=float(pitch_deg),
            roll_deg=float(roll_deg), depth_um=float(depth_um),
            tx_atlas_um=float(tx_atlas_um), ty_atlas_um=float(ty_atlas_um),
        )
        slice_layer.data = _current_slice_image()
        ref_plane, annotation_plane, border_plane = _render_planes()
        atlas_layer.data = ref_plane
        annotation_layer.data = annotation_plane.astype(np.int64)
        border_layer.data = border_plane
        _update_metadata_overlay()

    @magicgui(
        auto_call=True,
        dx_um={"widget_type": "FloatSlider", "min": -250.0, "max": 250.0, "step": 1.0},
        dy_um={"widget_type": "FloatSlider", "min": -250.0, "max": 250.0, "step": 1.0},
        dtheta_deg={"widget_type": "FloatSlider", "min": -2.0, "max": 2.0, "step": 0.05},
    )
    def residual_controls(
        dx_um: float = 0.0, dy_um: float = 0.0, dtheta_deg: float = 0.0
    ):
        i = state_holder["slice_index"]
        sn = int(geometry.slice_numbers[i])
        residuals[sn] = SliceResidual(
            slice_number=sn,
            dx_um=float(dx_um),
            dy_um=float(dy_um),
            dtheta_deg=float(dtheta_deg),
            dz_um=0.0,
        )
        ref_plane, annotation_plane, border_plane = _render_planes()
        atlas_layer.data = ref_plane
        annotation_layer.data = annotation_plane.astype(np.int64)
        border_layer.data = border_plane
        _update_metadata_overlay()

    @magicgui(call_button="Save state")
    def save_controls():
        pose = state_holder["pose"]
        state_payload = {
            "atlas_name": atlas_name,
            "atlas_resolution_um": float(atlas.resolution_um),
            "stack_path": str(context.stack_path),
            "global_transforms_path": str(context.transforms_path),
            "canvas_shape_yx": list(geometry.canvas_shape_yx),
            "overview_pixel_size_um": float(geometry.overview_pixel_size_um),
            "section_thickness_um": float(context.metadata.section_thickness_um),
            "z_um_by_slice": {
                str(int(sn)): float(z)
                for sn, z in zip(geometry.slice_numbers, geometry.z_um)
            },
            "bad_slices": geometry.bad_slice_numbers.tolist(),
            "pose": {
                "atlas_from_tangential_4x4": pose.atlas_from_tangential.tolist(),
                "yaw_deg": pose.yaw_deg,
                "pitch_deg": pose.pitch_deg,
                "roll_deg": pose.roll_deg,
                "depth_um": pose.depth_um,
                "tx_atlas_um": pose.tx_atlas_um,
                "ty_atlas_um": pose.ty_atlas_um,
                "rotation_order": "ZYX",
                "axes": {"yaw": "DV", "pitch": "ML", "roll": "AP"},
            },
            "residuals": {
                str(sn): {
                    "dx_um": float(r.dx_um),
                    "dy_um": float(r.dy_um),
                    "dtheta_deg": float(r.dtheta_deg),
                    "dz_um": 0.0,
                }
                for sn, r in residuals.items()
            },
            "residual_bounds": {
                "max_dx_um": 250.0,
                "max_dy_um": 250.0,
                "max_dtheta_deg": 2.0,
                "max_dz_um": 0.0,
                "smoothness_lambda": 0.0,
                "centering": "mean_zero",
            },
        }
        out = save_tangential_atlas_state(state_payload, state_path=state_path)
        print(f"saved tangential atlas state to {out}")

    viewer.window.add_dock_widget(pose_controls, area="right", name="Pose + slice")
    viewer.window.add_dock_widget(residual_controls, area="right", name="Per-slice residual")
    viewer.window.add_dock_widget(save_controls, area="right", name="Save")

    _ = anatomy_prefix, anatomy_channel_index  # reserved for future real-channel loading
    return viewer
