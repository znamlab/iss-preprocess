"""Napari-based reviewer for pairwise slice registrations.

GPU-accelerated alternative to ``review_pairwise_registrations_widget``.
Pops a Qt window via napari and runs on a VM with a display server
(VNC / NoMachine / X11). Mirrors the behaviour of the ipywidgets widget,
including bad-slice handling, bridge-pair auto-registration, and the
mirroring of ``bad_slices`` into each chamber's ``ops.yml``.
"""
from pathlib import Path

import numpy as np

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


_ORIENTATION_OPTIONS = [
    ("Auto", "auto"),
    ("No flip, 0 deg", "identity"),
    ("No flip, 180 deg", "rot180"),
    ("Flip, 0 deg", "flip_lr"),
    ("Flip, 180 deg", "flip_ud"),
]
_BAD_SLICE_OPTIONS = [("None", "none"), ("Fixed", "fixed"), ("Moving", "moving")]


def _make_pyramid(image, n_levels=4, reduce="mean", min_dim=32):
    """Build a list of progressively 2x-downsampled views.

    `reduce="mean"` for intensity images; `reduce="max"` for boolean masks
    so a pixel survives if any of its 2x2 children was True.
    """
    if reduce == "mean":
        op = np.mean
    elif reduce == "max":
        op = np.max
    else:
        raise ValueError(f"Unknown reduce: {reduce!r}")
    levels = [np.asarray(image)]
    for _ in range(int(n_levels) - 1):
        prev = levels[-1]
        h, w = prev.shape[:2]
        nh, nw = h // 2, w // 2
        if nh < min_dim or nw < min_dim:
            break
        cropped = prev[: nh * 2, : nw * 2]
        if cropped.ndim == 2:
            block = cropped.reshape(nh, 2, nw, 2)
            ds = op(block, axis=(1, 3))
        else:
            block = cropped.reshape(nh, 2, nw, 2, -1)
            ds = op(block, axis=(1, 3))
        levels.append(ds.astype(prev.dtype, copy=False))
    return levels


def review_pairwise_registrations_napari(
    stack_path,
    state_path=None,
    default_radius=150.0,
    default_angle_radius_deg=5.0,
    pyramid_levels=4,
):
    """Open a napari window for reviewing pairwise slice registrations.

    Mirrors :func:`review_pairwise_registrations_widget` functionality but
    uses napari's GPU canvas and Qt controls. Requires a display server on
    the host (VNC, NoMachine, X11 forwarding).

    Returns the napari ``Viewer``. In a Jupyter cell the window opens
    immediately. From a plain script call ``napari.run()`` afterwards to
    enter the Qt event loop.
    """
    try:
        import napari  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "review_pairwise_registrations_napari requires `napari`. "
            "Install with: pip install 'iss-preprocess[napari]'"
        ) from exc
    reviewer = _NapariPairwiseReviewer(
        stack_path=stack_path,
        state_path=state_path,
        default_radius=default_radius,
        default_angle_radius_deg=default_angle_radius_deg,
        pyramid_levels=pyramid_levels,
    )
    return reviewer.viewer


class _NapariPairwiseReviewer:
    def __init__(
        self,
        stack_path,
        state_path=None,
        default_radius=150.0,
        default_angle_radius_deg=5.0,
        pyramid_levels=4,
    ):
        import napari
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QComboBox,
            QDoubleSpinBox,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        self._Qt = Qt
        self._QLabel = QLabel
        self.stack_path = Path(stack_path)
        self.state_path = (
            Path(state_path)
            if state_path is not None
            else self.stack_path.with_name("pairwise_registration_state.json")
        )
        self.default_radius = float(default_radius)
        self.default_angle_radius_deg = float(default_angle_radius_deg)
        self.pyramid_levels = int(pyramid_levels)

        stack = load_unregistered_volume_stack(self.stack_path)
        self.entries = stack["manifest"]["entries"]
        self.images = stack["images"]
        self.masks = stack["user_masks"]
        self.index_by_slice = {
            int(e["slice_number"]): i for i, e in enumerate(self.entries)
        }

        self.state = load_pairwise_registration_state(self.state_path)
        self.state.setdefault("bad_slices", [])
        self.state.setdefault("pairs", {})

        finite = self.images[np.isfinite(self.images)]
        if finite.size:
            self._default_vmin = float(np.percentile(finite, 1))
            self._default_vmax = float(np.percentile(finite, 99.5))
        else:
            self._default_vmin = 0.0
            self._default_vmax = 1.0
        if self._default_vmax <= self._default_vmin:
            self._default_vmax = self._default_vmin + 1.0

        self._seeding = False
        self._controls_dirty = False
        self._pending_record = None  # uncommitted preview record
        self._busy = False

        # Cached fixed-image pyramid keyed by slice_number to skip work
        # on pair switches that share the fixed slice (rare) and on
        # transform-only updates (always — the fixed image never changes).
        self._fixed_pyramid_cache = {}

        # ---- Build viewer + layers --------------------------------------
        self.viewer = napari.Viewer(title="Pairwise registration review")
        self._build_layers()

        # ---- Build dock widget ------------------------------------------
        self._build_controls(
            QWidget,
            QVBoxLayout,
            QHBoxLayout,
            QGridLayout,
            QPushButton,
            QLabel,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
        )
        self.viewer.window.add_dock_widget(
            self._dock, name="Pairwise review", area="right"
        )

        self._refresh_pair_count()
        if self._n_pairs() > 0:
            self._on_pair_changed(0, seed_controls=True)
        else:
            self._update_status_label("No active pairs.")

    # ====================================================================
    # Layer setup
    # ====================================================================

    def _build_layers(self):
        # Use a small placeholder until the first pair is loaded; napari
        # requires layers to exist before they can be addressed by name.
        H, W = self.images.shape[1], self.images.shape[2]
        empty = np.zeros((H, W), dtype=np.float32)
        empty_mask = np.zeros((H, W), dtype=bool)
        cl = (self._default_vmin, self._default_vmax)

        self._fixed_layer = self.viewer.add_image(
            empty,
            name="fixed",
            colormap="cyan",
            blending="additive",
            contrast_limits=cl,
            visible=True,
        )
        self._moving_layer = self.viewer.add_image(
            empty,
            name="warped moving",
            colormap="magenta",
            blending="additive",
            contrast_limits=cl,
            visible=True,
        )
        self._fixed_mask_layer = self.viewer.add_labels(
            empty_mask.astype(np.uint8),
            name="fixed mask",
            opacity=0.25,
            visible=False,
        )
        self._moving_mask_layer = self.viewer.add_labels(
            empty_mask.astype(np.uint8),
            name="warped moving mask",
            opacity=0.25,
            visible=False,
        )

    # ====================================================================
    # Dock widget
    # ====================================================================

    def _build_controls(
        self,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QPushButton,
        QLabel,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
    ):
        dock = QWidget()
        layout = QVBoxLayout(dock)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._pair_label = QLabel("")
        self._pair_label.setWordWrap(True)
        layout.addWidget(self._pair_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._transform_label = QLabel("")
        self._transform_label.setWordWrap(True)
        layout.addWidget(self._transform_label)

        # Pair selector row
        pair_row = QHBoxLayout()
        self._prev_btn = QPushButton("Prev")
        self._next_btn = QPushButton("Next")
        self._pair_spin = QSpinBox()
        self._pair_spin.setRange(0, 0)
        self._pair_spin.setKeyboardTracking(False)
        self._pair_count_label = QLabel("/ 0")
        pair_row.addWidget(QLabel("Pair:"))
        pair_row.addWidget(self._pair_spin)
        pair_row.addWidget(self._pair_count_label)
        pair_row.addWidget(self._prev_btn)
        pair_row.addWidget(self._next_btn)
        pair_row.addStretch(1)
        layout.addLayout(pair_row)

        # Orientation + bad slice
        grid = QGridLayout()
        grid.addWidget(QLabel("Orientation:"), 0, 0)
        self._orientation_combo = QComboBox()
        for label, value in _ORIENTATION_OPTIONS:
            self._orientation_combo.addItem(label, value)
        grid.addWidget(self._orientation_combo, 0, 1)
        grid.addWidget(QLabel("Bad slice:"), 0, 2)
        self._bad_slice_combo = QComboBox()
        for label, value in _BAD_SLICE_OPTIONS:
            self._bad_slice_combo.addItem(label, value)
        grid.addWidget(self._bad_slice_combo, 0, 3)

        # dx, dy, angle
        self._dx_spin = self._make_double(QDoubleSpinBox, -10000, 10000, 3, 0.5)
        self._dy_spin = self._make_double(QDoubleSpinBox, -10000, 10000, 3, 0.5)
        self._angle_spin = self._make_double(QDoubleSpinBox, -180, 180, 3, 0.1)
        grid.addWidget(QLabel("dx:"), 1, 0)
        grid.addWidget(self._dx_spin, 1, 1)
        grid.addWidget(QLabel("dy:"), 1, 2)
        grid.addWidget(self._dy_spin, 1, 3)
        grid.addWidget(QLabel("angle (deg):"), 2, 0)
        grid.addWidget(self._angle_spin, 2, 1)

        # local search radii
        self._radius_spin = self._make_double(
            QDoubleSpinBox, 0, 1e5, 2, 5.0, value=self.default_radius
        )
        self._angle_radius_spin = self._make_double(
            QDoubleSpinBox, 0, 90, 2, 0.5, value=self.default_angle_radius_deg
        )
        grid.addWidget(QLabel("radius:"), 2, 2)
        grid.addWidget(self._radius_spin, 2, 3)
        grid.addWidget(QLabel("ang radius:"), 3, 0)
        grid.addWidget(self._angle_radius_spin, 3, 1)
        layout.addLayout(grid)

        # Action buttons
        btn_row1 = QHBoxLayout()
        self._run_auto_btn = QPushButton("Run auto")
        self._run_local_btn = QPushButton("Local refine")
        self._reload_btn = QPushButton("Reload preview")
        self._reset_btn = QPushButton("Reset to PC")
        btn_row1.addWidget(self._run_auto_btn)
        btn_row1.addWidget(self._run_local_btn)
        btn_row1.addWidget(self._reload_btn)
        btn_row1.addWidget(self._reset_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self._accept_btn = QPushButton("Accept")
        self._save_btn = QPushButton("Save")
        btn_row2.addWidget(self._accept_btn)
        btn_row2.addWidget(self._save_btn)
        btn_row2.addStretch(1)
        layout.addLayout(btn_row2)

        # Toggle helpers for the masks (off by default since napari already
        # has the per-layer visibility toggle; this is a convenience)
        toggle_row = QHBoxLayout()
        self._toggle_masks_btn = QPushButton("Toggle masks")
        toggle_row.addWidget(self._toggle_masks_btn)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        layout.addStretch(1)
        self._dock = dock

        self._action_buttons = [
            self._run_auto_btn,
            self._run_local_btn,
            self._reload_btn,
            self._reset_btn,
            self._accept_btn,
            self._save_btn,
            self._prev_btn,
            self._next_btn,
        ]

        # Wire signals
        self._pair_spin.valueChanged.connect(self._on_pair_spin_changed)
        self._prev_btn.clicked.connect(self._on_prev)
        self._next_btn.clicked.connect(self._on_next)
        self._orientation_combo.currentIndexChanged.connect(self._on_control_changed)
        self._bad_slice_combo.currentIndexChanged.connect(self._on_bad_slice_changed)
        for spin in (
            self._dx_spin,
            self._dy_spin,
            self._angle_spin,
            self._radius_spin,
            self._angle_radius_spin,
        ):
            spin.valueChanged.connect(self._on_control_changed)
        self._run_auto_btn.clicked.connect(self._on_run_auto)
        self._run_local_btn.clicked.connect(self._on_run_local)
        self._reload_btn.clicked.connect(self._on_reload_preview)
        self._reset_btn.clicked.connect(self._on_reset_to_pc)
        self._accept_btn.clicked.connect(self._on_accept)
        self._save_btn.clicked.connect(self._on_save)
        self._toggle_masks_btn.clicked.connect(self._on_toggle_masks)

    @staticmethod
    def _make_double(QDoubleSpinBox, lo, hi, decimals, step, value=0.0):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        return spin

    # ====================================================================
    # Pair / state helpers
    # ====================================================================

    def _current_pairs(self):
        return build_adjacent_slice_pairs(
            self.entries, bad_slices=self.state.get("bad_slices", [])
        )

    def _n_pairs(self):
        return len(self._current_pairs())

    def _refresh_pair_count(self):
        n = self._n_pairs()
        max_idx = max(n - 1, 0)
        self._seeding = True
        self._pair_spin.setRange(0, max_idx)
        if self._pair_spin.value() > max_idx:
            self._pair_spin.setValue(max_idx)
        self._pair_count_label.setText(f"/ {max(n - 1, 0)}")
        self._seeding = False

    def _current_key(self):
        pairs = self._current_pairs()
        if not pairs:
            return None, None, None
        idx = min(self._pair_spin.value(), len(pairs) - 1)
        fixed_entry, moving_entry = pairs[idx]
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        return f"{fixed_slice:03d}_{moving_slice:03d}", fixed_entry, moving_entry

    def _load_record(self):
        key, fixed_entry, moving_entry = self._current_key()
        if key is None:
            return None, None, None, None
        return key, fixed_entry, moving_entry, self.state["pairs"].get(key, {})

    def _save_state(self):
        save_pairwise_registration_state(self.state_path, self.state)

    # ====================================================================
    # Record / transform builders (parity with ipywidgets widget)
    # ====================================================================

    def _record_from_result(
        self,
        fixed_slice,
        moving_slice,
        result,
        *,
        manual_angle,
        manual_shift,
        local_radius,
        angular_radius,
        status,
    ):
        return {
            "fixed_slice": fixed_slice,
            "moving_slice": moving_slice,
            "orientation": result["orientation"],
            "auto_orientation": result["orientation"],
            "angle_deg": float(result["angle_deg"]),
            "auto_angle_deg": float(result["angle_deg"]),
            "shift_xy": [
                float(result["shift_xy"][0]),
                float(result["shift_xy"][1]),
            ],
            "auto_shift_xy": [
                float(result["shift_xy"][0]),
                float(result["shift_xy"][1]),
            ],
            "manual_angle_deg": float(manual_angle) if manual_angle is not None else None,
            "manual_shift_xy": (
                [float(manual_shift[0]), float(manual_shift[1])]
                if manual_shift is not None
                else None
            ),
            "local_search_radius": (
                float(local_radius) if local_radius is not None else None
            ),
            "angle_radius_deg": float(angular_radius),
            "matrix": np.asarray(result["matrix"]).tolist(),
            "score": float(result["score"]),
            "status": status,
            "method": result["method"],
        }

    def _preview_record_from_controls(self, fixed_entry, moving_entry, saved_record):
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        fixed_image = self.images[self.index_by_slice[fixed_slice]]
        shape = fixed_image.shape
        base_record = dict(saved_record or {})
        orientation_name = self._orientation_combo.currentData()
        if orientation_name == "auto":
            orientation_name = base_record.get("orientation", "identity")
            if orientation_name == "auto":
                orientation_name = "identity"
        dx = self._dx_spin.value()
        dy = self._dy_spin.value()
        angle = self._angle_spin.value()
        matrix = (
            _translation_matrix(dx, dy)
            @ _rotation_matrix(angle, shape)
            @ _orientation_matrix(shape, orientation_name)
        )
        prior_status = base_record.get("status", "pending")
        next_status = "accepted" if prior_status == "accepted" else "pending"
        return {
            "fixed_slice": fixed_slice,
            "moving_slice": moving_slice,
            "orientation": orientation_name,
            "auto_orientation": base_record.get(
                "auto_orientation", base_record.get("orientation")
            ),
            "angle_deg": float(angle),
            "auto_angle_deg": base_record.get(
                "auto_angle_deg", base_record.get("angle_deg")
            ),
            "shift_xy": [float(dx), float(dy)],
            "auto_shift_xy": base_record.get(
                "auto_shift_xy", base_record.get("shift_xy")
            ),
            "manual_angle_deg": float(angle),
            "manual_shift_xy": [float(dx), float(dy)],
            "local_search_radius": base_record.get(
                "local_search_radius", self._radius_spin.value()
            ),
            "angle_radius_deg": base_record.get(
                "angle_radius_deg", self._angle_radius_spin.value()
            ),
            "matrix": np.asarray(matrix).tolist(),
            "score": base_record.get("score"),
            "status": next_status,
            "method": "widget_preview_affine",
            "error": base_record.get("error"),
        }

    def _estimate_from_controls(self, fixed_entry, moving_entry, *, use_local, force_auto):
        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        key = f"{fixed_slice:03d}_{moving_slice:03d}"
        fixed_image = self.images[self.index_by_slice[fixed_slice]]
        moving_image = self.images[self.index_by_slice[moving_slice]]
        fixed_mask = self.masks[self.index_by_slice[fixed_slice]]
        moving_mask = self.masks[self.index_by_slice[moving_slice]]
        existing = self.state.get("pairs", {}).get(key, {})
        if force_auto:
            forced_orientation = "auto"
            manual_shift = None
            manual_angle = None
            local_radius = None
        else:
            forced_orientation = self._orientation_combo.currentData()
            use_manual_seed = (
                forced_orientation != "auto" or self._controls_dirty
            )
            if use_manual_seed:
                manual_shift = [self._dx_spin.value(), self._dy_spin.value()]
                manual_angle = self._angle_spin.value()
            else:
                manual_shift = None
                manual_angle = None
            local_radius = self._radius_spin.value() if use_local else None
        angular_radius = self._angle_radius_spin.value()
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
        prior_status = existing.get("status", "pending")
        next_status = "accepted" if prior_status == "accepted" else "pending"
        return key, self._record_from_result(
            fixed_slice,
            moving_slice,
            result,
            manual_angle=manual_angle,
            manual_shift=manual_shift,
            local_radius=local_radius,
            angular_radius=angular_radius,
            status=existing.get("status", next_status),
        )

    def _commit_current_record(self, *, accept_record=False):
        key, fixed_entry, moving_entry, saved_record = self._load_record()
        if key is None:
            return None
        record = self._pending_record
        if record is None and self._controls_dirty:
            record = self._preview_record_from_controls(
                fixed_entry, moving_entry, saved_record
            )
        if record is None:
            record = dict(saved_record) if saved_record else None
        if record is None:
            _, record = self._estimate_from_controls(
                fixed_entry, moving_entry, use_local=False, force_auto=True
            )
        record = dict(record)
        if accept_record:
            record["status"] = "accepted"
        self.state["pairs"][key] = record
        self._save_state()
        self._pending_record = None
        self._controls_dirty = False
        return record

    def _bridge_pair_across(self, bad_id, current_bad_slices):
        active_slice_list = sorted(
            int(entry["slice_number"])
            for entry in self.entries
            if int(entry["slice_number"]) not in {int(s) for s in current_bad_slices}
            and int(entry["slice_number"]) != int(bad_id)
        )
        pre_exclusion = sorted(
            int(entry["slice_number"])
            for entry in self.entries
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
            self.entries[self.index_by_slice[higher_slice]],
            self.entries[self.index_by_slice[lower_slice]],
        )

    # ====================================================================
    # Display update
    # ====================================================================

    def _seed_controls_from_record(self, record, *, use_auto=False):
        self._seeding = True
        try:
            orientation_seed = (
                record.get("auto_orientation", record.get("orientation", "auto"))
                if use_auto
                else record.get("orientation", "auto")
            )
            valid = {opt[1] for opt in _ORIENTATION_OPTIONS}
            if orientation_seed not in valid:
                orientation_seed = "auto"
            idx = self._orientation_combo.findData(orientation_seed)
            if idx >= 0:
                self._orientation_combo.setCurrentIndex(idx)

            if use_auto and record.get("auto_shift_xy") is not None:
                self._dx_spin.setValue(float(record["auto_shift_xy"][0]))
                self._dy_spin.setValue(float(record["auto_shift_xy"][1]))
            elif record.get("manual_shift_xy") is not None:
                self._dx_spin.setValue(float(record["manual_shift_xy"][0]))
                self._dy_spin.setValue(float(record["manual_shift_xy"][1]))
            else:
                shift_xy = record.get("shift_xy", [0.0, 0.0])
                self._dx_spin.setValue(float(shift_xy[0]))
                self._dy_spin.setValue(float(shift_xy[1]))

            angle_seed = (
                record.get("auto_angle_deg")
                if use_auto and record.get("auto_angle_deg") is not None
                else record.get("manual_angle_deg")
            )
            if angle_seed is None:
                angle_seed = record.get("angle_deg", 0.0)
            self._angle_spin.setValue(float(angle_seed) if angle_seed is not None else 0.0)
            radius = record.get("local_search_radius")
            self._radius_spin.setValue(
                float(radius) if radius is not None else self.default_radius
            )
            ang_radius = record.get("angle_radius_deg")
            self._angle_radius_spin.setValue(
                float(ang_radius)
                if ang_radius is not None
                else self.default_angle_radius_deg
            )

            # Bad-slice dropdown reflects what's already in state["bad_slices"].
            key, fixed_entry, moving_entry, _ = self._load_record()
            bad = {int(s) for s in self.state.get("bad_slices", [])}
            if fixed_entry is not None and int(fixed_entry["slice_number"]) in bad:
                self._set_combo_value(self._bad_slice_combo, "fixed")
            elif moving_entry is not None and int(moving_entry["slice_number"]) in bad:
                self._set_combo_value(self._bad_slice_combo, "moving")
            else:
                self._set_combo_value(self._bad_slice_combo, "none")
        finally:
            self._seeding = False
            self._controls_dirty = False
            self._pending_record = None

    @staticmethod
    def _set_combo_value(combo, value):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _update_pair_label(self, fixed_entry, moving_entry):
        bad = {int(s) for s in self.state.get("bad_slices", [])}
        f = int(fixed_entry["slice_number"])
        m = int(moving_entry["slice_number"])
        f_str = f"{f:03d}" + (" (bad slice)" if f in bad else "")
        m_str = f"{m:03d}" + (" (bad slice)" if m in bad else "")
        self._pair_label.setText(
            f"<b>Slice pair:</b> fixed {f_str} &larr; moving {m_str}"
        )

    def _update_status_label(self, message=None):
        record = self._pending_record
        if record is None:
            _, _, _, record = self._load_record()
        if record is None:
            self._status_label.setText(message or "")
            self._transform_label.setText("<b>Preview transform:</b> none")
            return
        status = record.get("status", "pending") if record else "pending"
        if self._pending_record is not None:
            status = f"{status}; preview not yet saved"
        elif self._controls_dirty:
            status = f"{status}; controls changed, press Reload preview"
        score = record.get("score") if record else None
        if score is not None and np.isfinite(float(score)):
            status_text = f"<b>Status:</b> {status} &nbsp; <b>Score:</b> {float(score):.4f}"
        else:
            status_text = f"<b>Status:</b> {status}"
        if message:
            status_text += f" &nbsp; <i>{message}</i>"
        self._status_label.setText(status_text)
        if record.get("matrix") is None:
            self._transform_label.setText("<b>Preview transform:</b> none")
        else:
            shift_xy = record.get("shift_xy", [0.0, 0.0])
            self._transform_label.setText(
                "<b>Preview transform:</b> "
                f"orientation={record.get('orientation', 'auto')} &nbsp; "
                f"dx={float(shift_xy[0]):.2f} &nbsp; "
                f"dy={float(shift_xy[1]):.2f} &nbsp; "
                f"angle={float(record.get('angle_deg', 0.0)):.2f} deg &nbsp; "
                f"method={record.get('method', 'unknown')}"
            )

    def _push_layers(self, fixed_image, warped_image, fixed_mask, warped_mask):
        f_levels = _make_pyramid(
            fixed_image.astype(np.float32, copy=False),
            n_levels=self.pyramid_levels,
            reduce="mean",
        )
        w_levels = _make_pyramid(
            warped_image.astype(np.float32, copy=False),
            n_levels=self.pyramid_levels,
            reduce="mean",
        )
        fm_levels = _make_pyramid(
            fixed_mask.astype(np.uint8, copy=False),
            n_levels=self.pyramid_levels,
            reduce="max",
        )
        wm_levels = _make_pyramid(
            warped_mask.astype(np.uint8, copy=False),
            n_levels=self.pyramid_levels,
            reduce="max",
        )
        # Drop down to single-level if the pyramid couldn't be built (very
        # small images). napari requires len(data) >= 2 for multiscale.
        self._fixed_layer.multiscale = len(f_levels) > 1
        self._fixed_layer.data = f_levels if len(f_levels) > 1 else f_levels[0]
        self._moving_layer.multiscale = len(w_levels) > 1
        self._moving_layer.data = w_levels if len(w_levels) > 1 else w_levels[0]
        self._fixed_mask_layer.multiscale = len(fm_levels) > 1
        self._fixed_mask_layer.data = (
            fm_levels if len(fm_levels) > 1 else fm_levels[0]
        )
        self._moving_mask_layer.multiscale = len(wm_levels) > 1
        self._moving_mask_layer.data = (
            wm_levels if len(wm_levels) > 1 else wm_levels[0]
        )

    def _render_current(self, *, seed_controls=False):
        key, fixed_entry, moving_entry, record = self._load_record()
        if key is None:
            self._update_status_label("No active pairs.")
            return
        if seed_controls and record:
            self._seed_controls_from_record(record)
        display_record = self._pending_record or record
        self._update_pair_label(fixed_entry, moving_entry)
        self._update_status_label()

        fixed_slice = int(fixed_entry["slice_number"])
        moving_slice = int(moving_entry["slice_number"])
        fixed_image = self.images[self.index_by_slice[fixed_slice]]
        fixed_mask = self.masks[self.index_by_slice[fixed_slice]].astype(bool)
        moving_image = self.images[self.index_by_slice[moving_slice]].astype(np.float32)
        moving_mask = self.masks[self.index_by_slice[moving_slice]].astype(np.float32)

        if display_record and display_record.get("matrix") is not None:
            matrix = np.asarray(display_record["matrix"], dtype=float)
            warped = warp_with_affine(
                moving_image, matrix, output_shape=fixed_image.shape, order=1
            )
            warped_mask = (
                warp_with_affine(
                    moving_mask, matrix, output_shape=fixed_image.shape, order=0
                )
                > 0.5
            )
        else:
            warped = moving_image
            warped_mask = moving_mask > 0.5

        self._push_layers(fixed_image, warped, fixed_mask, warped_mask)
        if display_record and display_record.get("error"):
            self._update_status_label(f"last error: {display_record['error']}")

    # ====================================================================
    # Slot handlers
    # ====================================================================

    def _set_busy(self, busy, *, message=""):
        self._busy = bool(busy)
        for btn in self._action_buttons:
            btn.setEnabled(not busy)
        if message:
            self._update_status_label(message)
        # Force Qt to repaint so the user sees the disabled state before we
        # drop into a long-running call (warp / phase correlation).
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

    def _on_pair_spin_changed(self, value):
        if self._seeding:
            return
        self._on_pair_changed(value, seed_controls=True)

    def _on_pair_changed(self, value, *, seed_controls):
        self._set_busy(True, message="Loading pair...")
        try:
            self._render_current(seed_controls=seed_controls)
        finally:
            self._set_busy(False)

    def _on_prev(self):
        self._pair_spin.setValue(max(self._pair_spin.value() - 1, 0))

    def _on_next(self):
        self._pair_spin.setValue(
            min(self._pair_spin.value() + 1, self._pair_spin.maximum())
        )

    def _on_control_changed(self, *_):
        if self._seeding:
            return
        self._controls_dirty = True
        self._pending_record = None
        self._update_status_label()

    def _on_bad_slice_changed(self, *_):
        if self._seeding:
            return
        # No image refresh needed — actual exclusion happens on Accept.
        self._update_status_label()

    def _on_run_auto(self):
        key, fixed_entry, moving_entry, _ = self._load_record()
        if key is None:
            return
        self._set_busy(True, message="Running automatic registration...")
        try:
            key, record = self._estimate_from_controls(
                fixed_entry, moving_entry, use_local=False, force_auto=True
            )
            self.state["pairs"][key] = record
            self._save_state()
            self._pending_record = None
            self._controls_dirty = False
            self._render_current(seed_controls=True)
        except Exception as exc:
            self._update_status_label(f"Run auto failed: {exc}")
        finally:
            self._set_busy(False)

    def _on_run_local(self):
        key, fixed_entry, moving_entry, _ = self._load_record()
        if key is None:
            return
        self._set_busy(True, message="Running local refine...")
        try:
            key, record = self._estimate_from_controls(
                fixed_entry, moving_entry, use_local=True, force_auto=False
            )
            self.state["pairs"][key] = record
            self._save_state()
            self._pending_record = None
            self._controls_dirty = False
            self._render_current(seed_controls=True)
        except Exception as exc:
            self._update_status_label(f"Local refine failed: {exc}")
        finally:
            self._set_busy(False)

    def _on_reload_preview(self):
        key, fixed_entry, moving_entry, saved_record = self._load_record()
        if key is None or not saved_record:
            return
        self._set_busy(True, message="Reapplying transform preview...")
        try:
            preview = self._preview_record_from_controls(
                fixed_entry, moving_entry, saved_record
            )
            self._pending_record = preview
            self._controls_dirty = False
            self._render_current(seed_controls=False)
        finally:
            self._set_busy(False)

    def _on_reset_to_pc(self):
        key, fixed_entry, moving_entry, saved_record = self._load_record()
        if key is None or not saved_record:
            return
        self._set_busy(True, message="Resetting to automatic fit...")
        try:
            self._seed_controls_from_record(saved_record, use_auto=True)
            preview = self._preview_record_from_controls(
                fixed_entry, moving_entry, saved_record
            )
            self._pending_record = preview
            self._controls_dirty = False
            self._render_current(seed_controls=False)
        finally:
            self._set_busy(False)

    def _on_save(self):
        self._set_busy(True, message="Saving preview...")
        try:
            self._commit_current_record(accept_record=False)
            self._render_current(seed_controls=True)
        finally:
            self._set_busy(False)

    def _on_accept(self):
        key, _, _, _ = self._load_record()
        if key is None:
            return
        self._set_busy(True, message="Saving accepted pair...")
        try:
            committed = self._commit_current_record(accept_record=True)
            if committed is None:
                return
            bad_value = self._bad_slice_combo.currentData()
            bridge_pair = None
            if bad_value != "none":
                bad_id = (
                    committed["fixed_slice"]
                    if bad_value == "fixed"
                    else committed["moving_slice"]
                )
                bridge_pair = self._bridge_pair_across(
                    bad_id, self.state.get("bad_slices", [])
                )
                bad = {int(v) for v in self.state.get("bad_slices", [])}
                bad.add(int(bad_id))
                self.state["bad_slices"] = sorted(bad)
            self._save_state()
            if bridge_pair is not None:
                self._update_status_label(
                    f"Auto-registering bridge pair "
                    f"{int(bridge_pair[0]['slice_number']):03d}_"
                    f"{int(bridge_pair[1]['slice_number']):03d}..."
                )
                from qtpy.QtWidgets import QApplication

                QApplication.processEvents()
                _, bridge_record = self._estimate_from_controls(
                    bridge_pair[0],
                    bridge_pair[1],
                    use_local=False,
                    force_auto=True,
                )
                bridge_key = (
                    f"{int(bridge_pair[0]['slice_number']):03d}_"
                    f"{int(bridge_pair[1]['slice_number']):03d}"
                )
                self.state["pairs"][bridge_key] = bridge_record
                self._save_state()
            if bad_value != "none":
                try:
                    write_bad_slices_to_chamber_ops(
                        state_path=self.state_path, warn_missing=False
                    )
                except Exception as exc:
                    self._update_status_label(
                        f"Failed to mirror bad_slices into chamber ops.yml: {exc}"
                    )
            self._refresh_pair_count()
            self._render_current(seed_controls=True)
        finally:
            self._set_busy(False)

    def _on_toggle_masks(self):
        new_visible = not self._fixed_mask_layer.visible
        self._fixed_mask_layer.visible = new_visible
        self._moving_mask_layer.visible = new_visible
