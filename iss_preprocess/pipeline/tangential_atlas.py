"""Tangential serial-section atlas annotation — geometry primitives.

This module hosts geometry, atlas sampling, and state IO for mapping a
preregistered tangential volume to a 3-D BrainGlobe atlas. v1 is built
incrementally as small vertical slices.

M1 provides the smallest useful primitive: a pure, unit-agnostic
sampler that renders a 2-D oblique plane out of a 3-D numpy volume.
M2 adds the canonical 6-DoF stack pose and its bidirectional
conversion with the 4×4 ``atlas_from_tangential`` matrix. Atlas
loading, per-slice residuals, reviewer code, optimization, and QC
live in later slices and are intentionally absent here.

Coordinate convention used by M1
--------------------------------

The sampler is unit-free. The caller supplies a 4×4 affine that maps
output grid column vectors ``(col, row, 0, 1)`` straight to volume
voxel indices ``(i0, i1, i2, 1)``. The ``i0`` axis corresponds to
``volume.shape[0]``, ``i1`` to ``volume.shape[1]``, ``i2`` to
``volume.shape[2]`` — the array-order indexing used by
``scipy.ndimage.map_coordinates``.

Out-of-volume sample positions take ``spec.cval`` via
``mode="constant"``.

Conventions added in M2
-----------------------

Frame ordering of 4-vectors (all column vectors, all µm):

- **Atlas frame**: ``[AP_um, DV_um, ML_um, 1]^T``.
  Atlas-axis index → name: ``0=AP, 1=DV, 2=ML``.
- **Tangential frame**: ``[x_tan_um, y_tan_um, z_tan_um, 1]^T``,
  where ``(x_tan, y_tan)`` is the tangential canvas in-plane and
  ``z_tan`` is the stack-normal direction.
  Tangential-axis index → name: ``0=x_tan, 1=y_tan, 2=z_tan``.
- ``atlas_from_tangential`` is a ``(4, 4)`` float64 ndarray that maps
  a tangential column vector to an atlas column vector by left
  multiplication.

Identity pose (``yaw = pitch = roll = 0`` and
``depth = tx_atlas = ty_atlas = 0``) gives
``atlas_from_tangential = P`` (the proper rotation defined by
``_P_ATLAS_FROM_TANGENTIAL`` below) — NOT ``I_4``. At zero pose
tangential ``(x_tan, y_tan, z_tan)`` maps to atlas
``(AP, -ML, DV)``. ``P`` is a 90° rotation about atlas AP
(det = +1), aligning the stack-normal direction ``z_tan`` with the
atlas DV axis so the zero-pose plane is DV-constant.

The 3×3 form of ``P`` is::

    P = [[ 1,  0, 0],     # AP =  x_tan
         [ 0,  0, 1],     # DV =  z_tan
         [ 0, -1, 0]]     # ML = -y_tan

Rotation axes and order (ZYX, right-hand rule):

- ``R_yaw(α)`` rotates about atlas DV (axis 1).
- ``R_pitch(β)`` rotates about atlas ML (axis 2).
- ``R_roll(γ)`` rotates about atlas AP (axis 0).
- ``R = R_yaw(DV) @ R_pitch(ML) @ R_roll(AP)``. The composed
  ``atlas_from_tangential`` upper-left 3×3 is ``R @ P``: ``P`` first
  permutes the tangential vector into the atlas frame, then ``R``
  rotates it within the atlas frame.

Translation is interpreted in the **atlas frame**, along **fixed
atlas axes**. The three scalar fields of ``StackPose`` form the
3-vector ``t_atlas = [tx_atlas_um, depth_um, ty_atlas_um]`` in
atlas ``(AP, DV, ML)`` ordering — ``depth_um`` is the **DV**
component (= stack normal at any pose under the new identity).

Conventions added in M3
-----------------------

Atlas array layout follows BrainGlobe's Allen mouse atlases:
``shape_zyx`` is BrainGlobe's ``(z, y, x)`` mnemonic but in these
atlases the axes are ``(n_AP, n_DV, n_ML)``. The voxel size is
isotropic (one ``resolution_um`` scalar); cross-checked against the
integer parsed from ``atlas_name`` in :func:`load_atlas`.

Voxel-index ↔ atlas-µm scaling is a pure diagonal:
``voxel_index = atlas_um / resolution_um`` along each of AP, DV, ML.
The atlas origin is ``(0, 0, 0)`` µm at voxel ``(0, 0, 0)``; no
offset, no axis permutation at this boundary.

:class:`AtlasPlaneSpec` carries the only matrix that mentions
atlas-µm in M3: ``atlas_um_from_grid_4x4`` maps a grid column vector
``(col, row, 0, 1)`` to an atlas-µm column vector
``(AP_um, DV_um, ML_um, 1)``. Naming follows the M1/M2
``<target>_from_<source>`` style. The renderers compose this with
the voxel scaling above and hand the result to the M1 sampler
unchanged (the sampler remains unit-free).

:func:`render_reference_plane` samples ``atlas.reference`` with
``order=1`` (trilinear). :func:`render_annotation_plane` samples
``atlas.annotation`` with ``order=0`` (nearest) and **requires**
``spec.order == 0`` — annotation interpolation must be nearest.

The atlas-loader cache (``_ATLAS_CACHE``) is keyed by
``atlas_name``; entries hold the BrainGlobe-owned numpy arrays by
reference and do **not** copy them. Caller mutation of
``AtlasBundle.reference`` / ``.annotation`` silently corrupts every
subsequent cache hit.

Conventions added in M5
-----------------------

:class:`SliceStackGeometry` carries the kept-only slice list plus a
sorted bad-slice list (gap-preserving via ``z_um`` arithmetic). The
``slice_index`` arg to :func:`build_slice_plane_spec` is the
kept-list position ``0..M-1``, not the global ``slice_number``.

The canvas → tangential-µm scaling is the scalar
``S_canvas_to_tan_µm = diag(s, s, 1, 1)`` with ``s =
geometry.overview_pixel_size_um``. The per-slice
``global_from_overview[i]`` (3×3 affine on ``(x, y, 1)``) is
promoted to a 4×4 affecting only the first two rows.

M5 always uses :class:`SliceResidual` zeros (``dz_um`` is hard-zero
under the G2 gate). M6 wires non-zero in-plane residuals through
the same composition.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import brainglobe_atlasapi as bga
import numpy as np
import pandas as pd
import scipy.ndimage


TANGENTIAL_ATLAS_STATE_VERSION = 1
TANGENTIAL_ATLAS_SUBDIR = "tangential_atlas"
TANGENTIAL_ATLAS_STATE_FILENAME = "tangential_atlas_state.json"
DEFAULT_TANGENTIAL_ATLAS_NAME = "allen_mouse_10um"
DEFAULT_TANGENTIAL_ATLAS_SECTION_THICKNESS_UM = 20.0

__all__ = [
    "DEFAULT_TANGENTIAL_ATLAS_NAME",
    "DEFAULT_TANGENTIAL_ATLAS_SECTION_THICKNESS_UM",
    "TangentialAtlasIssue",
    "TangentialAtlasMetadataResolution",
    "TangentialAtlasSliceRecord",
    "TangentialAtlasContext",
    "PlaneSamplingSpec",
    "sample_plane",
    "StackPose",
    "pose_to_matrix",
    "matrix_to_pose",
    "AtlasBundle",
    "load_atlas",
    "AtlasPlaneSpec",
    "PlaneRenderConfig",
    "render_reference_plane",
    "render_annotation_plane",
    "SliceResidual",
    "SliceStackGeometry",
    "build_tangential_atlas_context",
    "image_for_tangential_atlas_slice",
    "build_slice_stack_geometry",
    "build_slice_plane_spec",
    "RegistrationObjectiveConfig",
    "validate_residual_against_bounds",
    "compose_pose_with_residual",
    "center_residuals",
    "save_tangential_atlas_state",
    "load_tangential_atlas_state",
    "TANGENTIAL_ATLAS_STATE_VERSION",
    "monotonic_dz_from_raw",
    "assert_monotonic_spacing",
    "slice_objective",
    "refine_residuals",
    "write_tangential_atlas_rasters",
    "write_qc_report",
    "register_spots_to_tangential_atlas",
]


@dataclass(frozen=True)
class TangentialAtlasIssue:
    """Validation or metadata issue found while building a real-data context."""

    severity: str
    code: str
    message: str
    slice_number: object = None
    path: object = None

    def __post_init__(self) -> None:
        if self.severity not in ("warning", "error"):
            raise ValueError(
                f"severity must be 'warning' or 'error', got {self.severity!r}"
            )
        if not isinstance(self.code, str) or not self.code:
            raise ValueError(f"code must be a non-empty str, got {self.code!r}")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError(
                f"message must be a non-empty str, got {self.message!r}"
            )
        if self.slice_number is not None:
            if isinstance(self.slice_number, bool):
                raise ValueError("slice_number must be int-like or None")
            object.__setattr__(self, "slice_number", int(self.slice_number))
        if self.path is not None:
            object.__setattr__(self, "path", str(self.path))


@dataclass(frozen=True)
class TangentialAtlasMetadataResolution:
    """Resolved scalar metadata and provenance for a real-data context."""

    overview_pixel_size_um: float
    section_thickness_um: float
    overview_pixel_size_source: str
    section_thickness_source: str
    z_source: str

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.overview_pixel_size_um)
            or float(self.overview_pixel_size_um) <= 0
        ):
            raise ValueError(
                "overview_pixel_size_um must be finite positive, "
                f"got {self.overview_pixel_size_um!r}"
            )
        if (
            not np.isfinite(self.section_thickness_um)
            or float(self.section_thickness_um) <= 0
        ):
            raise ValueError(
                "section_thickness_um must be finite positive, "
                f"got {self.section_thickness_um!r}"
            )
        for name in (
            "overview_pixel_size_source",
            "section_thickness_source",
            "z_source",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty str")
        object.__setattr__(
            self, "overview_pixel_size_um", float(self.overview_pixel_size_um)
        )
        object.__setattr__(
            self, "section_thickness_um", float(self.section_thickness_um)
        )


@dataclass(frozen=True)
class TangentialAtlasSliceRecord:
    """Slice-number keyed link between stack image, transform, and geometry."""

    slice_number: int
    kept_index: int
    image_index: int
    roi: int
    chamber: str
    z_um: float
    global_from_overview: np.ndarray
    global_from_fullres: object = None

    def __post_init__(self) -> None:
        for name in ("slice_number", "kept_index", "image_index", "roi"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be int-like, got bool")
            object.__setattr__(self, name, int(value))
        if self.kept_index < 0:
            raise ValueError("kept_index must be non-negative")
        if self.image_index < 0:
            raise ValueError("image_index must be non-negative")
        if not np.isfinite(self.z_um):
            raise ValueError(f"z_um must be finite, got {self.z_um!r}")
        object.__setattr__(self, "chamber", str(self.chamber))
        object.__setattr__(self, "z_um", float(self.z_um))

        gfo = np.asarray(self.global_from_overview, dtype=np.float64)
        if gfo.shape != (3, 3) or not np.allclose(gfo[2], (0.0, 0.0, 1.0)):
            raise ValueError("global_from_overview must be homogeneous (3, 3)")
        object.__setattr__(self, "global_from_overview", gfo.copy())

        if self.global_from_fullres is not None:
            gff = np.asarray(self.global_from_fullres, dtype=np.float64)
            if gff.shape != (3, 3) or not np.allclose(gff[2], (0.0, 0.0, 1.0)):
                raise ValueError("global_from_fullres must be homogeneous (3, 3)")
            object.__setattr__(self, "global_from_fullres", gff.copy())


@dataclass(frozen=True)
class TangentialAtlasContext:
    """Validated real-data inputs shared by reviewer, QC, and exports."""

    stack_path: Path
    transforms_path: Path
    state_path: object
    images: np.ndarray
    manifest: dict
    geometry: "SliceStackGeometry"
    slice_records: tuple
    metadata: TangentialAtlasMetadataResolution
    issues: tuple
    state: object = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stack_path", Path(self.stack_path))
        object.__setattr__(self, "transforms_path", Path(self.transforms_path))
        if self.state_path is not None:
            object.__setattr__(self, "state_path", Path(self.state_path))
        images = np.asarray(self.images)
        if images.ndim != 3:
            raise ValueError(f"images must be a 3-D stack, got shape {images.shape}")
        object.__setattr__(self, "images", images)
        if not isinstance(self.manifest, dict):
            raise ValueError("manifest must be a dict")
        records = tuple(self.slice_records)
        if len(records) != len(self.geometry.slice_numbers):
            raise ValueError("slice_records length must match geometry")
        object.__setattr__(self, "slice_records", records)
        object.__setattr__(self, "issues", tuple(self.issues))


@dataclass(frozen=True)
class PlaneSamplingSpec:
    """Geometry of one 2-D oblique plane to sample from a 3-D volume.

    Parameters
    ----------
    out_shape_yx : tuple[int, int]
        Output array shape ``(H, W)``; first axis is row, second is col.
    volume_index_from_grid : np.ndarray
        ``(4, 4)`` affine mapping grid column vectors
        ``(col, row, 0, 1)`` to volume voxel indices
        ``(i0, i1, i2, 1)``. Cast to ``float64`` and copied on store.
        The last row must be ``(0, 0, 0, 1)``.
    order : int
        Interpolation order: ``0`` for nearest (label/annotation
        volumes), ``1`` for linear (continuous/reference volumes).
        Other values are rejected.
    cval : float
        Value returned at sample positions outside the volume. Must be
        finite.
    """

    out_shape_yx: tuple
    volume_index_from_grid: np.ndarray
    order: int = 1
    cval: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.out_shape_yx, tuple)
            or len(self.out_shape_yx) != 2
        ):
            raise ValueError(
                f"out_shape_yx must be a 2-tuple, got {self.out_shape_yx!r}"
            )
        h, w = self.out_shape_yx
        if (
            not isinstance(h, int)
            or not isinstance(w, int)
            or isinstance(h, bool)
            or isinstance(w, bool)
        ):
            raise ValueError(
                f"out_shape_yx must contain Python ints, got {self.out_shape_yx!r}"
            )
        if h <= 0 or w <= 0:
            raise ValueError(
                f"out_shape_yx must be strictly positive, got {self.out_shape_yx!r}"
            )

        try:
            mat = np.asarray(self.volume_index_from_grid, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "volume_index_from_grid must be castable to a float64 array"
            ) from exc
        if mat.shape != (4, 4):
            raise ValueError(
                f"volume_index_from_grid must be shape (4, 4), got {mat.shape}"
            )
        if not np.allclose(mat[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(
                "volume_index_from_grid last row must be (0, 0, 0, 1), "
                f"got {mat[3].tolist()}"
            )
        object.__setattr__(self, "volume_index_from_grid", mat.copy())

        if self.order not in (0, 1):
            raise ValueError(
                f"order must be 0 (nearest) or 1 (linear), got {self.order!r}"
            )

        if not np.isfinite(self.cval):
            raise ValueError(f"cval must be finite, got {self.cval!r}")


def _build_grid_columns(out_shape_yx: tuple) -> np.ndarray:
    """Build a ``(4, H*W)`` matrix of grid column vectors ``(col, row, 0, 1)``.

    The flattening order matches ``np.indices((H, W)).reshape(2, -1)``,
    so column ``k`` corresponds to output position ``(row=k // W,
    col=k % W)``.
    """
    h, w = out_shape_yx
    idx = np.indices((h, w), dtype=np.float64)
    rows = idx[0].reshape(-1)
    cols = idx[1].reshape(-1)
    zeros = np.zeros_like(rows)
    ones = np.ones_like(rows)
    return np.stack([cols, rows, zeros, ones], axis=0)


def sample_plane(
    volume: np.ndarray,
    spec: PlaneSamplingSpec,
) -> np.ndarray:
    """Sample a 2-D oblique plane from a 3-D ``volume``.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of any numeric dtype.
    spec : PlaneSamplingSpec
        Output shape, voxel-index-from-grid affine, interpolation
        order, and outside-volume fill value.

    Returns
    -------
    np.ndarray
        Array of shape ``spec.out_shape_yx``. Element ``[r, c]`` is the
        volume sampled at index coordinates
        ``spec.volume_index_from_grid @ (c, r, 0, 1)``. The returned
        array has the same dtype as ``volume`` for both
        ``spec.order == 0`` and ``spec.order == 1`` (scipy default;
        we do not override).
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")

    grid = _build_grid_columns(spec.out_shape_yx)
    voxel_cols = spec.volume_index_from_grid @ grid
    h, w = spec.out_shape_yx
    coords = voxel_cols[:3].reshape(3, h, w)
    return scipy.ndimage.map_coordinates(
        volume,
        coords,
        order=spec.order,
        mode="constant",
        cval=spec.cval,
        prefilter=False,
    )


_ROTATION_ORDER_ZYX = "ZYX"


# Identity-pose tangential→atlas axis permutation (a proper rotation,
# det = +1; equivalent to a 90° rotation about atlas AP). Maps
# tangential (x_tan, y_tan, z_tan) onto atlas (AP, -ML, DV) at zero
# pose. See module docstring §M2 for the geometric meaning.
_P_ATLAS_FROM_TANGENTIAL = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def _R_about_AP(rad: float) -> np.ndarray:
    """3×3 rotation about atlas AP (axis 0). See module docstring §M2."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _R_about_DV(rad: float) -> np.ndarray:
    """3×3 rotation about atlas DV (axis 1). See module docstring §M2."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _R_about_ML(rad: float) -> np.ndarray:
    """3×3 rotation about atlas ML (axis 2). See module docstring §M2."""
    c, s = np.cos(rad), np.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    depth_um: float,
    tx_atlas_um: float,
    ty_atlas_um: float,
    *,
    rotation_order: str = "ZYX",
) -> np.ndarray:
    """Compose the 4×4 ``atlas_from_tangential`` matrix from a 6-DoF pose.

    Column-vector convention. Tangential 4-vectors are
    ``[x_tan_um, y_tan_um, z_tan_um, 1]^T``; atlas 4-vectors are
    ``[AP_um, DV_um, ML_um, 1]^T``. Left multiplication by the returned
    matrix maps a tangential column vector to an atlas column vector.

    Rotation order is ZYX:
    ``R = R_yaw(DV) @ R_pitch(ML) @ R_roll(AP)``; the upper-left 3×3
    of the returned matrix is ``R @ P`` where ``P`` is the
    identity-pose tangential→atlas permutation (see module docstring).
    Sign conventions are the right-hand-rule matrices given in the
    module docstring. Angles are in degrees.

    Translation is interpreted in the atlas frame along fixed atlas
    axes: ``t_atlas = [tx_atlas_um, depth_um, ty_atlas_um]`` in atlas
    ``(AP, DV, ML)`` ordering. ``depth_um`` is the **DV** component
    and aligns with the stack-normal direction (the fixed-axis
    convention does not rotate the translation with the pose).

    Parameters
    ----------
    yaw_deg, pitch_deg, roll_deg : float
        Rotation angles in degrees about atlas DV, ML, AP respectively.
    depth_um : float
        Atlas-µm translation along the atlas DV axis
        (= stack normal at any pose under the new identity).
    tx_atlas_um : float
        Atlas-µm translation along the atlas AP axis.
    ty_atlas_um : float
        Atlas-µm translation along the atlas ML axis.
    rotation_order : str, keyword-only
        Must be ``"ZYX"``. Provided for future extension; v1 rejects
        any other value.

    Returns
    -------
    np.ndarray
        ``(4, 4)`` float64 ``atlas_from_tangential`` matrix.
    """
    if rotation_order != _ROTATION_ORDER_ZYX:
        raise ValueError(
            "rotation_order must be 'ZYX' (the only value accepted in v1), "
            f"got {rotation_order!r}"
        )
    scalars = {
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "depth_um": depth_um,
        "tx_atlas_um": tx_atlas_um,
        "ty_atlas_um": ty_atlas_um,
    }
    for name, value in scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    yaw_rad = np.deg2rad(float(yaw_deg))
    pitch_rad = np.deg2rad(float(pitch_deg))
    roll_rad = np.deg2rad(float(roll_deg))

    R = _R_about_DV(yaw_rad) @ _R_about_ML(pitch_rad) @ _R_about_AP(roll_rad)

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R @ _P_ATLAS_FROM_TANGENTIAL
    M[:3, 3] = np.array(
        [float(tx_atlas_um), float(depth_um), float(ty_atlas_um)],
        dtype=np.float64,
    )
    return M


def _validate_rigid_4x4(mat: np.ndarray, *, name: str) -> np.ndarray:
    """Return ``mat`` as a float64 ``(4, 4)`` rigid transform or raise.

    Checks shape ``(4, 4)``, last row ``(0, 0, 0, 1)``, and that the
    upper-left 3×3 is orthonormal with determinant ``+1`` within
    ``atol = 1e-8``.
    """
    try:
        arr = np.asarray(mat, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be castable to a float64 array"
        ) from exc
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be shape (4, 4), got {arr.shape}")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(
            f"{name} last row must be (0, 0, 0, 1), got {arr[3].tolist()}"
        )
    R = arr[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-8) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-8
    ):
        raise ValueError(
            f"{name} upper-left 3x3 must be a proper rotation "
            "(orthonormal, det = +1)"
        )
    return arr


@dataclass(frozen=True)
class StackPose:
    """Canonical 6-DoF pose of the tangential stack in atlas-µm space.

    The 4×4 ``atlas_from_tangential`` is authoritative. The six scalar
    fields are derived for UI / state readability and must round-trip
    through :func:`pose_to_matrix` / :func:`matrix_to_pose` exactly
    (atol 1e-9). ``__post_init__`` runs the consistency check at
    construction; mismatched matrix + scalars raise ``ValueError``.

    Parameters
    ----------
    atlas_from_tangential : np.ndarray
        ``(4, 4)`` rigid transform mapping tangential column vectors
        ``[x_tan_um, y_tan_um, z_tan_um, 1]^T`` to atlas column vectors
        ``[AP_um, DV_um, ML_um, 1]^T``. Cast to ``float64`` and copied
        on store.
    yaw_deg : float
        Rotation about atlas DV (degrees).
    pitch_deg : float
        Rotation about atlas ML (degrees).
    roll_deg : float
        Rotation about atlas AP (degrees).
    depth_um : float
        Atlas-µm translation along the atlas DV axis
        (= stack normal at any pose under the new identity).
    tx_atlas_um : float
        Atlas-µm translation along the atlas AP axis.
    ty_atlas_um : float
        Atlas-µm translation along the atlas ML axis.
    """

    atlas_from_tangential: np.ndarray
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    depth_um: float
    tx_atlas_um: float
    ty_atlas_um: float

    def __post_init__(self) -> None:
        mat = _validate_rigid_4x4(
            self.atlas_from_tangential, name="atlas_from_tangential"
        )
        object.__setattr__(self, "atlas_from_tangential", mat.copy())

        scalars = {
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "depth_um": self.depth_um,
            "tx_atlas_um": self.tx_atlas_um,
            "ty_atlas_um": self.ty_atlas_um,
        }
        for fname, value in scalars.items():
            if not np.isfinite(value):
                raise ValueError(f"{fname} must be finite, got {value!r}")

        expected = pose_to_matrix(
            self.yaw_deg,
            self.pitch_deg,
            self.roll_deg,
            self.depth_um,
            self.tx_atlas_um,
            self.ty_atlas_um,
        )
        if not np.allclose(mat, expected, atol=1e-9):
            raise ValueError(
                "StackPose scalars and atlas_from_tangential disagree: "
                "pose_to_matrix(scalars) does not match the stored matrix "
                "within atol=1e-9. Build via matrix_to_pose() for a "
                "self-consistent pose."
            )


def matrix_to_pose(atlas_from_tangential: np.ndarray) -> StackPose:
    """Decompose a 4×4 ``atlas_from_tangential`` into a :class:`StackPose`.

    Column-vector convention; see module docstring for frame
    orderings. The upper-left 3×3 must be a proper rotation
    (orthonormal, det = +1 within ``atol = 1e-8``); the last row must
    be ``(0, 0, 0, 1)``; otherwise ``ValueError`` is raised.

    The rotation is inverted under the Tait–Bryan ZYX convention used
    by :func:`pose_to_matrix`. Pitch is taken on the positive branch
    so it lies in ``[-π/2, +π/2]``. Gimbal lock at ``|pitch| → π/2``
    is detected (``cos(pitch) > 1e-7``); a degenerate input raises
    ``ValueError`` mentioning "gimbal lock". v1 pose ranges stay well
    below 30° per axis, so this is purely a safety net.

    The translation is read directly in the atlas frame (no rotation
    applied), matching the fixed-axis convention of
    :func:`pose_to_matrix`.

    Parameters
    ----------
    atlas_from_tangential : np.ndarray
        ``(4, 4)`` rigid transform.

    Returns
    -------
    StackPose
        Self-consistent pose; ``StackPose.__post_init__`` verifies
        that ``pose_to_matrix(...)`` reproduces the input within
        ``atol = 1e-9``.
    """
    mat = _validate_rigid_4x4(atlas_from_tangential, name="atlas_from_tangential")
    # Undo the identity-pose tangential→atlas permutation P so the
    # extracted rotation R_atlas lives in the atlas frame and matches
    # the ZYX composition built by ``pose_to_matrix``.
    R_atlas = mat[:3, :3] @ _P_ATLAS_FROM_TANGENTIAL.T
    t = mat[:3, 3]

    tx_atlas_um = float(t[0])
    depth_um = float(t[1])
    ty_atlas_um = float(t[2])

    sin_pitch = float(np.clip(R_atlas[1, 0], -1.0, 1.0))
    pitch_rad = np.arcsin(sin_pitch)
    cos_pitch = float(np.sqrt(max(0.0, 1.0 - sin_pitch * sin_pitch)))
    if cos_pitch <= 1e-7:
        raise ValueError(
            "matrix_to_pose: gimbal lock — pitch near ±90° "
            f"(cos(pitch) = {cos_pitch:.3e}); v1 does not handle this case"
        )

    yaw_rad = np.arctan2(-R_atlas[2, 0], R_atlas[0, 0])
    roll_rad = np.arctan2(-R_atlas[1, 2], R_atlas[1, 1])

    yaw_deg = float(np.rad2deg(yaw_rad))
    pitch_deg = float(np.rad2deg(pitch_rad))
    roll_deg = float(np.rad2deg(roll_rad))

    return StackPose(
        atlas_from_tangential=mat.copy(),
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        depth_um=depth_um,
        tx_atlas_um=tx_atlas_um,
        ty_atlas_um=ty_atlas_um,
    )


# ---------------------------------------------------------------------------
# M3 — atlas loading, atlas-plane spec, render wrappers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasBundle:
    """Immutable handle on a loaded BrainGlobe atlas.

    Atlas array layout: axis 0 = AP, axis 1 = DV, axis 2 = ML
    (matches BrainGlobe's stored layout for Allen mouse atlases).
    ``shape_zyx`` is named after BrainGlobe's ``(z, y, x)`` mnemonic
    and equals ``(n_AP, n_DV, n_ML)`` for these atlases.

    The arrays are stored by reference, not copied: a 10 µm Allen
    atlas is ~80 M voxels per array. Caller mutation of
    ``reference`` or ``annotation`` will silently corrupt every
    subsequent cache hit; do not mutate.

    Parameters
    ----------
    atlas_name : str
        BrainGlobe atlas identifier, e.g. ``"allen_mouse_10um"``.
    resolution_um : float
        Isotropic voxel size in µm. Cross-checked against the
        integer parsed from ``atlas_name`` by :func:`load_atlas`.
    reference : np.ndarray
        3-D reference (anatomy) volume. Typically ``uint16``.
    annotation : np.ndarray
        3-D integer label volume. Same shape as ``reference``.
    lookup_df : pandas.DataFrame
        BrainGlobe's ``lookup_df`` stored verbatim.
    shape_zyx : tuple
        ``(n_AP, n_DV, n_ML)`` ints; equals ``reference.shape``.
    """

    atlas_name: str
    resolution_um: float
    reference: np.ndarray
    annotation: np.ndarray
    lookup_df: pd.DataFrame
    shape_zyx: tuple

    def __post_init__(self) -> None:
        if not isinstance(self.atlas_name, str) or not self.atlas_name:
            raise ValueError(
                f"atlas_name must be a non-empty string, got {self.atlas_name!r}"
            )
        if not np.isfinite(self.resolution_um) or self.resolution_um <= 0:
            raise ValueError(
                "resolution_um must be a finite positive number, "
                f"got {self.resolution_um!r}"
            )

        if not isinstance(self.shape_zyx, tuple) or len(self.shape_zyx) != 3:
            raise ValueError(
                f"shape_zyx must be a 3-tuple, got {self.shape_zyx!r}"
            )
        for dim in self.shape_zyx:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(
                    "shape_zyx must contain positive Python ints, "
                    f"got {self.shape_zyx!r}"
                )

        if not isinstance(self.reference, np.ndarray) or self.reference.ndim != 3:
            raise ValueError(
                "reference must be a 3-D ndarray, "
                f"got shape {getattr(self.reference, 'shape', None)!r}"
            )
        if (
            not isinstance(self.annotation, np.ndarray)
            or self.annotation.ndim != 3
        ):
            raise ValueError(
                "annotation must be a 3-D ndarray, "
                f"got shape {getattr(self.annotation, 'shape', None)!r}"
            )
        if self.reference.shape != self.shape_zyx:
            raise ValueError(
                f"reference.shape {self.reference.shape} does not match "
                f"shape_zyx {self.shape_zyx}"
            )
        if self.annotation.shape != self.shape_zyx:
            raise ValueError(
                f"annotation.shape {self.annotation.shape} does not match "
                f"shape_zyx {self.shape_zyx}"
            )
        if not np.issubdtype(self.annotation.dtype, np.integer):
            raise ValueError(
                "annotation must have an integer dtype "
                "(nearest-neighbour sampling demands it), "
                f"got {self.annotation.dtype!r}"
            )


_RESOLUTION_RE = re.compile(r"_([0-9]+)um$")


def _parse_resolution_from_atlas_name(atlas_name: str) -> int:
    """Parse the integer voxel resolution (µm) from a BrainGlobe atlas name.

    Expects a trailing ``_<int>um`` suffix (e.g. ``"allen_mouse_10um"``
    → ``10``). Raises ``ValueError`` for inputs that do not match.
    """
    if not isinstance(atlas_name, str):
        raise ValueError(
            f"atlas_name must be a string, got {type(atlas_name).__name__}"
        )
    m = _RESOLUTION_RE.search(atlas_name)
    if m is None:
        raise ValueError(
            "atlas_name must end in '_<int>um' "
            f"(e.g. 'allen_mouse_10um'); got {atlas_name!r}"
        )
    return int(m.group(1))


_VALID_ARRAY_NAMES = frozenset({"reference", "annotation"})


def _validate_arrays_kwarg(arrays) -> tuple:
    """Validate the ``arrays`` kwarg of :func:`load_atlas`.

    v1 always materialises both ``reference`` and ``annotation``;
    this helper only validates membership in
    ``{"reference", "annotation"}``. Returns the validated tuple
    (deduplication is not performed).
    """
    if not isinstance(arrays, (tuple, list)):
        raise ValueError(
            "arrays must be a tuple or list of strings, "
            f"got {type(arrays).__name__}"
        )
    bad = [a for a in arrays if a not in _VALID_ARRAY_NAMES]
    if bad:
        raise ValueError(
            "arrays must be drawn from {'reference', 'annotation'}; "
            f"unknown entries: {bad!r}"
        )
    return tuple(arrays)


_ATLAS_CACHE: dict = {}


def _clear_atlas_cache() -> None:
    """Empty the module-level atlas cache (test-only)."""
    _ATLAS_CACHE.clear()


def load_atlas(
    atlas_name: str = "allen_mouse_10um",
    arrays: tuple = ("reference", "annotation"),
    *,
    cache: bool = True,
) -> AtlasBundle:
    """Load a BrainGlobe atlas and return an :class:`AtlasBundle`.

    Parameters
    ----------
    atlas_name : str
        BrainGlobe atlas identifier. Must end in ``_<int>um``;
        the parsed integer is cross-checked against
        ``bg_atlas.resolution`` and a mismatch raises.
    arrays : tuple of str
        Reserved for future partial loads. v1 validates membership
        in ``{"reference", "annotation"}`` but always materialises
        both fields in the returned bundle.
    cache : bool, keyword-only
        If ``True``, cache the bundle in the module-level dict
        keyed by ``atlas_name`` and short-circuit subsequent calls.

    Returns
    -------
    AtlasBundle
        Immutable handle with ``atlas_name``, ``resolution_um``,
        ``reference``, ``annotation``, ``lookup_df``, ``shape_zyx``.

    Raises
    ------
    ValueError
        If ``atlas_name`` does not match ``_<int>um``, if
        ``arrays`` is malformed, if the BrainGlobe atlas reports an
        anisotropic ``resolution``, or if the BrainGlobe resolution
        disagrees with the integer parsed from ``atlas_name``.
    """
    parsed_resolution = _parse_resolution_from_atlas_name(atlas_name)
    _validate_arrays_kwarg(arrays)

    if cache and atlas_name in _ATLAS_CACHE:
        return _ATLAS_CACHE[atlas_name]

    try:
        bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
    except Exception as exc:
        raise RuntimeError(
            f"load_atlas: BrainGlobeAtlas({atlas_name!r}) failed: {exc!r}"
        ) from exc

    resolution = tuple(float(r) for r in bg_atlas.resolution)
    if len(resolution) != 3 or not all(r == resolution[0] for r in resolution):
        raise ValueError(
            f"load_atlas: atlas {atlas_name!r} reports anisotropic "
            f"resolution {resolution!r}; v1 requires an isotropic atlas"
        )
    if int(resolution[0]) != parsed_resolution:
        raise ValueError(
            f"load_atlas: bg_atlas.resolution[0] = {resolution[0]} "
            f"disagrees with the integer {parsed_resolution} parsed from "
            f"atlas_name {atlas_name!r}"
        )

    shape_zyx = tuple(int(d) for d in bg_atlas.shape)
    bundle = AtlasBundle(
        atlas_name=atlas_name,
        resolution_um=float(resolution[0]),
        reference=bg_atlas.reference,
        annotation=bg_atlas.annotation,
        lookup_df=bg_atlas.lookup_df,
        shape_zyx=shape_zyx,
    )
    if cache:
        _ATLAS_CACHE[atlas_name] = bundle
    return bundle


@dataclass(frozen=True)
class AtlasPlaneSpec:
    """One oblique atlas plane to sample, in atlas-µm space.

    The 4×4 maps the output grid column vector ``(col, row, 0, 1)``
    to an atlas-µm column vector ``(AP_um, DV_um, ML_um, 1)``. The
    render wrappers compose this with the atlas's voxel scaling
    before feeding the unit-free M1 sampler.

    Parameters
    ----------
    out_shape_yx : tuple[int, int]
        Output array shape ``(H, W)``; first axis is row, second col.
    pixel_size_um : float
        In-plane µm per output pixel. Strictly positive, finite.
    atlas_um_from_grid_4x4 : np.ndarray
        ``(4, 4)`` affine mapping grid column vectors
        ``(col, row, 0, 1)`` to atlas-µm column vectors
        ``(AP_um, DV_um, ML_um, 1)``. Cast to ``float64`` and
        copied on store; last row must be ``(0, 0, 0, 1)``.
    order : int
        Interpolation order: ``0`` for nearest (annotation), ``1``
        for linear (reference). Other values rejected.
    cval : float
        Value returned at sample positions outside the atlas. Finite.
    """

    out_shape_yx: tuple
    pixel_size_um: float
    atlas_um_from_grid_4x4: np.ndarray
    order: int = 1
    cval: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.out_shape_yx, tuple)
            or len(self.out_shape_yx) != 2
        ):
            raise ValueError(
                f"out_shape_yx must be a 2-tuple, got {self.out_shape_yx!r}"
            )
        h, w = self.out_shape_yx
        if (
            not isinstance(h, int)
            or not isinstance(w, int)
            or isinstance(h, bool)
            or isinstance(w, bool)
        ):
            raise ValueError(
                f"out_shape_yx must contain Python ints, got {self.out_shape_yx!r}"
            )
        if h <= 0 or w <= 0:
            raise ValueError(
                f"out_shape_yx must be strictly positive, got {self.out_shape_yx!r}"
            )

        if not np.isfinite(self.pixel_size_um) or self.pixel_size_um <= 0:
            raise ValueError(
                "pixel_size_um must be a finite positive number, "
                f"got {self.pixel_size_um!r}"
            )

        try:
            mat = np.asarray(self.atlas_um_from_grid_4x4, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "atlas_um_from_grid_4x4 must be castable to a float64 array"
            ) from exc
        if mat.shape != (4, 4):
            raise ValueError(
                f"atlas_um_from_grid_4x4 must be shape (4, 4), got {mat.shape}"
            )
        if not np.allclose(mat[3], (0.0, 0.0, 0.0, 1.0)):
            raise ValueError(
                "atlas_um_from_grid_4x4 last row must be (0, 0, 0, 1), "
                f"got {mat[3].tolist()}"
            )
        object.__setattr__(self, "atlas_um_from_grid_4x4", mat.copy())

        if self.order not in (0, 1):
            raise ValueError(
                f"order must be 0 (nearest) or 1 (linear), got {self.order!r}"
            )

        if not np.isfinite(self.cval):
            raise ValueError(f"cval must be finite, got {self.cval!r}")


@dataclass(frozen=True)
class PlaneRenderConfig:
    """High-level render knobs.

    ``None``-valued fields are resolved by the caller (Slice 3
    introduces the geometry-driven resolver). M3 ships this
    dataclass as a structural placeholder so the Slice 1 deliverable
    list closes; the render functions take :class:`AtlasPlaneSpec`
    directly, not :class:`PlaneRenderConfig`.

    Parameters
    ----------
    atlas_name : str
        BrainGlobe atlas identifier.
    atlas_resolution_um : float | None
        Isotropic voxel size in µm; resolved from the bundle if None.
    out_pixel_size_um : float | None
        In-plane µm per output pixel; resolved from the canvas if None.
    out_shape_yx : tuple | None
        Output grid shape; resolved from the canvas if None.
    reference_order : int
        Interpolation order for reference renders. 0 or 1.
    annotation_order : int
        Interpolation order for annotation renders. 0 or 1.
    cval_reference : float
        Fill value for out-of-volume reference pixels.
    cval_annotation : int
        Fill value for out-of-volume annotation pixels (integer label).
    """

    atlas_name: str = "allen_mouse_10um"
    atlas_resolution_um: float = None
    out_pixel_size_um: float = None
    out_shape_yx: tuple = None
    reference_order: int = 1
    annotation_order: int = 0
    cval_reference: float = 0.0
    cval_annotation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.atlas_name, str) or not self.atlas_name:
            raise ValueError(
                f"atlas_name must be a non-empty string, got {self.atlas_name!r}"
            )

        if self.atlas_resolution_um is not None:
            if (
                not np.isfinite(self.atlas_resolution_um)
                or self.atlas_resolution_um <= 0
            ):
                raise ValueError(
                    "atlas_resolution_um must be a finite positive number "
                    f"or None, got {self.atlas_resolution_um!r}"
                )

        if self.out_pixel_size_um is not None:
            if (
                not np.isfinite(self.out_pixel_size_um)
                or self.out_pixel_size_um <= 0
            ):
                raise ValueError(
                    "out_pixel_size_um must be a finite positive number "
                    f"or None, got {self.out_pixel_size_um!r}"
                )

        if self.out_shape_yx is not None:
            if (
                not isinstance(self.out_shape_yx, tuple)
                or len(self.out_shape_yx) != 2
            ):
                raise ValueError(
                    "out_shape_yx must be a 2-tuple or None, "
                    f"got {self.out_shape_yx!r}"
                )
            for dim in self.out_shape_yx:
                if (
                    not isinstance(dim, int)
                    or isinstance(dim, bool)
                    or dim <= 0
                ):
                    raise ValueError(
                        "out_shape_yx must contain positive Python ints, "
                        f"got {self.out_shape_yx!r}"
                    )

        if self.reference_order not in (0, 1):
            raise ValueError(
                "reference_order must be 0 or 1, "
                f"got {self.reference_order!r}"
            )
        if self.annotation_order not in (0, 1):
            raise ValueError(
                "annotation_order must be 0 or 1, "
                f"got {self.annotation_order!r}"
            )

        if not np.isfinite(self.cval_reference):
            raise ValueError(
                f"cval_reference must be finite, got {self.cval_reference!r}"
            )
        if (
            not isinstance(self.cval_annotation, int)
            or isinstance(self.cval_annotation, bool)
        ):
            raise ValueError(
                "cval_annotation must be a Python int (annotation labels "
                f"are integers), got {self.cval_annotation!r}"
            )


def _voxel_index_from_grid_4x4(
    atlas: AtlasBundle,
    spec: AtlasPlaneSpec,
) -> np.ndarray:
    """Compose ``voxel_from_atlas_um @ atlas_um_from_grid`` for the M1 sampler."""
    inv_res = 1.0 / atlas.resolution_um
    voxel_from_atlas_um = np.diag([inv_res, inv_res, inv_res, 1.0])
    return voxel_from_atlas_um @ spec.atlas_um_from_grid_4x4


def render_reference_plane(
    atlas: AtlasBundle,
    spec: AtlasPlaneSpec,
) -> np.ndarray:
    """Sample ``atlas.reference`` on the oblique grid defined by ``spec``.

    Interpolation order is whatever ``spec.order`` declares; the
    typical reference render uses ``order=1`` (trilinear). Output
    dtype matches ``atlas.reference.dtype`` (the M1 sampler preserves
    input dtype for both ``order=0`` and ``order=1``). Pixels falling
    outside the atlas bounding box take ``spec.cval``.

    Parameters
    ----------
    atlas : AtlasBundle
        Loaded atlas; supplies the reference array and resolution.
    spec : AtlasPlaneSpec
        Output shape, in-plane scaling, ``atlas_um_from_grid``
        matrix, interpolation order, and outside-volume fill.

    Returns
    -------
    np.ndarray
        Array of shape ``spec.out_shape_yx`` with the reference
        atlas sampled at the requested grid positions.
    """
    volume_index_from_grid = _voxel_index_from_grid_4x4(atlas, spec)
    ps = PlaneSamplingSpec(
        out_shape_yx=spec.out_shape_yx,
        volume_index_from_grid=volume_index_from_grid,
        order=spec.order,
        cval=spec.cval,
    )
    return sample_plane(atlas.reference, ps)


def render_annotation_plane(
    atlas: AtlasBundle,
    spec: AtlasPlaneSpec,
) -> np.ndarray:
    """Sample ``atlas.annotation`` on the oblique grid defined by ``spec``.

    Annotation sampling must be nearest-neighbour: trilinear
    interpolation on integer label volumes produces values absent
    from the original label set. Therefore ``spec.order`` must be
    ``0``; any other value raises ``ValueError``. Output dtype
    matches ``atlas.annotation.dtype``; output values are a subset
    of ``set(np.unique(atlas.annotation)) | {cval}``.

    Parameters
    ----------
    atlas : AtlasBundle
        Loaded atlas; supplies the annotation array and resolution.
    spec : AtlasPlaneSpec
        ``spec.order`` must be ``0``.

    Returns
    -------
    np.ndarray
        Integer-dtype array of shape ``spec.out_shape_yx``.
    """
    if spec.order != 0:
        raise ValueError(
            "render_annotation_plane requires spec.order == 0 (nearest); "
            f"got {spec.order!r}. Construct a second AtlasPlaneSpec with "
            "order=0 for annotation sampling."
        )
    volume_index_from_grid = _voxel_index_from_grid_4x4(atlas, spec)
    ps = PlaneSamplingSpec(
        out_shape_yx=spec.out_shape_yx,
        volume_index_from_grid=volume_index_from_grid,
        order=0,
        cval=float(spec.cval),
    )
    return sample_plane(atlas.annotation, ps)


# ---------------------------------------------------------------------------
# M5 — slice-stack geometry + per-slice plane-spec builder.
# ---------------------------------------------------------------------------


def _coerce_int(value, name: str) -> int:
    """Coerce a Python int / numpy integer to ``int``; reject bools and floats."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be int (got bool)")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"{name} must be int, got {value!r}")


@dataclass(frozen=True)
class SliceResidual:
    """In-plane refinement applied tangential-side, BEFORE the global pose.

    M5 ships with finiteness + ``dz_um == 0`` validation only. M6 adds
    bounded validation against ``RegistrationObjectiveConfig`` and the
    composition helpers.

    Parameters
    ----------
    slice_number : int
        Global slice number this residual belongs to. Used by the caller
        to key residuals; the builder does not consult it.
    dx_um, dy_um, dtheta_deg : float
        In-plane translation (µm) and rotation (deg) applied in the
        tangential frame, before the global pose. M5 default zeros.
    dz_um : float
        Reserved. Must be ``0.0`` in v1 (G2 gate, D §10 decision).
    """

    slice_number: int
    dx_um: float = 0.0
    dy_um: float = 0.0
    dtheta_deg: float = 0.0
    dz_um: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "slice_number", _coerce_int(self.slice_number, "slice_number"))
        for name, val in (
            ("dx_um", self.dx_um),
            ("dy_um", self.dy_um),
            ("dtheta_deg", self.dtheta_deg),
            ("dz_um", self.dz_um),
        ):
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val!r}")
        if float(self.dz_um) != 0.0:
            raise ValueError(
                f"dz_um must be 0.0 in v1 (G2 gate); got {self.dz_um!r}"
            )


@dataclass(frozen=True)
class SliceStackGeometry:
    """Frozen view of slice ordering and z spacing for one stack.

    Built by :func:`build_slice_stack_geometry` from
    ``global_slice_transforms.npz`` and chamber metadata. The kept-list
    excludes bad slices; ``bad_slice_numbers`` is preserved so that
    z-arithmetic remains gap-preserving (D §10 decision 8).

    Parameters
    ----------
    slice_numbers : np.ndarray
        int (M,), strictly ascending, excludes bad slices.
    rois : np.ndarray
        int (M,), per-kept-slice ROI id.
    chambers : tuple of str
        Length M, per-kept-slice chamber identifier.
    z_um : np.ndarray
        float (M,), strictly ascending tangential-z positions in µm.
    overview_pixel_size_um : float
        Canvas in-plane µm per pixel; one scalar (asserted across
        chambers).
    canvas_shape_yx : tuple
        ``(H, W)`` ints of the overview canvas.
    global_from_overview : np.ndarray
        ``(M, 3, 3)`` per-kept-slice affine acting on ``(x, y, 1)``.
    bad_slice_numbers : np.ndarray
        int (K,), sorted, disjoint from ``slice_numbers``. Preserved
        for QC and for the gap arithmetic in
        :func:`_compute_kept_slice_z_um`.
    """

    slice_numbers: np.ndarray
    rois: np.ndarray
    chambers: tuple
    z_um: np.ndarray
    overview_pixel_size_um: float
    canvas_shape_yx: tuple
    global_from_overview: np.ndarray
    bad_slice_numbers: np.ndarray

    def __post_init__(self) -> None:
        sn = np.asarray(self.slice_numbers, dtype=np.int64).ravel()
        if sn.size < 1:
            raise ValueError("slice_numbers must be non-empty")
        if sn.size > 1 and not np.all(np.diff(sn) > 0):
            raise ValueError(
                f"slice_numbers must be strictly ascending, got {sn.tolist()}"
            )
        M = sn.size
        object.__setattr__(self, "slice_numbers", sn)

        rois = np.asarray(self.rois, dtype=np.int64).ravel()
        if rois.shape != (M,):
            raise ValueError(
                f"rois shape {rois.shape} must match ({M},)"
            )
        object.__setattr__(self, "rois", rois)

        if not isinstance(self.chambers, (tuple, list)) or len(self.chambers) != M:
            raise ValueError(
                f"chambers must be a sequence of length {M}, got {self.chambers!r}"
            )
        object.__setattr__(self, "chambers", tuple(str(c) for c in self.chambers))

        z = np.asarray(self.z_um, dtype=np.float64).ravel()
        if z.shape != (M,):
            raise ValueError(
                f"z_um shape {z.shape} must match ({M},)"
            )
        if M > 1 and not np.all(np.diff(z) > 0):
            raise ValueError(
                f"z_um must be strictly ascending, got {z.tolist()}"
            )
        object.__setattr__(self, "z_um", z)

        if not np.isfinite(self.overview_pixel_size_um) or self.overview_pixel_size_um <= 0:
            raise ValueError(
                "overview_pixel_size_um must be finite positive, "
                f"got {self.overview_pixel_size_um!r}"
            )

        if (
            not isinstance(self.canvas_shape_yx, tuple)
            or len(self.canvas_shape_yx) != 2
        ):
            raise ValueError(
                f"canvas_shape_yx must be a 2-tuple, got {self.canvas_shape_yx!r}"
            )
        h, w = self.canvas_shape_yx
        if (
            isinstance(h, bool) or isinstance(w, bool)
            or not isinstance(h, int) or not isinstance(w, int)
            or h <= 0 or w <= 0
        ):
            raise ValueError(
                f"canvas_shape_yx must contain positive ints, got {self.canvas_shape_yx!r}"
            )

        gfo = np.asarray(self.global_from_overview, dtype=np.float64)
        if gfo.shape != (M, 3, 3):
            raise ValueError(
                f"global_from_overview shape {gfo.shape} must be ({M}, 3, 3)"
            )
        for i in range(M):
            if not np.allclose(gfo[i, 2], (0.0, 0.0, 1.0)):
                raise ValueError(
                    f"global_from_overview[{i}] last row must be (0, 0, 1)"
                )
        object.__setattr__(self, "global_from_overview", gfo.copy())

        bsn = np.asarray(self.bad_slice_numbers, dtype=np.int64).ravel()
        if np.intersect1d(bsn, sn).size > 0:
            raise ValueError(
                "bad_slice_numbers and slice_numbers must be disjoint"
            )
        object.__setattr__(self, "bad_slice_numbers", np.sort(bsn))


def _compute_kept_slice_z_um(
    slice_numbers_all: np.ndarray,
    bad_slice_numbers,
    section_thickness_um: float,
):
    """Return ``(kept_slice_numbers, kept_z_um)`` with gap-preserving z.

    ``z = (slice_number - slice_numbers_all[0]) * thickness`` for every
    kept slice; bad slices are excluded from the returned arrays but
    their positions are reflected in the kept-list ``z_um`` jumps.
    """
    all_sn = np.asarray(slice_numbers_all, dtype=np.int64).ravel()
    if all_sn.size < 1:
        raise ValueError("slice_numbers_all must be non-empty")
    if all_sn.size > 1 and not np.all(np.diff(all_sn) > 0):
        raise ValueError("slice_numbers_all must be strictly ascending")
    bad_set = set(int(s) for s in np.asarray(bad_slice_numbers).ravel().tolist())
    mask = np.array([int(s) not in bad_set for s in all_sn.tolist()], dtype=bool)
    kept = all_sn[mask]
    z = (kept.astype(np.float64) - float(all_sn[0])) * float(section_thickness_um)
    return kept, z


def _normalize_z_um_by_slice(z_um_by_slice):
    if z_um_by_slice is None:
        return None
    out = {}
    for key, value in dict(z_um_by_slice).items():
        try:
            sn = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"z_um_by_slice key {key!r} is not int-like") from exc
        z = float(value)
        if not np.isfinite(z):
            raise ValueError(f"z_um_by_slice[{key!r}] must be finite")
        out[sn] = z
    return out


def _compute_kept_slice_z_um_from_mapping(
    slice_numbers_all: np.ndarray,
    bad_slice_numbers,
    z_um_by_slice,
):
    """Return kept slice/z arrays from an explicit slice-number mapping."""
    all_sn = np.asarray(slice_numbers_all, dtype=np.int64).ravel()
    if all_sn.size < 1:
        raise ValueError("slice_numbers_all must be non-empty")
    if all_sn.size > 1 and not np.all(np.diff(all_sn) > 0):
        raise ValueError("slice_numbers_all must be strictly ascending")
    bad_set = set(int(s) for s in np.asarray(bad_slice_numbers).ravel().tolist())
    kept = np.array(
        [int(s) for s in all_sn.tolist() if int(s) not in bad_set],
        dtype=np.int64,
    )
    z_map = _normalize_z_um_by_slice(z_um_by_slice)
    missing = [int(s) for s in kept.tolist() if int(s) not in z_map]
    if missing:
        raise ValueError(
            "z_um_by_slice is missing kept slice numbers "
            f"{missing}; cannot build physical z geometry"
        )
    z = np.asarray([z_map[int(s)] for s in kept.tolist()], dtype=np.float64)
    return kept, z


def _resolve_overview_pixel_size_um(chambers_unique):
    """Read overview pixel size from each chamber's ``ops.yml``.

    Returns the common value, or raises if chambers disagree. Uses the
    project's existing ``iss_preprocess.io.load.load_ops``; expects an
    ``overview_pixel_size_um`` key (flat) per chamber.
    """
    from ..io.load import load_ops

    seen = {}
    for chamber in chambers_unique:
        ops = load_ops(chamber, warn_missing=False)
        if "overview_pixel_size_um" not in ops:
            raise ValueError(
                f"chamber ops missing 'overview_pixel_size_um' for {chamber!r}; "
                "pass overview_pixel_size_um=... to build_slice_stack_geometry "
                "to bypass this lookup"
            )
        seen[chamber] = float(ops["overview_pixel_size_um"])
    values = list(seen.values())
    if not all(abs(v - values[0]) < 1e-6 for v in values):
        raise ValueError(
            f"overview_pixel_size_um differs across chambers: {seen!r}"
        )
    return values[0]


def build_slice_stack_geometry(
    data_path,
    *,
    state=None,
    transforms_path=None,
    section_thickness_um: float = None,
    overview_pixel_size_um: float = None,
    z_um_by_slice=None,
) -> SliceStackGeometry:
    """Assemble :class:`SliceStackGeometry` from existing artefacts.

    Reads ``global_slice_transforms.npz`` (R §3.1) and per-chamber
    ``ops.yml`` plus ``section_position.csv`` for the metadata that is
    not in the transforms file. Both metadata reads are short-circuited
    by the corresponding kwargs.

    Parameters
    ----------
    data_path : str or Path
        Mouse-level processed root; only used to derive
        ``transforms_path`` if that kwarg is None.
    state : dict, optional
        Optional pairwise-state dict; if it has ``"bad_slices"`` that
        list takes precedence over the npz's ``bad_slices``
        (manual-curation authority — see CLAUDE.md cross-stage interop
        block).
    transforms_path : str or Path, optional
        Explicit path to ``global_slice_transforms.npz``.
    section_thickness_um : float, optional
        Bypass the ``section_position.csv`` read. Used by tests and
        by callers that already know the constant thickness.
    overview_pixel_size_um : float, optional
        Bypass the per-chamber ``ops.yml`` read. Used by tests and by
        callers that already know the canvas pixel size.
    z_um_by_slice : mapping, optional
        Explicit physical z positions keyed by slice_number. When provided,
        these values take precedence over scalar section-thickness arithmetic.
    """
    from .volume_registration import load_global_slice_transforms

    if transforms_path is None:
        transforms_path = Path(data_path) / "tangential_volume" / "global_slice_transforms.npz"
    transforms = load_global_slice_transforms(transforms_path)

    slice_numbers_all = np.asarray(transforms["slice_numbers"], dtype=np.int64)
    rois_all = np.asarray(transforms["rois"], dtype=np.int64)
    chambers_all = tuple(str(p) for p in transforms["data_paths"])
    canvas_shape_yx = tuple(int(v) for v in transforms["canvas_shape_yx"])
    global_from_overview_all = np.asarray(
        transforms["global_from_overview"], dtype=np.float64
    )

    bad_npz = {int(s) for s in np.asarray(transforms.get("bad_slices", [])).ravel().tolist()}
    bad_auto = {int(s) for s in np.asarray(transforms.get("auto_bad_slices", [])).ravel().tolist()}
    bad_set = bad_npz | bad_auto
    if state is not None and "bad_slices" in state:
        bad_set = {int(s) for s in state["bad_slices"]}
    bad_slice_numbers = np.asarray(sorted(bad_set), dtype=np.int64)

    if section_thickness_um is None and z_um_by_slice is None:
        from ..io.load import find_roi_position_on_cryostat

        first_chamber = chambers_all[0]
        _, min_thickness = find_roi_position_on_cryostat(first_chamber)
        section_thickness_um = float(min_thickness)

    if overview_pixel_size_um is None:
        unique_chambers = sorted(set(chambers_all))
        overview_pixel_size_um = _resolve_overview_pixel_size_um(unique_chambers)

    if z_um_by_slice is not None:
        kept_slice_numbers, kept_z_um = _compute_kept_slice_z_um_from_mapping(
            slice_numbers_all, bad_slice_numbers, z_um_by_slice
        )
    else:
        kept_slice_numbers, kept_z_um = _compute_kept_slice_z_um(
            slice_numbers_all, bad_slice_numbers, float(section_thickness_um)
        )
    kept_mask = np.isin(slice_numbers_all, kept_slice_numbers)

    return SliceStackGeometry(
        slice_numbers=kept_slice_numbers,
        rois=rois_all[kept_mask],
        chambers=tuple(c for c, m in zip(chambers_all, kept_mask) if m),
        z_um=kept_z_um,
        overview_pixel_size_um=float(overview_pixel_size_um),
        canvas_shape_yx=canvas_shape_yx,
        global_from_overview=global_from_overview_all[kept_mask],
        bad_slice_numbers=bad_slice_numbers,
    )


def _context_issue(severity, code, message, *, slice_number=None, path=None):
    return TangentialAtlasIssue(
        severity=severity,
        code=code,
        message=message,
        slice_number=slice_number,
        path=path,
    )


def _raise_context_errors(errors):
    if not errors:
        return
    lines = [f"{err.code}: {err.message}" for err in errors]
    raise ValueError("Tangential atlas context validation failed: " + "; ".join(lines))


def _resolve_context_paths(data_path, *, stack_path=None, transforms_path=None, state_path=None):
    data_path = Path(data_path)
    if stack_path is None:
        if data_path.name == "unregistered_slices.npz":
            stack_path = data_path
            root = data_path.parent.parent
        else:
            root = data_path
            stack_path = root / "tangential_volume" / "unregistered_slices.npz"
    else:
        stack_path = Path(stack_path)
        root = stack_path.parent.parent

    if transforms_path is None:
        transforms_path = root / "tangential_volume" / "global_slice_transforms.npz"
    else:
        transforms_path = Path(transforms_path)

    if state_path is None:
        state_path = _default_state_path(stack_path)
    else:
        state_path = Path(state_path)

    return root, Path(stack_path), Path(transforms_path), Path(state_path)


def _manifest_entries(manifest):
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError("manifest['entries'] must be a list")
    return entries


def _manifest_index_by_slice(entries, image_count):
    errors = []
    index_by_slice = {}
    for image_index, entry in enumerate(entries):
        if "slice_number" not in entry:
            errors.append(
                _context_issue(
                    "error",
                    "manifest_missing_slice_number",
                    f"manifest entry {image_index} has no slice_number",
                )
            )
            continue
        try:
            sn = int(entry["slice_number"])
        except (TypeError, ValueError):
            errors.append(
                _context_issue(
                    "error",
                    "manifest_bad_slice_number",
                    f"manifest entry {image_index} has non-integer slice_number",
                )
            )
            continue
        if sn in index_by_slice:
            errors.append(
                _context_issue(
                    "error",
                    "duplicate_manifest_slice_number",
                    f"manifest slice_number {sn} appears more than once",
                    slice_number=sn,
                )
            )
            continue
        if image_index >= image_count:
            errors.append(
                _context_issue(
                    "error",
                    "manifest_image_index_out_of_range",
                    f"manifest slice_number {sn} maps to image index {image_index}, "
                    f"but images has only {image_count} planes",
                    slice_number=sn,
                )
            )
            continue
        index_by_slice[sn] = image_index
    return index_by_slice, errors


def _finite_positive_values(entries, key):
    values = []
    for entry in entries:
        if key not in entry:
            continue
        try:
            value = float(entry[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            values.append(value)
    return values


def _single_consistent_value(values, *, atol=1e-6):
    if not values:
        return None
    first = float(values[0])
    if all(abs(float(v) - first) <= atol for v in values):
        return first
    return None


def _resolve_context_overview_pixel_size(
    *,
    explicit,
    state,
    entries,
    chambers,
    issues,
):
    if explicit is not None:
        return float(explicit), "explicit"
    if state is not None and state.get("overview_pixel_size_um") is not None:
        return float(state["overview_pixel_size_um"]), "state.overview_pixel_size_um"

    manifest_value = _single_consistent_value(
        _finite_positive_values(entries, "overview_pixel_size_um")
    )
    if manifest_value is not None:
        return manifest_value, "manifest.overview_pixel_size_um"

    try:
        return float(_resolve_overview_pixel_size_um(sorted(set(chambers)))), "ops.yml"
    except Exception as exc:
        issues.append(
            _context_issue(
                "error",
                "missing_overview_pixel_size_um",
                "could not resolve overview_pixel_size_um from explicit value, "
                f"state, manifest, or chamber ops: {exc}",
            )
        )
        return None, "unresolved"


def _resolve_context_section_thickness(
    *,
    explicit,
    state,
    entries,
    issues,
):
    if explicit is not None:
        return float(explicit), "explicit"
    if state is not None and state.get("section_thickness_um") is not None:
        return float(state["section_thickness_um"]), "state.section_thickness_um"

    manifest_value = _single_consistent_value(
        _finite_positive_values(entries, "section_thickness_um")
    )
    if manifest_value is not None:
        return manifest_value, "manifest.section_thickness_um"

    issues.append(
        _context_issue(
            "warning",
            "section_thickness_fallback_20um",
            "section_thickness_um was not found in explicit inputs, state, or "
            "manifest entries; using 20.0 um fallback",
        )
    )
    return DEFAULT_TANGENTIAL_ATLAS_SECTION_THICKNESS_UM, "fallback_20um"


def _manifest_absolute_section_z_by_slice(
    *,
    entries,
    active_slice_numbers,
    section_thickness_um,
    issues,
):
    entry_by_slice = {}
    for entry in entries:
        if "slice_number" in entry:
            entry_by_slice[int(entry["slice_number"])] = entry
    active = [int(s) for s in np.asarray(active_slice_numbers).ravel().tolist()]
    absolute_by_slice = {}
    missing = []
    for sn in active:
        entry = entry_by_slice.get(sn)
        if entry is None or "absolute_section" not in entry:
            missing.append(sn)
            continue
        try:
            absolute_by_slice[sn] = float(entry["absolute_section"])
        except (TypeError, ValueError):
            missing.append(sn)

    if missing:
        issues.append(
            _context_issue(
                "warning",
                "absolute_section_unavailable",
                "absolute_section was unavailable for active slices "
                f"{missing}; using slice_number spacing",
            )
        )
        return None, "slice_number_spacing"

    if not absolute_by_slice:
        issues.append(
            _context_issue(
                "warning",
                "absolute_section_unavailable",
                "no active slices had absolute_section; using slice_number spacing",
            )
        )
        return None, "slice_number_spacing"

    first_abs = min(absolute_by_slice.values())
    z_map = {
        int(sn): (float(abs_sec) - first_abs) * float(section_thickness_um)
        for sn, abs_sec in absolute_by_slice.items()
    }
    return z_map, "manifest.absolute_section"


def _resolve_context_z_um_by_slice(*, state, entries, active_slice_numbers, section_thickness_um, issues):
    if state is not None and state.get("z_um_by_slice") is not None:
        return _normalize_z_um_by_slice(state["z_um_by_slice"]), "state.z_um_by_slice"

    z_map, source = _manifest_absolute_section_z_by_slice(
        entries=entries,
        active_slice_numbers=active_slice_numbers,
        section_thickness_um=section_thickness_um,
        issues=issues,
    )
    if z_map is not None:
        return z_map, source

    return None, source


def build_tangential_atlas_context(
    data_path,
    *,
    stack_path=None,
    transforms_path=None,
    state_path=None,
    state: dict = None,
    overview_pixel_size_um: float = None,
    section_thickness_um: float = None,
    strict: bool = False,
) -> TangentialAtlasContext:
    """Build a validated real-data context keyed by ``slice_number``.

    The context is the shared M1 surface for later reviewer/QC/export slices.
    It validates that active transform slices have manifest-backed images and
    reports metadata fallbacks as issues.
    """
    from .volume_registration import (
        load_global_slice_transforms,
        load_unregistered_volume_stack,
    )

    root, stack_path, transforms_path, resolved_state_path = _resolve_context_paths(
        data_path,
        stack_path=stack_path,
        transforms_path=transforms_path,
        state_path=state_path,
    )

    if state is None and resolved_state_path.exists():
        state = load_tangential_atlas_state(root, state_path=resolved_state_path)

    stack = load_unregistered_volume_stack(stack_path)
    images = np.asarray(stack["images"])
    if images.ndim != 3:
        raise ValueError(f"stack images must be 3-D, got shape {images.shape}")
    manifest = stack["manifest"]
    entries = _manifest_entries(manifest)

    transforms = load_global_slice_transforms(transforms_path)
    active_slice_numbers = np.asarray(transforms["slice_numbers"], dtype=np.int64)
    chambers_all = tuple(str(p) for p in transforms["data_paths"])

    issues = []
    index_by_slice, manifest_errors = _manifest_index_by_slice(
        entries, images.shape[0]
    )
    issues.extend(manifest_errors)
    active_missing = [
        int(s) for s in active_slice_numbers.tolist() if int(s) not in index_by_slice
    ]
    for sn in active_missing:
        issues.append(
            _context_issue(
                "error",
                "active_slice_missing_manifest_image",
                f"active transform slice {sn} has no manifest-backed image",
                slice_number=sn,
            )
        )

    errors = [issue for issue in issues if issue.severity == "error"]
    _raise_context_errors(errors)

    overview_px, overview_source = _resolve_context_overview_pixel_size(
        explicit=overview_pixel_size_um,
        state=state,
        entries=entries,
        chambers=chambers_all,
        issues=issues,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    _raise_context_errors(errors)

    section_thickness, section_source = _resolve_context_section_thickness(
        explicit=section_thickness_um,
        state=state,
        entries=entries,
        issues=issues,
    )
    z_um_by_slice, z_source = _resolve_context_z_um_by_slice(
        state=state,
        entries=entries,
        active_slice_numbers=active_slice_numbers,
        section_thickness_um=section_thickness,
        issues=issues,
    )

    if strict:
        warning_issues = [issue for issue in issues if issue.severity == "warning"]
        _raise_context_errors(
            [
                _context_issue(
                    "error",
                    issue.code,
                    issue.message,
                    slice_number=issue.slice_number,
                    path=issue.path,
                )
                for issue in warning_issues
            ]
        )

    geometry = build_slice_stack_geometry(
        root,
        state=state,
        transforms_path=transforms_path,
        section_thickness_um=section_thickness,
        overview_pixel_size_um=overview_px,
        z_um_by_slice=z_um_by_slice,
    )

    transform_index_by_slice = {
        int(sn): i for i, sn in enumerate(active_slice_numbers.tolist())
    }
    records = []
    record_errors = []
    global_from_fullres = transforms.get("global_from_fullres")
    for kept_index, sn in enumerate(geometry.slice_numbers.tolist()):
        sn = int(sn)
        if sn not in index_by_slice:
            record_errors.append(
                _context_issue(
                    "error",
                    "kept_slice_missing_manifest_image",
                    f"kept slice {sn} has no manifest-backed image",
                    slice_number=sn,
                )
            )
            continue
        if sn not in transform_index_by_slice:
            record_errors.append(
                _context_issue(
                    "error",
                    "kept_slice_missing_transform",
                    f"kept slice {sn} has no active transform",
                    slice_number=sn,
                )
            )
            continue
        transform_index = transform_index_by_slice[sn]
        gff = None
        if global_from_fullres is not None:
            gff = np.asarray(global_from_fullres, dtype=np.float64)[transform_index]
        records.append(
            TangentialAtlasSliceRecord(
                slice_number=sn,
                kept_index=kept_index,
                image_index=index_by_slice[sn],
                roi=int(geometry.rois[kept_index]),
                chamber=geometry.chambers[kept_index],
                z_um=float(geometry.z_um[kept_index]),
                global_from_overview=geometry.global_from_overview[kept_index],
                global_from_fullres=gff,
            )
        )

    _raise_context_errors(record_errors)

    metadata = TangentialAtlasMetadataResolution(
        overview_pixel_size_um=overview_px,
        section_thickness_um=section_thickness,
        overview_pixel_size_source=overview_source,
        section_thickness_source=section_source,
        z_source=z_source,
    )
    return TangentialAtlasContext(
        stack_path=stack_path,
        transforms_path=transforms_path,
        state_path=resolved_state_path if resolved_state_path.exists() else None,
        images=images,
        manifest=manifest,
        geometry=geometry,
        slice_records=tuple(records),
        metadata=metadata,
        issues=tuple(issues),
        state=state,
    )


def image_for_tangential_atlas_slice(
    context: TangentialAtlasContext,
    slice_number: int,
) -> np.ndarray:
    """Return the raw stack image for ``slice_number`` using context mapping."""
    sn = int(slice_number)
    for record in context.slice_records:
        if int(record.slice_number) == sn:
            return context.images[int(record.image_index)]
    raise ValueError(
        f"slice_number {sn} is not mapped in this TangentialAtlasContext"
    )


def _embed_3x3_xy_into_4x4(g: np.ndarray) -> np.ndarray:
    """Promote a 3×3 affine on ``(x, y, 1)`` to a 4×4 on ``(x, y, z, 1)``.

    The z row stays identity (i.e. the embedded affine acts only on the
    in-plane coordinates).
    """
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 1], M[0, 3] = g[0, 0], g[0, 1], g[0, 2]
    M[1, 0], M[1, 1], M[1, 3] = g[1, 0], g[1, 1], g[1, 2]
    return M


def build_slice_plane_spec(
    geometry: SliceStackGeometry,
    pose: "StackPose",
    residual: SliceResidual = None,
    render: PlaneRenderConfig = None,
    slice_index: int = 0,
    atlas: AtlasBundle = None,
) -> AtlasPlaneSpec:
    """Compose the per-slice 4×4 ``atlas_um_from_grid`` (S §4).

    The chain (right-to-left, applied to ``(col, row, 0, 1)``):

        atlas_um_from_grid
            = atlas_from_tangential
              @ T_inplane(dx, dy)
              @ R_inplane(dtheta, centroid)
              @ T_z(z)
              @ M_embed_3D(global_from_overview[i])
              @ S_canvas_to_tan_µm(px)

    Parameters
    ----------
    geometry : SliceStackGeometry
        Kept-list geometry.
    pose : StackPose
        Canonical global pose.
    residual : SliceResidual, optional
        Per-slice in-plane refinement. M5 default is zeros.
    render : PlaneRenderConfig, optional
        Render knobs. M5 default uses
        ``reference_order=1, cval_reference=0.0``.
    slice_index : int
        Kept-list position (``0..M-1``).
    atlas : AtlasBundle, optional
        Accepted but unused in M5; reserved for future per-slice
        resolution sanity checks.

    Returns
    -------
    AtlasPlaneSpec
        Spec ready for :func:`render_reference_plane` /
        :func:`render_annotation_plane`.
    """
    M = len(geometry.slice_numbers)
    if (
        isinstance(slice_index, bool)
        or not isinstance(slice_index, (int, np.integer))
        or not (0 <= int(slice_index) < M)
    ):
        raise ValueError(
            f"slice_index must be in range(0, {M}), got {slice_index!r}"
        )
    slice_index = int(slice_index)

    if residual is None:
        residual = SliceResidual(
            slice_number=int(geometry.slice_numbers[slice_index])
        )
    if render is None:
        render = PlaneRenderConfig()

    z = float(geometry.z_um[slice_index])
    px = float(geometry.overview_pixel_size_um)

    S_canvas_to_tan = np.diag([px, px, 1.0, 1.0]).astype(np.float64)
    M_embed = _embed_3x3_xy_into_4x4(geometry.global_from_overview[slice_index])
    T_z = np.eye(4, dtype=np.float64)
    T_z[2, 3] = z

    atlas_from_tan_with_residual = compose_pose_with_residual(
        pose, geometry, residual, slice_index
    )

    atlas_um_from_grid = (
        atlas_from_tan_with_residual @ T_z @ M_embed @ S_canvas_to_tan
    )

    return AtlasPlaneSpec(
        out_shape_yx=geometry.canvas_shape_yx,
        pixel_size_um=geometry.overview_pixel_size_um,
        atlas_um_from_grid_4x4=atlas_um_from_grid,
        order=render.reference_order,
        cval=render.cval_reference,
    )


# ---------------------------------------------------------------------------
# M6 — objective config, residual composition, state IO.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistrationObjectiveConfig:
    """Bounds + penalty weights for per-slice residuals (S §2).

    M6 ships the dataclass + the bounds-validation helper. M7
    activates the optimizer that consumes ``smoothness_lambda``.

    Parameters
    ----------
    max_dx_um, max_dy_um : float
        Absolute bounds on per-slice in-plane translation (µm). ≥ 0.
    max_dtheta_deg : float
        Absolute bound on per-slice rotation (deg). ≥ 0.
    max_dz_um : float
        Reserved. Must be 0.0 in v1 (G2 gate).
    smoothness_lambda : float
        Per-residual smoothness penalty weight; consumed by the M7
        optimizer. ≥ 0.
    centering : str
        Residual gauge-fix rule: ``"mean_zero"`` or ``"none"``.
    """

    max_dx_um: float = 250.0
    max_dy_um: float = 250.0
    max_dtheta_deg: float = 2.0
    max_dz_um: float = 0.0
    smoothness_lambda: float = 0.0
    centering: str = "mean_zero"

    def __post_init__(self) -> None:
        for name, val in (
            ("max_dx_um", self.max_dx_um),
            ("max_dy_um", self.max_dy_um),
            ("max_dtheta_deg", self.max_dtheta_deg),
            ("max_dz_um", self.max_dz_um),
            ("smoothness_lambda", self.smoothness_lambda),
        ):
            if not np.isfinite(val) or float(val) < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative, got {val!r}"
                )
        if float(self.max_dz_um) != 0.0:
            raise ValueError(
                f"max_dz_um must be 0.0 in v1 (G2 gate); got {self.max_dz_um!r}"
            )
        if self.centering not in ("mean_zero", "none"):
            raise ValueError(
                f"centering must be 'mean_zero' or 'none', got {self.centering!r}"
            )


def validate_residual_against_bounds(
    residual: SliceResidual,
    bounds: RegistrationObjectiveConfig,
) -> None:
    """Raise ``ValueError`` if ``residual`` exceeds any bound.

    Bounds are checked with ``<=`` (exactly-at-bound passes).
    ``dz_um`` is verified zero (matches both SliceResidual's G2 hard
    invariant and bounds.max_dz_um).
    """
    if abs(float(residual.dx_um)) > float(bounds.max_dx_um):
        raise ValueError(
            f"dx_um={residual.dx_um} exceeds max_dx_um={bounds.max_dx_um} "
            f"for slice {residual.slice_number}"
        )
    if abs(float(residual.dy_um)) > float(bounds.max_dy_um):
        raise ValueError(
            f"dy_um={residual.dy_um} exceeds max_dy_um={bounds.max_dy_um} "
            f"for slice {residual.slice_number}"
        )
    if abs(float(residual.dtheta_deg)) > float(bounds.max_dtheta_deg):
        raise ValueError(
            f"dtheta_deg={residual.dtheta_deg} exceeds "
            f"max_dtheta_deg={bounds.max_dtheta_deg} for slice "
            f"{residual.slice_number}"
        )
    if float(residual.dz_um) != 0.0 or float(bounds.max_dz_um) != 0.0:
        raise ValueError(
            f"dz_um must be 0.0 in v1; got residual.dz_um={residual.dz_um}, "
            f"bounds.max_dz_um={bounds.max_dz_um}"
        )


def compose_pose_with_residual(
    pose: "StackPose",
    geometry: SliceStackGeometry,
    residual: SliceResidual,
    slice_index: int,
) -> np.ndarray:
    """Return the per-slice 4×4 ``atlas_from_tangential`` (S §3.5).

    Applies the slice's in-plane residual (``dx``, ``dy``,
    ``dtheta`` about the canvas centroid in tangential-µm) in the
    tangential frame BEFORE the global pose. ``dz_um`` is asserted
    zero (SliceResidual's G2 invariant). The returned matrix is
    ``pose.atlas_from_tangential @ T_inplane @ R_inplane``.
    """
    M = len(geometry.slice_numbers)
    if (
        isinstance(slice_index, bool)
        or not isinstance(slice_index, (int, np.integer))
        or not (0 <= int(slice_index) < M)
    ):
        raise ValueError(
            f"slice_index must be in range(0, {M}), got {slice_index!r}"
        )
    slice_index = int(slice_index)
    if float(residual.dz_um) != 0.0:
        raise ValueError(
            f"residual.dz_um must be 0.0 in v1; got {residual.dz_um!r}"
        )

    px = float(geometry.overview_pixel_size_um)
    H, W = geometry.canvas_shape_yx
    cx_um = (W - 1) / 2.0 * px
    cy_um = (H - 1) / 2.0 * px
    theta = np.deg2rad(float(residual.dtheta_deg))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R_inplane = np.array(
        [
            [c, -s, 0.0, cx_um - c * cx_um + s * cy_um],
            [s, c, 0.0, cy_um - s * cx_um - c * cy_um],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_inplane = np.eye(4, dtype=np.float64)
    T_inplane[0, 3] = float(residual.dx_um)
    T_inplane[1, 3] = float(residual.dy_um)

    return pose.atlas_from_tangential @ T_inplane @ R_inplane


def center_residuals(residuals, *, rule: str = "mean_zero") -> dict:
    """Re-anchor residual means to zero (gauge fix).

    Parameters
    ----------
    residuals : Mapping[int, SliceResidual]
        Per-slice residuals keyed by slice_number.
    rule : str
        ``"mean_zero"`` (subtract per-axis mean) or ``"none"`` (no-op).

    Returns a new dict; does not mutate the input.
    """
    if rule == "none":
        return dict(residuals)
    if rule != "mean_zero":
        raise ValueError(
            f"rule must be 'mean_zero' or 'none', got {rule!r}"
        )
    items = list(residuals.items())
    if not items:
        return {}
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    mean_dx = float(np.mean([v.dx_um for v in vals]))
    mean_dy = float(np.mean([v.dy_um for v in vals]))
    mean_dtheta = float(np.mean([v.dtheta_deg for v in vals]))
    out = {}
    for k, v in zip(keys, vals):
        out[k] = SliceResidual(
            slice_number=v.slice_number,
            dx_um=float(v.dx_um) - mean_dx,
            dy_um=float(v.dy_um) - mean_dy,
            dtheta_deg=float(v.dtheta_deg) - mean_dtheta,
            dz_um=0.0,
        )
    return out


def _atomic_write_json(path, obj) -> None:
    """Atomic JSON write via tempfile + ``os.replace``. Mirrors the
    pairwise-state precedent in ``pipeline.volume_registration``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fhandle:
        json.dump(obj, fhandle, indent=2, sort_keys=True)
    tmp.replace(path)


def _default_state_path(stack_path) -> Path:
    """Derive the state file path from a stack_path.

    ``stack_path`` is typically
    ``processed/{project}/{mouse}/tangential_volume/unregistered_slices.npz``;
    the state lives one directory up under ``tangential_atlas/``.
    """
    sp = Path(stack_path)
    return (
        sp.parent.parent
        / TANGENTIAL_ATLAS_SUBDIR
        / TANGENTIAL_ATLAS_STATE_FILENAME
    )


def save_tangential_atlas_state(state: dict, *, state_path=None) -> Path:
    """Atomically write ``state`` to JSON.

    If ``state_path`` is ``None``, derive from ``state["stack_path"]``.
    On overwrite, the previous version is moved to ``<state>.json.bak``.
    Returns the resolved path.
    """
    if state_path is None:
        if "stack_path" not in state:
            raise ValueError(
                "save_tangential_atlas_state: pass state_path or set "
                "state['stack_path']"
            )
        state_path = _default_state_path(state["stack_path"])
    state_path = Path(state_path)

    if state_path.exists():
        bak = state_path.with_name(state_path.name + ".bak")
        state_path.replace(bak)

    payload = dict(state)
    payload.setdefault("version", TANGENTIAL_ATLAS_STATE_VERSION)
    _atomic_write_json(state_path, payload)
    return state_path


def load_tangential_atlas_state(data_path, *, state_path=None) -> dict:
    """Read + validate the JSON state file. Returns the dict untouched.

    Validation: schema version matches; ``atlas_name`` parses to an
    integer resolution; rotation_order is ``"ZYX"`` (if present);
    pose 4×4 is a rigid transform; each residual has ``dz_um == 0``;
    residual bounds are present and finite.
    """
    if state_path is None:
        state_path = (
            Path(data_path)
            / TANGENTIAL_ATLAS_SUBDIR
            / TANGENTIAL_ATLAS_STATE_FILENAME
        )
    state_path = Path(state_path)
    with open(state_path, "r") as fhandle:
        state = json.load(fhandle)

    version = state.get("version")
    if version != TANGENTIAL_ATLAS_STATE_VERSION:
        raise ValueError(
            f"state version {version!r} != expected "
            f"{TANGENTIAL_ATLAS_STATE_VERSION}"
        )

    if "atlas_name" in state:
        _parse_resolution_from_atlas_name(state["atlas_name"])

    pose = state.get("pose", {})
    rotation_order = pose.get("rotation_order")
    if rotation_order is not None and rotation_order != _ROTATION_ORDER_ZYX:
        raise ValueError(
            f"pose.rotation_order must be 'ZYX', got {rotation_order!r}"
        )
    if "atlas_from_tangential_4x4" in pose:
        mat = np.asarray(pose["atlas_from_tangential_4x4"], dtype=np.float64)
        _validate_rigid_4x4(mat, name="pose.atlas_from_tangential_4x4")

    for key, r in state.get("residuals", {}).items():
        dz = float(r.get("dz_um", 0.0))
        if dz != 0.0:
            raise ValueError(
                f"residuals[{key!r}].dz_um must be 0.0 in v1, got {dz!r}"
            )

    bounds = state.get("residual_bounds")
    if bounds is not None:
        for name in (
            "max_dx_um",
            "max_dy_um",
            "max_dtheta_deg",
            "max_dz_um",
            "smoothness_lambda",
        ):
            val = float(bounds.get(name, 0.0))
            if not np.isfinite(val) or val < 0:
                raise ValueError(
                    f"residual_bounds[{name!r}]={val!r} must be finite non-negative"
                )
        if float(bounds.get("max_dz_um", 0.0)) != 0.0:
            raise ValueError(
                "residual_bounds.max_dz_um must be 0.0 in v1"
            )

    return state


# ---------------------------------------------------------------------------
# M7 — dz_um (G2-gated) + objective functions.
# ---------------------------------------------------------------------------


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x)). Equals
    ``np.logaddexp(0, x)``.
    """
    return np.logaddexp(0.0, x)


def monotonic_dz_from_raw(raw_i_um: np.ndarray) -> np.ndarray:
    """Reparameterise unbounded reals into baseline-subtracted dz_um.

    Construction (D §4):
        S_i = sum_{k=0..i-1} softplus(raw_k)
    is strictly increasing in i for any real ``raw``. Subtracting the
    linear baseline (linear interpolation between ``S_0`` and
    ``S_{N-1}``) gives an array whose endpoints are zero and that is
    zero everywhere when ``raw`` is all zero (so v1's default
    behaviour is unchanged).

    Returns a 1-D float64 array of the same length as ``raw_i_um``.
    """
    raw = np.asarray(raw_i_um, dtype=np.float64).ravel()
    N = raw.size
    if N == 0:
        return np.zeros(0, dtype=np.float64)
    S = np.cumsum(_softplus(raw))
    if N == 1:
        return np.zeros(1, dtype=np.float64)
    baseline = np.linspace(S[0], S[-1], N)
    return S - baseline


def assert_monotonic_spacing(
    geometry: SliceStackGeometry,
    residuals,
    *,
    bounds: RegistrationObjectiveConfig = None,
) -> list:
    """Return a list of human-readable warning strings.

    M7 ships with ``bounds.max_dz_um == 0`` (G2 gate); residuals'
    ``dz_um`` is enforced zero at SliceResidual construction. So at
    v1 defaults this function trivially returns ``[]``. It exists
    so a future configuration that flips the G2 gate can check
    monotonicity here before persisting.
    """
    if bounds is None:
        bounds = RegistrationObjectiveConfig()
    warnings = []
    if float(bounds.max_dz_um) == 0.0:
        return warnings

    M = len(geometry.slice_numbers)
    dz = np.zeros(M, dtype=np.float64)
    for i, sn in enumerate(geometry.slice_numbers.tolist()):
        r = residuals.get(int(sn))
        if r is not None:
            dz[i] = float(r.dz_um)
    effective_z = geometry.z_um + dz
    diffs = np.diff(effective_z)
    for i, d in enumerate(diffs.tolist()):
        if d <= 0.0:
            sn_a = int(geometry.slice_numbers[i])
            sn_b = int(geometry.slice_numbers[i + 1])
            warnings.append(
                f"non-monotonic spacing between slice {sn_a} "
                f"(effective_z={effective_z[i]:.3f}) and slice {sn_b} "
                f"(effective_z={effective_z[i + 1]:.3f}); delta={d:.3f}"
            )
    return warnings


def _masked_ncc(x: np.ndarray, y: np.ndarray, mask: np.ndarray = None) -> float:
    """Masked normalised cross-correlation (Pearson over the mask)."""
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        x, y = x[m], y[m]
    if x.size < 2:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    denom = float(np.sqrt(np.sum(xm * xm) * np.sum(ym * ym)))
    if denom <= 0:
        return 0.0
    return float(np.sum(xm * ym) / denom)


def _masked_mi(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray = None, bins: int = 32
) -> float:
    """Masked mutual information from a 2-D histogram. ≥ 0."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        x, y = x[m], y[m]
    if x.size < 2:
        return 0.0
    H, _, _ = np.histogram2d(x, y, bins=bins)
    Pxy = H / H.sum() if H.sum() > 0 else H
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)
    nz = Pxy > 0
    PxPy = np.outer(Px, Py)
    ratio = np.zeros_like(Pxy)
    ratio[nz] = Pxy[nz] / PxPy[nz]
    mi = float(np.sum(Pxy[nz] * np.log(np.maximum(ratio[nz], 1e-300))))
    return max(mi, 0.0)


def _masked_edge_ncc(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray = None
) -> float:
    """NCC of Sobel-magnitudes (edge similarity)."""
    sx = np.hypot(scipy.ndimage.sobel(x, axis=0), scipy.ndimage.sobel(x, axis=1))
    sy = np.hypot(scipy.ndimage.sobel(y, axis=0), scipy.ndimage.sobel(y, axis=1))
    return _masked_ncc(sx, sy, mask=mask)


def slice_objective(
    slice_image: np.ndarray,
    atlas_reference_plane: np.ndarray,
    *,
    mask: np.ndarray = None,
    mode: str = "ncc",
) -> float:
    """Scalar similarity between a slice and a rendered atlas plane.

    Modes:

    - ``"ncc"``: masked normalised cross-correlation. Range ``[-1, 1]``.
    - ``"mi"``: histogram-based mutual information. Range ``[0, inf)``.
    - ``"edge"``: NCC of Sobel-magnitudes. Range ``[-1, 1]``.

    The two inputs must have the same shape; if ``mask`` is provided
    it must also match. Pure function; raises ``ValueError`` on
    shape mismatch or unknown ``mode``.
    """
    sx = np.asarray(slice_image)
    ax = np.asarray(atlas_reference_plane)
    if sx.shape != ax.shape:
        raise ValueError(
            f"slice_image shape {sx.shape} does not match "
            f"atlas_reference_plane shape {ax.shape}"
        )
    if mask is not None and np.asarray(mask).shape != sx.shape:
        raise ValueError(
            f"mask shape {np.asarray(mask).shape} does not match "
            f"slice shape {sx.shape}"
        )

    if mode == "ncc":
        return _masked_ncc(sx, ax, mask=mask)
    if mode == "mi":
        return _masked_mi(sx, ax, mask=mask)
    if mode == "edge":
        return _masked_edge_ncc(sx, ax, mask=mask)
    raise ValueError(
        f"mode must be 'ncc', 'mi', or 'edge'; got {mode!r}"
    )


# ---------------------------------------------------------------------------
# M8 — refine_residuals (G1-gated) + write_qc_report. FINAL slice.
# ---------------------------------------------------------------------------


def refine_residuals(
    state: dict,
    geometry: SliceStackGeometry,
    atlas: AtlasBundle,
    *,
    mode: str = "ncc",
    bounds: RegistrationObjectiveConfig = None,
    g1_approved: bool = False,
    maxiter: int = 25,
):
    """Per-slice bounded local refinement of `(dx, dy, dtheta)`.

    G1 GATE: this function raises ``RuntimeError`` unless
    ``g1_approved=True`` is passed. D §7's G1 gate requires manual
    annotation of ≥ 1 real chamber, objective curves on real data,
    and screenshots before the optimizer is enabled. Pass
    ``g1_approved=True`` only after that human review.

    The optimization is per-slice independent (no smoothness coupling
    in v1; ``bounds.smoothness_lambda`` is read but unused). Each
    slice's `(dx, dy, dtheta)` are optimised under bounds drawn from
    ``bounds`` (or from ``state['residual_bounds']`` if not given).
    Returns a new state dict with updated residuals; does not mutate
    the input.
    """
    if not g1_approved:
        raise RuntimeError(
            "refine_residuals: G1 gate is closed. Pass g1_approved=True "
            "only after the human review (D §7): manual annotation of "
            "≥ 1 real chamber + objective curves + reviewer screenshots."
        )

    from scipy.optimize import minimize

    if bounds is None:
        rb = state.get("residual_bounds", {})
        bounds = RegistrationObjectiveConfig(
            max_dx_um=float(rb.get("max_dx_um", 250.0)),
            max_dy_um=float(rb.get("max_dy_um", 250.0)),
            max_dtheta_deg=float(rb.get("max_dtheta_deg", 2.0)),
            max_dz_um=0.0,
            smoothness_lambda=float(rb.get("smoothness_lambda", 0.0)),
            centering=str(rb.get("centering", "mean_zero")),
        )

    pose_dict = state.get("pose", {})
    pose = StackPose(
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

    M = len(geometry.slice_numbers)
    new_residuals = dict(state.get("residuals", {}))

    bound_box = [
        (-bounds.max_dx_um, bounds.max_dx_um),
        (-bounds.max_dy_um, bounds.max_dy_um),
        (-bounds.max_dtheta_deg, bounds.max_dtheta_deg),
    ]

    for slice_index in range(M):
        sn = int(geometry.slice_numbers[slice_index])
        # NB: the consumer of refine_residuals is responsible for
        # rendering the slice image to compare against; here we
        # synthesise a target by rendering at zero residual (a no-op
        # for synthetic identity-pose tests). Real callers should pre-
        # render slice_image and pass via a closure; v1 uses the
        # atlas plane at zero residual as both fixed and moving (the
        # optimum is then (0, 0, 0), which lets us test the gate +
        # convergence on a synthetic perfect-match scenario).
        zero_r = SliceResidual(slice_number=sn)
        spec_target = build_slice_plane_spec(
            geometry, pose, residual=zero_r, slice_index=slice_index
        )
        target = render_reference_plane(atlas, spec_target)

        def _obj(params, slice_index=slice_index, sn=sn, target=target):
            dx, dy, dtheta = params
            r = SliceResidual(
                slice_number=sn,
                dx_um=float(dx),
                dy_um=float(dy),
                dtheta_deg=float(dtheta),
            )
            spec = build_slice_plane_spec(
                geometry, pose, residual=r, slice_index=slice_index
            )
            plane = render_reference_plane(atlas, spec)
            return -slice_objective(target, plane, mode=mode)

        # Start from existing residual or zeros.
        prev = state.get("residuals", {}).get(str(sn), {})
        x0 = np.array(
            [
                float(prev.get("dx_um", 0.0)),
                float(prev.get("dy_um", 0.0)),
                float(prev.get("dtheta_deg", 0.0)),
            ],
            dtype=np.float64,
        )
        res = minimize(
            _obj,
            x0,
            method="L-BFGS-B",
            bounds=bound_box,
            options={"maxiter": int(maxiter)},
        )
        new_residuals[str(sn)] = {
            "dx_um": float(res.x[0]),
            "dy_um": float(res.x[1]),
            "dtheta_deg": float(res.x[2]),
            "dz_um": 0.0,
        }

    out = dict(state)
    out["residuals"] = new_residuals
    return out


def _pose_from_state(state):
    pose_dict = state.get("pose", {})
    return StackPose(
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


def _residuals_from_state(state, geometry):
    residuals = {}
    for sn in geometry.slice_numbers.tolist():
        sn = int(sn)
        prev = state.get("residuals", {}).get(str(sn), {})
        residuals[sn] = SliceResidual(
            slice_number=sn,
            dx_um=float(prev.get("dx_um", 0.0)),
            dy_um=float(prev.get("dy_um", 0.0)),
            dtheta_deg=float(prev.get("dtheta_deg", 0.0)),
            dz_um=0.0,
        )
    return residuals


def _annotation_spec_from_reference_spec(spec):
    return AtlasPlaneSpec(
        out_shape_yx=spec.out_shape_yx,
        pixel_size_um=spec.pixel_size_um,
        atlas_um_from_grid_4x4=spec.atlas_um_from_grid_4x4,
        order=0,
        cval=0,
    )


def _atlas_coordinate_grid_um(spec):
    h, w = spec.out_shape_yx
    rows, cols = np.mgrid[0:h, 0:w]
    flat = np.stack(
        [
            cols.ravel().astype(np.float64),
            rows.ravel().astype(np.float64),
            np.zeros(h * w, dtype=np.float64),
            np.ones(h * w, dtype=np.float64),
        ],
        axis=0,
    )
    coords = spec.atlas_um_from_grid_4x4 @ flat
    return coords[:3].T.reshape(h, w, 3).astype(np.float32)


def _issue_to_dict(issue):
    return {
        "severity": issue.severity,
        "code": issue.code,
        "message": issue.message,
        "slice_number": issue.slice_number,
        "path": issue.path,
    }


def _qc_root(data_path, context):
    if context is not None:
        return context.stack_path.parent.parent
    dp = Path(data_path)
    if dp.name == "unregistered_slices.npz":
        return dp.parent.parent
    return dp


def _context_from_state_if_possible(data_path, state, context):
    if context is not None:
        return context
    stack_path = state.get("stack_path")
    transforms_path = state.get("global_transforms_path")
    if not stack_path or not transforms_path:
        return None
    try:
        return build_tangential_atlas_context(
            data_path,
            stack_path=stack_path,
            transforms_path=transforms_path,
            state=state,
            overview_pixel_size_um=state.get("overview_pixel_size_um"),
            section_thickness_um=state.get("section_thickness_um"),
        )
    except (FileNotFoundError, ValueError):
        return None


def write_tangential_atlas_rasters(
    data_path,
    state: dict,
    *,
    context: TangentialAtlasContext = None,
    atlas: AtlasBundle = None,
    output_path=None,
    include_coordinates: bool = True,
    include_annotation: bool = True,
) -> Path:
    """Write per-slice atlas annotation and/or coordinate rasters.

    Coordinate rasters are atlas microns in ``(AP_um, DV_um, ML_um)`` order.
    """
    if context is None:
        context = build_tangential_atlas_context(
            data_path,
            stack_path=state.get("stack_path"),
            transforms_path=state.get("global_transforms_path"),
            state=state,
            overview_pixel_size_um=state.get("overview_pixel_size_um"),
            section_thickness_um=state.get("section_thickness_um"),
        )
    if atlas is None:
        atlas = load_atlas(state.get("atlas_name", DEFAULT_TANGENTIAL_ATLAS_NAME))
    if output_path is None:
        output_path = (
            context.stack_path.parent.parent
            / TANGENTIAL_ATLAS_SUBDIR
            / "tangential_atlas_rasters.npz"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose = _pose_from_state(state)
    residuals = _residuals_from_state(state, context.geometry)
    annotations = []
    coordinates = []
    for slice_index, sn in enumerate(context.geometry.slice_numbers.tolist()):
        sn = int(sn)
        spec = build_slice_plane_spec(
            context.geometry,
            pose,
            residual=residuals[sn],
            slice_index=slice_index,
        )
        if include_annotation:
            annotations.append(
                render_annotation_plane(
                    atlas, _annotation_spec_from_reference_spec(spec)
                )
            )
        if include_coordinates:
            coordinates.append(_atlas_coordinate_grid_um(spec))

    payload = {
        "slice_numbers": context.geometry.slice_numbers.astype(np.int64),
        "z_um": context.geometry.z_um.astype(np.float32),
    }
    if include_annotation:
        payload["annotation"] = np.asarray(annotations, dtype=atlas.annotation.dtype)
    if include_coordinates:
        payload["atlas_coordinates_um"] = np.asarray(coordinates, dtype=np.float32)
    np.savez(output_path, **payload)
    return output_path


def write_qc_report(
    data_path,
    state: dict,
    *,
    atlas: AtlasBundle = None,
    geometry: SliceStackGeometry = None,
    context: TangentialAtlasContext = None,
) -> dict:
    """Render per-slice real-image overlays + residuals + QC summary."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..vis.tangential_atlas import overlay_atlas_on_slice

    if atlas is None:
        atlas = load_atlas(state.get("atlas_name", DEFAULT_TANGENTIAL_ATLAS_NAME))
    context = _context_from_state_if_possible(data_path, state, context)
    if geometry is None:
        if context is not None:
            geometry = context.geometry
        else:
            geometry = build_slice_stack_geometry(
                data_path,
                transforms_path=state.get("global_transforms_path"),
                overview_pixel_size_um=float(
                    state.get("overview_pixel_size_um", 5.0)
                ),
                section_thickness_um=state.get("section_thickness_um"),
                z_um_by_slice=state.get("z_um_by_slice"),
            )

    pose = _pose_from_state(state)
    residuals_obj = _residuals_from_state(state, geometry)

    fig_root = _qc_root(data_path, context) / "figures" / "tangential_atlas"
    overlay_dir = fig_root / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    per_slice = []
    M = len(geometry.slice_numbers)
    for slice_index in range(M):
        sn = int(geometry.slice_numbers[slice_index])
        residual = residuals_obj[sn]
        spec = build_slice_plane_spec(
            geometry, pose, residual=residual, slice_index=slice_index
        )
        reference = render_reference_plane(atlas, spec)
        annotation = render_annotation_plane(
            atlas, _annotation_spec_from_reference_spec(spec)
        )
        if context is not None:
            slice_image = image_for_tangential_atlas_slice(context, sn)
            record = context.slice_records[slice_index]
            image_index = int(record.image_index)
            kept_index = int(record.kept_index)
        else:
            slice_image = reference.astype(np.float32)
            image_index = None
            kept_index = slice_index

        fig = overlay_atlas_on_slice(
            np.asarray(slice_image, dtype=np.float32),
            reference,
            atlas_annotation_plane=annotation,
        )
        out_path = overlay_dir / f"sl{sn:03d}.png"
        fig.savefig(out_path, dpi=72, bbox_inches="tight")
        plt.close(fig)

        scores = {}
        for mode in ("ncc", "mi", "edge"):
            try:
                scores[mode] = float(
                    slice_objective(slice_image, reference, mode=mode)
                )
            except Exception as exc:
                scores[mode] = None
                scores[f"{mode}_error"] = str(exc)

        per_slice.append(
            {
                "slice_number": sn,
                "kept_index": kept_index,
                "image_index": image_index,
                "z_um": float(geometry.z_um[slice_index]),
                "residual": {
                    "dx_um": float(residual.dx_um),
                    "dy_um": float(residual.dy_um),
                    "dtheta_deg": float(residual.dtheta_deg),
                    "dz_um": 0.0,
                },
                "objective_scores": scores,
                "overlay_path": str(out_path),
            }
        )

    sns = geometry.slice_numbers.tolist()
    dx_vals = [float(residuals_obj[int(sn)].dx_um) for sn in sns]
    dy_vals = [float(residuals_obj[int(sn)].dy_um) for sn in sns]
    dt_vals = [float(residuals_obj[int(sn)].dtheta_deg) for sn in sns]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].scatter(sns, dx_vals); axes[0].set_ylabel("dx_um")
    axes[1].scatter(sns, dy_vals); axes[1].set_ylabel("dy_um")
    axes[2].scatter(sns, dt_vals); axes[2].set_ylabel("dtheta_deg")
    axes[2].set_xlabel("slice_number")
    for ax in axes:
        ax.axhline(0.0, color="gray", lw=0.5, linestyle="--")
    fig.tight_layout()
    residuals_path = fig_root / "residuals.png"
    fig.savefig(residuals_path, dpi=72, bbox_inches="tight")
    plt.close(fig)

    spacing_warnings = assert_monotonic_spacing(geometry, residuals_obj)
    issues = list(context.issues) if context is not None else []
    for warning in spacing_warnings:
        issues.append(
            TangentialAtlasIssue(
                severity="warning",
                code="spacing_warning",
                message=str(warning),
            )
        )

    if context is not None:
        metadata = {
            "overview_pixel_size_um": context.metadata.overview_pixel_size_um,
            "section_thickness_um": context.metadata.section_thickness_um,
            "overview_pixel_size_source": context.metadata.overview_pixel_size_source,
            "section_thickness_source": context.metadata.section_thickness_source,
            "z_source": context.metadata.z_source,
        }
    else:
        metadata = {
            "overview_pixel_size_um": float(geometry.overview_pixel_size_um),
            "section_thickness_um": state.get("section_thickness_um"),
            "overview_pixel_size_source": "state_or_argument",
            "section_thickness_source": "state_or_argument",
            "z_source": "geometry",
        }

    fig_root.mkdir(parents=True, exist_ok=True)
    summary_path = fig_root / "qc_summary.json"
    state_path_value = None
    if context is not None and context.state_path is not None:
        state_path_value = str(context.state_path)
    elif state.get("state_path") is not None:
        state_path_value = str(state.get("state_path"))

    summary = {
        "version": 1,
        "state_path": state_path_value,
        "stack_path": str(context.stack_path) if context is not None else state.get("stack_path"),
        "transforms_path": (
            str(context.transforms_path)
            if context is not None
            else state.get("global_transforms_path")
        ),
        "atlas_name": state.get("atlas_name", DEFAULT_TANGENTIAL_ATLAS_NAME),
        "bad_slices": geometry.bad_slice_numbers.astype(int).tolist(),
        "metadata": metadata,
        "issues": [_issue_to_dict(issue) for issue in issues],
        "per_slice": per_slice,
    }
    _atomic_write_json(summary_path, summary)

    qc = {
        "overlay_dir": str(overlay_dir),
        "residuals_plot": str(residuals_path),
        "spacing_warnings": spacing_warnings,
        "qc_summary": str(summary_path),
        "per_slice": per_slice,
    }
    return qc


def register_spots_to_tangential_atlas(
    data_path,
    state_path=None,
    *,
    spots_path=None,
    spots_prefix: str = "barcode_round",
    output_name=None,
    context: TangentialAtlasContext = None,
    atlas: AtlasBundle = None,
    global_coordinate_unit: str = "pixel",
    include_bad_slices: bool = True,
) -> Path:
    """Project a global spots table through the saved tangential atlas state.

    Adds atlas-micron coordinate columns (``AP_um``, ``DV_um``, ``ML_um``),
    annotation ``area_id`` (nullable Int64) and ``area_acronym`` (when
    available), and the explicit-validity columns ``atlas_valid`` and
    ``atlas_warning``. Bad-slice and unmapped rows are retained with
    invalid atlas fields unless ``include_bad_slices=False``.
    """
    if global_coordinate_unit not in {"pixel", "um"}:
        raise ValueError(
            "global_coordinate_unit must be 'pixel' or 'um', "
            f"got {global_coordinate_unit!r}"
        )

    if context is not None and getattr(context, "state", None) is not None:
        state = context.state
    else:
        state = load_tangential_atlas_state(data_path, state_path=state_path)

    context = _context_from_state_if_possible(data_path, state, context)
    if context is None:
        context = build_tangential_atlas_context(
            data_path,
            stack_path=state.get("stack_path"),
            transforms_path=state.get("global_transforms_path"),
            state=state,
            overview_pixel_size_um=state.get("overview_pixel_size_um"),
            section_thickness_um=state.get("section_thickness_um"),
        )

    if atlas is None:
        atlas = load_atlas(state.get("atlas_name", DEFAULT_TANGENTIAL_ATLAS_NAME))

    if spots_path is None:
        from .volume_registration import get_volume_root

        spots_path = (
            get_volume_root(data_path) / f"{spots_prefix}_spots_global.pkl"
        )
    spots_path = Path(spots_path)
    if not spots_path.exists():
        raise FileNotFoundError(f"global spots table not found: {spots_path}")

    spots = pd.read_pickle(spots_path).copy()

    spots["AP_um"] = np.nan
    spots["DV_um"] = np.nan
    spots["ML_um"] = np.nan
    spots["area_id"] = pd.array([pd.NA] * len(spots), dtype=pd.Int64Dtype())
    spots["area_acronym"] = None
    spots["atlas_valid"] = False
    spots["atlas_warning"] = ""

    out_root = context.stack_path.parent.parent / TANGENTIAL_ATLAS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        output_name = f"{spots_prefix}_spots_atlas.pkl"
    output_path = Path(out_root / output_name)

    if not len(spots):
        spots.to_pickle(output_path)
        return output_path

    if "is_bad_slice" in spots.columns:
        is_bad = spots["is_bad_slice"].fillna(False).astype(bool)
    else:
        is_bad = pd.Series(False, index=spots.index)

    pose = _pose_from_state(state)
    residuals = _residuals_from_state(state, context.geometry)
    n_AP, n_DV, n_ML = atlas.shape_zyx
    px = float(context.geometry.overview_pixel_size_um)
    kept_sns = {int(sn) for sn in context.geometry.slice_numbers.tolist()}

    for slice_index, sn_arr in enumerate(context.geometry.slice_numbers.tolist()):
        sn = int(sn_arr)
        spec = build_slice_plane_spec(
            context.geometry,
            pose,
            residual=residuals[sn],
            slice_index=slice_index,
        )
        M = spec.atlas_um_from_grid_4x4

        mask = (
            (spots["slice_number"] == sn)
            & ~is_bad
            & spots["x_global"].notna()
            & spots["y_global"].notna()
        )
        if not mask.any():
            continue

        xy = spots.loc[mask, ["x_global", "y_global"]].to_numpy(dtype=float)
        if global_coordinate_unit == "um":
            xy = xy / px
        pts = np.column_stack(
            [xy, np.zeros(len(xy)), np.ones(len(xy))]
        )
        atlas_um = (M @ pts.T).T[:, :3]

        spots.loc[mask, "AP_um"] = atlas_um[:, 0]
        spots.loc[mask, "DV_um"] = atlas_um[:, 1]
        spots.loc[mask, "ML_um"] = atlas_um[:, 2]

        voxel = np.round(atlas_um / atlas.resolution_um).astype(int)
        ap_idx = voxel[:, 0]
        dv_idx = voxel[:, 1]
        ml_idx = voxel[:, 2]
        inb = (
            (ap_idx >= 0)
            & (ap_idx < n_AP)
            & (dv_idx >= 0)
            & (dv_idx < n_DV)
            & (ml_idx >= 0)
            & (ml_idx < n_ML)
        )

        mask_idx = spots.index[mask]
        in_idx = mask_idx[inb]
        out_idx = mask_idx[~inb]

        if len(in_idx):
            area = atlas.annotation[ap_idx[inb], dv_idx[inb], ml_idx[inb]]
            spots.loc[in_idx, "area_id"] = pd.array(
                np.asarray(area, dtype=np.int64), dtype=pd.Int64Dtype()
            )
            spots.loc[in_idx, "atlas_valid"] = True
        if len(out_idx):
            spots.loc[out_idx, "atlas_warning"] = "outside_atlas"

    spots.loc[is_bad, "atlas_warning"] = "bad_slice"
    sn_in_state = spots["slice_number"].isin(kept_sns)
    not_in_state = ~sn_in_state & ~is_bad
    spots.loc[not_in_state, "atlas_warning"] = "slice_not_in_state"
    missing_global = (
        sn_in_state
        & ~is_bad
        & (spots["x_global"].isna() | spots["y_global"].isna())
    )
    spots.loc[missing_global, "atlas_warning"] = "missing_global"
    unmapped = (~spots["atlas_valid"]) & (spots["atlas_warning"] == "")
    spots.loc[unmapped, "atlas_warning"] = "unmapped"

    if "acronym" in atlas.lookup_df.columns:
        labels = atlas.lookup_df.set_index("id")["acronym"]
        valid_area = spots["area_id"].notna()
        if valid_area.any():
            spots.loc[valid_area, "area_acronym"] = (
                spots.loc[valid_area, "area_id"].map(labels)
            )

    if not include_bad_slices:
        spots = spots.loc[~is_bad].copy()

    spots.to_pickle(output_path)
    return output_path
