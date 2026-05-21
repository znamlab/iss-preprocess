"""Tests for ``iss_preprocess.pipeline.tangential_atlas`` (M1).

Synthetic-only. No BrainGlobe download, no real-data dependencies.
"""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.ndimage

from iss_preprocess.pipeline import tangential_atlas as _tangential_atlas_mod
from iss_preprocess.pipeline.tangential_atlas import (
    AtlasBundle,
    AtlasPlaneSpec,
    DEFAULT_TANGENTIAL_ATLAS_SECTION_THICKNESS_UM,
    PlaneRenderConfig,
    PlaneSamplingSpec,
    StackPose,
    TangentialAtlasContext,
    TangentialAtlasIssue,
    build_tangential_atlas_context,
    _clear_atlas_cache,
    _parse_resolution_from_atlas_name,
    _validate_arrays_kwarg,
    image_for_tangential_atlas_slice,
    load_atlas,
    matrix_to_pose,
    pose_to_matrix,
    render_annotation_plane,
    render_reference_plane,
    sample_plane,
    write_tangential_atlas_rasters,
)


def _synthetic_volume(shape=(30, 40, 50), seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape).astype(np.float32)
    return scipy.ndimage.gaussian_filter(raw, sigma=2.0)


def _synthetic_label_volume(shape=(20, 30, 40), seed=1):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape).astype(np.float32)
    smoothed = scipy.ndimage.gaussian_filter(raw, sigma=2.5)
    quantiles = np.quantile(smoothed, [0.25, 0.5, 0.75])
    labels = np.zeros(shape, dtype=np.int16)
    labels[smoothed > quantiles[0]] = 1
    labels[smoothed > quantiles[1]] = 2
    labels[smoothed > quantiles[2]] = 3
    return labels


class TestPlaneSamplingSpec:
    def test_happy_path(self):
        spec = PlaneSamplingSpec(
            out_shape_yx=(8, 10),
            volume_index_from_grid=np.eye(4),
            order=1,
            cval=0.0,
        )
        assert spec.out_shape_yx == (8, 10)
        assert spec.volume_index_from_grid.dtype == np.float64
        assert spec.volume_index_from_grid.shape == (4, 4)
        # Frozen + copied: caller-side mutation must not affect spec.
        orig = np.eye(4)
        spec2 = PlaneSamplingSpec(
            out_shape_yx=(8, 10),
            volume_index_from_grid=orig,
            order=0,
        )
        orig[0, 3] = 999.0
        assert spec2.volume_index_from_grid[0, 3] == 0.0

    @pytest.mark.parametrize("bad_shape", [(3, 3), (4, 3), (4,), (4, 4, 4)])
    def test_rejects_wrong_matrix_shape(self, bad_shape):
        with pytest.raises(ValueError, match="volume_index_from_grid"):
            PlaneSamplingSpec(
                out_shape_yx=(5, 5),
                volume_index_from_grid=np.zeros(bad_shape),
            )

    def test_rejects_last_row_not_homogeneous(self):
        bad = np.eye(4)
        bad[3, 0] = 0.5
        with pytest.raises(ValueError, match="last row"):
            PlaneSamplingSpec(
                out_shape_yx=(5, 5),
                volume_index_from_grid=bad,
            )

    @pytest.mark.parametrize("bad_order", [-1, 2, 3])
    def test_rejects_invalid_order(self, bad_order):
        with pytest.raises(ValueError, match="order"):
            PlaneSamplingSpec(
                out_shape_yx=(5, 5),
                volume_index_from_grid=np.eye(4),
                order=bad_order,
            )

    @pytest.mark.parametrize(
        "bad_shape",
        [(0, 5), (5, 0), (-1, 5), (5, -1), [5, 5], (5,), (5, 5, 5)],
    )
    def test_rejects_invalid_out_shape(self, bad_shape):
        with pytest.raises(ValueError, match="out_shape_yx"):
            PlaneSamplingSpec(
                out_shape_yx=bad_shape,
                volume_index_from_grid=np.eye(4),
            )

    @pytest.mark.parametrize("bad_cval", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite_cval(self, bad_cval):
        with pytest.raises(ValueError, match="cval"):
            PlaneSamplingSpec(
                out_shape_yx=(5, 5),
                volume_index_from_grid=np.eye(4),
                cval=bad_cval,
            )


class TestSamplePlane:
    def test_identity_sample_matches_transposed_slab(self):
        # With volume_index_from_grid = I, grid (col, row, 0, 1) maps to
        # voxel index (i0=col, i1=row, i2=0). Therefore
        # out[r, c] == volume[c, r, 0] == volume[:, :, 0].T[r, c].
        volume = _synthetic_volume(shape=(30, 40, 50))
        h = volume.shape[1]
        w = volume.shape[0]
        spec = PlaneSamplingSpec(
            out_shape_yx=(h, w),
            volume_index_from_grid=np.eye(4),
            order=1,
        )
        out = sample_plane(volume, spec)
        expected = volume[:, :, 0].T
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_linear_interp_half_voxel_midpoint(self):
        # Gradient volume along i0: vol[i, j, k] = i + 0.5.
        # Shift sampling by +0.5 in i0 -> linear interp midpoint of
        # vol[c, r, 0] = c + 0.5 and vol[c+1, r, 0] = c + 1.5,
        # i.e. c + 1.0 at output (r, c).
        d = 10
        i_idx, _, _ = np.indices((d, d, d))
        volume = (i_idx + 0.5).astype(np.float64)
        mat = np.eye(4)
        mat[0, 3] = 0.5
        spec = PlaneSamplingSpec(
            out_shape_yx=(5, 5),
            volume_index_from_grid=mat,
            order=1,
        )
        out = sample_plane(volume, spec)
        expected = np.tile(np.arange(5, dtype=np.float64)[None, :] + 1.0, (5, 1))
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_nearest_preserves_dtype_and_label_set(self):
        labels = _synthetic_label_volume()
        assert labels.dtype == np.int16
        present = set(np.unique(labels).tolist())

        mat = np.eye(4)
        mat[0, 3] = 2.5
        mat[1, 3] = 1.5
        mat[2, 3] = 3.5

        spec_nn = PlaneSamplingSpec(
            out_shape_yx=(10, 12),
            volume_index_from_grid=mat,
            order=0,
        )
        out_nn = sample_plane(labels, spec_nn)
        assert out_nn.dtype == labels.dtype
        assert set(np.unique(out_nn).tolist()) <= present

    @pytest.mark.parametrize("order", [0, 1])
    def test_sample_plane_preserves_input_dtype(self, order):
        # scipy.ndimage.map_coordinates returns the input dtype unless an
        # explicit ``output`` array is supplied. We do not override this.
        labels = _synthetic_label_volume()
        floats = _synthetic_volume()
        mat = np.eye(4)
        mat[0, 3] = 2.5
        for volume in (labels, floats):
            spec = PlaneSamplingSpec(
                out_shape_yx=(8, 8),
                volume_index_from_grid=mat,
                order=order,
            )
            out = sample_plane(volume, spec)
            assert out.dtype == volume.dtype

    def test_linear_interp_produces_intermediate_values_on_float_labels(self):
        # On a float-cast label volume, linear interpolation at fractional
        # offsets produces values that are not in the original integer
        # label set — sanity check that order=1 actually interpolates.
        labels = _synthetic_label_volume().astype(np.float64)
        present = set(np.unique(labels).tolist())
        mat = np.eye(4)
        mat[0, 3] = 2.5
        mat[1, 3] = 1.5
        mat[2, 3] = 3.5
        spec = PlaneSamplingSpec(
            out_shape_yx=(10, 12),
            volume_index_from_grid=mat,
            order=1,
        )
        out = sample_plane(labels, spec)
        assert not (set(np.unique(out).tolist()) <= present)

    @pytest.mark.parametrize("order", [0, 1])
    def test_out_of_volume_takes_cval(self, order):
        volume = _synthetic_volume(shape=(20, 25, 30))
        mat = np.eye(4)
        mat[0, 3] = -1000.0  # all samples land at i0 < 0
        spec = PlaneSamplingSpec(
            out_shape_yx=(7, 9),
            volume_index_from_grid=mat,
            order=order,
            cval=-7.5,
        )
        out = sample_plane(volume, spec)
        assert out.shape == (7, 9)
        np.testing.assert_array_equal(
            out, np.full((7, 9), -7.5, dtype=out.dtype)
        )

    @pytest.mark.parametrize(
        "out_shape", [(7, 9), (3, 15), (15, 3), (10, 10), (1, 20)]
    )
    def test_output_shape(self, out_shape):
        # _synthetic_volume returns float; we additionally assert that
        # the output is floating for a float input.
        volume = _synthetic_volume(shape=(20, 25, 30))
        assert np.issubdtype(volume.dtype, np.floating)
        spec = PlaneSamplingSpec(
            out_shape_yx=out_shape,
            volume_index_from_grid=np.eye(4),
            order=1,
        )
        out = sample_plane(volume, spec)
        assert out.shape == out_shape
        assert np.issubdtype(out.dtype, np.floating)

    def test_inplane_45deg_rotation_handedness(self):
        # Place a Gaussian blob inside an (i1, i2) slab at i0=25, offset
        # +10 voxels along +i2 from the slab centre (25, 25, 25). With
        # the matrix defined below, the blob lands +10 voxels along
        # +col in the output at 0 degrees. After a +45 deg in-plane
        # rotation of the sampling pattern about the output centre, the
        # blob centroid should land at (cy - 5*sqrt(2), cx + 5*sqrt(2)).
        d = 51
        sigma = 1.5
        ii, jj, kk = np.indices((d, d, d))
        blob_i0, blob_i1, blob_i2 = 25, 25, 35
        volume = np.exp(
            -(
                (ii - blob_i0) ** 2
                + (jj - blob_i1) ** 2
                + (kk - blob_i2) ** 2
            )
            / (2.0 * sigma ** 2)
        )

        h, w = 41, 41
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        slab_i0, slab_i1, slab_i2 = 25.0, 25.0, 25.0

        def make_spec(theta_rad):
            c, s = np.cos(theta_rad), np.sin(theta_rad)
            # i0 = slab_i0
            # i1 = s*col + c*row + (slab_i1 - s*cx - c*cy)
            # i2 = c*col - s*row + (slab_i2 - c*cx + s*cy)
            mat = np.array(
                [
                    [0.0, 0.0, 0.0, slab_i0],
                    [s, c, 0.0, slab_i1 - s * cx - c * cy],
                    [c, -s, 0.0, slab_i2 - c * cx + s * cy],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            return PlaneSamplingSpec(
                out_shape_yx=(h, w),
                volume_index_from_grid=mat,
                order=1,
            )

        def centroid(img):
            total = float(img.sum())
            rs, cs = np.indices(img.shape, dtype=np.float64)
            return (float((rs * img).sum() / total),
                    float((cs * img).sum() / total))

        out0 = sample_plane(volume, make_spec(0.0))
        out45 = sample_plane(volume, make_spec(np.pi / 4.0))

        cr0, cc0 = centroid(out0)
        cr45, cc45 = centroid(out45)

        # 0 deg: blob at output (cy, cx + 10)
        assert abs(cr0 - cy) < 0.5
        assert abs(cc0 - (cx + 10.0)) < 0.5

        # +45 deg: blob at output (cy - 5*sqrt(2), cx + 5*sqrt(2))
        expected_cr = cy - 5.0 * np.sqrt(2.0)
        expected_cc = cx + 5.0 * np.sqrt(2.0)
        assert abs(cr45 - expected_cr) < 0.5
        assert abs(cc45 - expected_cc) < 0.5

    def test_rejects_non_3d_volume(self):
        spec = PlaneSamplingSpec(
            out_shape_yx=(5, 5),
            volume_index_from_grid=np.eye(4),
        )
        with pytest.raises(ValueError, match="3-D"):
            sample_plane(np.zeros((10, 10)), spec)


_P_4x4 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class TestPoseToMatrix:
    def test_zero_pose_is_axis_aligned_permutation(self):
        # Identity pose maps tangential (x_tan, y_tan, z_tan) onto atlas
        # (AP, -ML, DV) via the proper rotation P (90° about atlas AP).
        M = pose_to_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_array_equal(M, _P_4x4)

    def test_yaw_90_rotates_AP_to_minus_ML(self):
        # yaw is rotation about atlas DV. At identity x_tan → AP, then
        # yaw rotates AP → -ML in the atlas frame.
        M = pose_to_matrix(90.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        out = M @ np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 0.0, -1.0, 0.0], atol=1e-12)

    def test_pitch_90_rotates_AP_to_DV(self):
        # pitch is rotation about atlas ML. At identity x_tan → AP, then
        # pitch rotates AP → DV in the atlas frame.
        M = pose_to_matrix(0.0, 90.0, 0.0, 0.0, 0.0, 0.0)
        out = M @ np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0, 0.0], atol=1e-12)

    def test_roll_90_rotates_y_tan_to_DV(self):
        # roll is rotation about atlas AP. At identity y_tan → -ML, and
        # _R_about_AP(90°) rotates -ML → +DV in the atlas frame.
        M = pose_to_matrix(0.0, 0.0, 90.0, 0.0, 0.0, 0.0)
        out = M @ np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [0.0, 1.0, 0.0, 0.0], atol=1e-12)

    def test_translation_atlas_frame_at_zero_rotation(self):
        M = pose_to_matrix(
            0.0, 0.0, 0.0,
            depth_um=5.0, tx_atlas_um=2.0, ty_atlas_um=3.0,
        )
        # Order: (tx_atlas, depth, ty_atlas) → (t_AP, t_DV, t_ML).
        np.testing.assert_allclose(M[:3, 3], [2.0, 5.0, 3.0], atol=1e-12)

    @pytest.mark.parametrize(
        "yaw,pitch,roll",
        [
            (30.0, 15.0, -20.0),
            (-45.0, 5.0, 10.0),
            (0.0, 45.0, 0.0),
            (10.0, 0.0, -10.0),
            (90.0, 0.0, 90.0),
        ],
    )
    def test_translation_independent_of_rotation(self, yaw, pitch, roll):
        # Translation interpreted in atlas frame along fixed axes:
        # the (tx, depth, ty) -> (AP, DV, ML) column must equal the
        # input atlas-frame 3-vector for any rotation.
        M = pose_to_matrix(
            yaw, pitch, roll,
            depth_um=5.0, tx_atlas_um=2.0, ty_atlas_um=3.0,
        )
        np.testing.assert_allclose(M[:3, 3], [2.0, 5.0, 3.0], atol=1e-12)

    @pytest.mark.parametrize("bad_order", ["XYZ", "zyx", "", "ZXY"])
    def test_rejects_non_zyx_rotation_order(self, bad_order):
        with pytest.raises(ValueError, match="rotation_order"):
            pose_to_matrix(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           rotation_order=bad_order)

    @pytest.mark.parametrize(
        "field",
        ["yaw_deg", "pitch_deg", "roll_deg",
         "depth_um", "tx_atlas_um", "ty_atlas_um"],
    )
    @pytest.mark.parametrize(
        "bad_value", [float("nan"), float("inf"), float("-inf")]
    )
    def test_rejects_nonfinite_inputs(self, field, bad_value):
        kwargs = dict(
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        kwargs[field] = bad_value
        with pytest.raises(ValueError, match=field):
            pose_to_matrix(**kwargs)

    def test_output_is_float64_4x4(self):
        M = pose_to_matrix(10.0, -5.0, 2.0, 1.0, 2.0, 3.0)
        assert M.dtype == np.float64
        assert M.shape == (4, 4)
        np.testing.assert_allclose(M[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


class TestMatrixToPose:
    def test_permutation_matrix_yields_zero_pose(self):
        # Under the §3.0 identity, the zero-pose matrix is the
        # tangential→atlas permutation P, not I_4.
        pose = matrix_to_pose(_P_4x4)
        assert isinstance(pose, StackPose)
        for field in (
            "yaw_deg", "pitch_deg", "roll_deg",
            "depth_um", "tx_atlas_um", "ty_atlas_um",
        ):
            assert abs(getattr(pose, field)) < 1e-12
        np.testing.assert_allclose(
            pose.atlas_from_tangential, _P_4x4, atol=1e-12
        )

    @pytest.mark.parametrize("bad_shape", [(3, 3), (4, 3), (5, 5)])
    def test_rejects_non_4x4(self, bad_shape):
        with pytest.raises(ValueError, match="shape"):
            matrix_to_pose(np.eye(*bad_shape) if bad_shape[0] == bad_shape[1]
                           else np.zeros(bad_shape))

    @pytest.mark.parametrize(
        "last_row",
        [(0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0)],
    )
    def test_rejects_non_homogeneous_last_row(self, last_row):
        M = np.eye(4)
        M[3] = last_row
        with pytest.raises(ValueError, match="last row"):
            matrix_to_pose(M)

    def test_rejects_non_orthonormal_R(self):
        # det = 8, not orthonormal
        M = np.eye(4)
        M[:3, :3] = 2.0 * np.eye(3)
        with pytest.raises(ValueError, match="rotation"):
            matrix_to_pose(M)

        # Shear: orthogonality breaks.
        M2 = np.eye(4)
        M2[:3, :3] = np.array(
            [[1.0, 0.5, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        with pytest.raises(ValueError, match="rotation"):
            matrix_to_pose(M2)

    def test_gimbal_lock_raises(self):
        M = pose_to_matrix(0.0, 90.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="gimbal"):
            matrix_to_pose(M)


class TestPoseRoundTrip:
    def test_pose_matrix_pose_round_trip_random(self):
        rng = np.random.default_rng(2)
        n = 200
        for _ in range(n):
            yaw = float(rng.uniform(-30.0, 30.0))
            pitch = float(rng.uniform(-30.0, 30.0))
            roll = float(rng.uniform(-30.0, 30.0))
            depth = float(rng.uniform(-1000.0, 1000.0))
            tx = float(rng.uniform(-1000.0, 1000.0))
            ty = float(rng.uniform(-1000.0, 1000.0))

            M = pose_to_matrix(yaw, pitch, roll, depth, tx, ty)
            pose = matrix_to_pose(M)
            M2 = pose_to_matrix(
                pose.yaw_deg,
                pose.pitch_deg,
                pose.roll_deg,
                pose.depth_um,
                pose.tx_atlas_um,
                pose.ty_atlas_um,
            )
            np.testing.assert_allclose(M, M2, atol=1e-10)

            assert abs(pose.yaw_deg - yaw) < 1e-10
            assert abs(pose.pitch_deg - pitch) < 1e-10
            assert abs(pose.roll_deg - roll) < 1e-10
            assert abs(pose.depth_um - depth) < 1e-10
            assert abs(pose.tx_atlas_um - tx) < 1e-10
            assert abs(pose.ty_atlas_um - ty) < 1e-10

    def test_stackpose_construction_consistency_check(self):
        # Identity matrix but non-zero yaw scalar — mismatch must raise.
        with pytest.raises(ValueError, match="disagree"):
            StackPose(
                atlas_from_tangential=np.eye(4),
                yaw_deg=10.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                depth_um=0.0,
                tx_atlas_um=0.0,
                ty_atlas_um=0.0,
            )


# ---------------------------------------------------------------------------
# M3 fixtures and test doubles.
# ---------------------------------------------------------------------------


def _synthetic_atlas_arrays(
    shape=(60, 80, 100), resolution_um=10.0, seed=42
):
    """Build a (reference, annotation) pair for synthetic-atlas tests.

    Reference is a smoothed gaussian random field cast to uint16.
    Annotation is a 4-class quantile threshold of the same field cast
    to uint32. Shape follows BrainGlobe's (n_AP, n_DV, n_ML) layout.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape).astype(np.float32)
    smoothed = scipy.ndimage.gaussian_filter(raw, sigma=2.0)
    ref_min = float(smoothed.min())
    ref_span = float(smoothed.max() - ref_min) or 1.0
    reference = ((smoothed - ref_min) / ref_span * 50000).astype(np.uint16)

    quantiles = np.quantile(smoothed, [0.25, 0.5, 0.75])
    annotation = np.zeros(shape, dtype=np.uint32)
    annotation[smoothed > quantiles[0]] = 1
    annotation[smoothed > quantiles[1]] = 2
    annotation[smoothed > quantiles[2]] = 3
    return reference, annotation


class _FakeBrainGlobeAtlas:
    """In-memory stand-in for ``brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas``.

    Tests monkeypatch
    ``iss_preprocess.pipeline.tangential_atlas.bga.bg_atlas.BrainGlobeAtlas``
    with this class. ``call_count`` tracks how many bundles the
    real (faked) loader built, used by cache tests.
    """

    call_count = 0
    next_reference = None
    next_annotation = None
    next_resolution = (10.0, 10.0, 10.0)
    next_shape = None
    next_lookup_df = None

    def __init__(self, atlas_name):
        type(self).call_count += 1
        if type(self).next_reference is None or type(self).next_annotation is None:
            ref, ann = _synthetic_atlas_arrays()
            type(self).next_reference = ref
            type(self).next_annotation = ann
        self.atlas_name = atlas_name
        self.reference = type(self).next_reference
        self.annotation = type(self).next_annotation
        self.resolution = type(self).next_resolution
        self.shape = (
            type(self).next_shape
            if type(self).next_shape is not None
            else self.reference.shape
        )
        self.lookup_df = (
            type(self).next_lookup_df
            if type(self).next_lookup_df is not None
            else pd.DataFrame({"id": [0, 1, 2, 3], "name": ["bg", "a", "b", "c"]})
        )

    @classmethod
    def reset(cls):
        cls.call_count = 0
        cls.next_reference = None
        cls.next_annotation = None
        cls.next_resolution = (10.0, 10.0, 10.0)
        cls.next_shape = None
        cls.next_lookup_df = None


@pytest.fixture(autouse=False)
def fake_brainglobe(monkeypatch):
    """Install ``_FakeBrainGlobeAtlas`` in place of the real loader."""
    _FakeBrainGlobeAtlas.reset()
    monkeypatch.setattr(
        _tangential_atlas_mod.bga.bg_atlas,
        "BrainGlobeAtlas",
        _FakeBrainGlobeAtlas,
    )
    _clear_atlas_cache()
    yield _FakeBrainGlobeAtlas
    _clear_atlas_cache()
    _FakeBrainGlobeAtlas.reset()


def _make_axis_aligned_dv_spec(
    atlas, j_dv, out_shape_yx=None, order=1, cval=0.0
):
    """Build an AtlasPlaneSpec that samples ``reference[:, j_dv, :].T``."""
    res = atlas.resolution_um
    n_ap, _, n_ml = atlas.shape_zyx
    if out_shape_yx is None:
        out_shape_yx = (n_ml, n_ap)
    mat = np.array(
        [
            [res, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, float(j_dv) * res],
            [0.0, res, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return AtlasPlaneSpec(
        out_shape_yx=out_shape_yx,
        pixel_size_um=res,
        atlas_um_from_grid_4x4=mat,
        order=order,
        cval=cval,
    )


def _make_atlas_bundle(reference=None, annotation=None, resolution_um=10.0):
    if reference is None or annotation is None:
        reference, annotation = _synthetic_atlas_arrays()
    return AtlasBundle(
        atlas_name="allen_mouse_10um",
        resolution_um=resolution_um,
        reference=reference,
        annotation=annotation,
        lookup_df=pd.DataFrame({"id": [0, 1, 2, 3]}),
        shape_zyx=tuple(int(d) for d in reference.shape),
    )


# ---------------------------------------------------------------------------
# TestAtlasBundle
# ---------------------------------------------------------------------------


class TestAtlasBundle:
    def test_happy_path_construction(self):
        ref, ann = _synthetic_atlas_arrays()
        bundle = _make_atlas_bundle(ref, ann, 10.0)
        assert bundle.atlas_name == "allen_mouse_10um"
        assert bundle.resolution_um == 10.0
        assert bundle.shape_zyx == ref.shape
        assert bundle.reference is ref
        assert bundle.annotation is ann
        assert isinstance(bundle.lookup_df, pd.DataFrame)

    @pytest.mark.parametrize("which", ["reference", "annotation"])
    def test_rejects_wrong_ndim(self, which):
        ref, ann = _synthetic_atlas_arrays()
        kwargs = dict(
            atlas_name="allen_mouse_10um",
            resolution_um=10.0,
            reference=ref,
            annotation=ann,
            lookup_df=pd.DataFrame(),
            shape_zyx=tuple(int(d) for d in ref.shape),
        )
        kwargs[which] = np.zeros((5, 5))
        with pytest.raises(ValueError, match=which):
            AtlasBundle(**kwargs)

    def test_rejects_shape_mismatch(self):
        ref, ann = _synthetic_atlas_arrays()
        ann_wrong = ann[:, :-1, :].copy()
        with pytest.raises(ValueError, match="shape"):
            AtlasBundle(
                atlas_name="allen_mouse_10um",
                resolution_um=10.0,
                reference=ref,
                annotation=ann_wrong,
                lookup_df=pd.DataFrame(),
                shape_zyx=tuple(int(d) for d in ref.shape),
            )

    def test_rejects_non_integer_annotation_dtype(self):
        ref, _ = _synthetic_atlas_arrays()
        ann_float = np.zeros_like(ref, dtype=np.float64)
        with pytest.raises(ValueError, match="integer"):
            AtlasBundle(
                atlas_name="allen_mouse_10um",
                resolution_um=10.0,
                reference=ref,
                annotation=ann_float,
                lookup_df=pd.DataFrame(),
                shape_zyx=tuple(int(d) for d in ref.shape),
            )

    @pytest.mark.parametrize(
        "bad_resolution", [0.0, -1.0, float("inf"), float("nan")]
    )
    def test_rejects_nonpositive_resolution(self, bad_resolution):
        ref, ann = _synthetic_atlas_arrays()
        with pytest.raises(ValueError, match="resolution_um"):
            AtlasBundle(
                atlas_name="allen_mouse_10um",
                resolution_um=bad_resolution,
                reference=ref,
                annotation=ann,
                lookup_df=pd.DataFrame(),
                shape_zyx=tuple(int(d) for d in ref.shape),
            )

    def test_rejects_empty_atlas_name(self):
        ref, ann = _synthetic_atlas_arrays()
        with pytest.raises(ValueError, match="atlas_name"):
            AtlasBundle(
                atlas_name="",
                resolution_um=10.0,
                reference=ref,
                annotation=ann,
                lookup_df=pd.DataFrame(),
                shape_zyx=tuple(int(d) for d in ref.shape),
            )


# ---------------------------------------------------------------------------
# TestLoadAtlas
# ---------------------------------------------------------------------------


class TestLoadAtlas:
    def test_happy_path_returns_bundle(self, fake_brainglobe):
        bundle = load_atlas("allen_mouse_10um")
        assert isinstance(bundle, AtlasBundle)
        assert bundle.atlas_name == "allen_mouse_10um"
        assert bundle.resolution_um == 10.0
        assert bundle.reference.shape == bundle.shape_zyx
        assert bundle.annotation.shape == bundle.shape_zyx
        assert np.issubdtype(bundle.annotation.dtype, np.integer)

    @pytest.mark.parametrize("res_int", [5, 10, 25, 100])
    def test_parses_resolution_from_name(self, fake_brainglobe, res_int):
        _FakeBrainGlobeAtlas.next_resolution = (float(res_int),) * 3
        bundle = load_atlas(f"allen_mouse_{res_int}um")
        assert bundle.resolution_um == float(res_int)

    @pytest.mark.parametrize(
        "bad_name",
        ["allen_mouse", "allen_mouse_10", "allen_mouse_um", "", "10um_prefix"],
    )
    def test_rejects_malformed_atlas_name(self, bad_name):
        with pytest.raises(ValueError, match="_<int>um"):
            load_atlas(bad_name)

    def test_raises_on_resolution_mismatch(self, fake_brainglobe):
        _FakeBrainGlobeAtlas.next_resolution = (25.0, 25.0, 25.0)
        with pytest.raises(ValueError) as excinfo:
            load_atlas("allen_mouse_10um")
        msg = str(excinfo.value)
        assert "10" in msg and "25" in msg

    def test_raises_on_anisotropic_resolution(self, fake_brainglobe):
        _FakeBrainGlobeAtlas.next_resolution = (10.0, 10.0, 25.0)
        with pytest.raises(ValueError, match="anisotropic"):
            load_atlas("allen_mouse_10um")

    def test_cache_hit_skips_brainglobe_call(self, fake_brainglobe):
        load_atlas("allen_mouse_10um")
        load_atlas("allen_mouse_10um")
        assert _FakeBrainGlobeAtlas.call_count == 1

    def test_cache_disabled_per_call(self, fake_brainglobe):
        load_atlas("allen_mouse_10um", cache=False)
        load_atlas("allen_mouse_10um", cache=False)
        assert _FakeBrainGlobeAtlas.call_count == 2

    def test_cache_isolated_per_atlas_name(self, fake_brainglobe):
        # Build a 25um shape by trimming the synthetic arrays.
        ref10, ann10 = _synthetic_atlas_arrays()
        bundle10 = load_atlas("allen_mouse_10um")
        _FakeBrainGlobeAtlas.next_resolution = (25.0, 25.0, 25.0)
        _FakeBrainGlobeAtlas.next_reference = ref10
        _FakeBrainGlobeAtlas.next_annotation = ann10
        bundle25 = load_atlas("allen_mouse_25um")
        assert _FakeBrainGlobeAtlas.call_count == 2
        assert bundle10 is not bundle25
        assert bundle10.resolution_um == 10.0
        assert bundle25.resolution_um == 25.0

    @pytest.mark.parametrize(
        "arrays_kwarg",
        [
            ("reference", "annotation"),
            ("reference",),
            ("annotation",),
        ],
    )
    def test_arrays_kwarg_accepts_valid_subsets(
        self, fake_brainglobe, arrays_kwarg
    ):
        bundle = load_atlas("allen_mouse_10um", arrays=arrays_kwarg)
        # v1 always materialises both fields regardless of the kwarg.
        assert bundle.reference is not None
        assert bundle.annotation is not None

    def test_arrays_kwarg_rejects_unknown_member(self, fake_brainglobe):
        with pytest.raises(ValueError, match="reference"):
            load_atlas("allen_mouse_10um", arrays=("foo",))


# ---------------------------------------------------------------------------
# TestAtlasPlaneSpec
# ---------------------------------------------------------------------------


class TestAtlasPlaneSpec:
    def test_happy_path(self):
        mat = np.eye(4)
        spec = AtlasPlaneSpec(
            out_shape_yx=(8, 10),
            pixel_size_um=10.0,
            atlas_um_from_grid_4x4=mat,
            order=1,
            cval=0.0,
        )
        assert spec.out_shape_yx == (8, 10)
        assert spec.pixel_size_um == 10.0
        assert spec.atlas_um_from_grid_4x4.dtype == np.float64
        # Frozen + copied: caller mutation must not affect spec.
        mat[0, 3] = 999.0
        assert spec.atlas_um_from_grid_4x4[0, 3] == 0.0

    @pytest.mark.parametrize("bad_shape", [(3, 3), (4, 3), (4,), (4, 4, 4)])
    def test_rejects_wrong_matrix_shape(self, bad_shape):
        with pytest.raises(ValueError, match="atlas_um_from_grid"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=10.0,
                atlas_um_from_grid_4x4=np.zeros(bad_shape),
            )

    def test_rejects_last_row_not_homogeneous(self):
        bad = np.eye(4)
        bad[3, 0] = 0.5
        with pytest.raises(ValueError, match="last row"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=10.0,
                atlas_um_from_grid_4x4=bad,
            )

    @pytest.mark.parametrize("bad_order", [-1, 2, 3])
    def test_rejects_invalid_order(self, bad_order):
        with pytest.raises(ValueError, match="order"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=10.0,
                atlas_um_from_grid_4x4=np.eye(4),
                order=bad_order,
            )

    @pytest.mark.parametrize(
        "bad_shape",
        [(0, 5), (5, 0), (-1, 5), (5, -1), [5, 5], (5,), (5, 5, 5)],
    )
    def test_rejects_invalid_shape_tuple(self, bad_shape):
        with pytest.raises(ValueError, match="out_shape_yx"):
            AtlasPlaneSpec(
                out_shape_yx=bad_shape,
                pixel_size_um=10.0,
                atlas_um_from_grid_4x4=np.eye(4),
            )

    @pytest.mark.parametrize("bad_px", [0.0, -1.0])
    def test_rejects_nonpositive_pixel_size_um(self, bad_px):
        with pytest.raises(ValueError, match="pixel_size_um"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=bad_px,
                atlas_um_from_grid_4x4=np.eye(4),
            )

    @pytest.mark.parametrize("bad_px", [float("inf"), float("nan")])
    def test_rejects_nonfinite_pixel_size_um(self, bad_px):
        with pytest.raises(ValueError, match="pixel_size_um"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=bad_px,
                atlas_um_from_grid_4x4=np.eye(4),
            )

    @pytest.mark.parametrize("bad_cval", [float("inf"), float("nan")])
    def test_rejects_nonfinite_cval(self, bad_cval):
        with pytest.raises(ValueError, match="cval"):
            AtlasPlaneSpec(
                out_shape_yx=(5, 5),
                pixel_size_um=10.0,
                atlas_um_from_grid_4x4=np.eye(4),
                cval=bad_cval,
            )


# ---------------------------------------------------------------------------
# TestPlaneRenderConfig
# ---------------------------------------------------------------------------


class TestPlaneRenderConfig:
    def test_defaults_construct(self):
        cfg = PlaneRenderConfig()
        assert cfg.atlas_name == "allen_mouse_10um"
        assert cfg.atlas_resolution_um is None
        assert cfg.out_pixel_size_um is None
        assert cfg.out_shape_yx is None
        assert cfg.reference_order == 1
        assert cfg.annotation_order == 0
        assert cfg.cval_reference == 0.0
        assert cfg.cval_annotation == 0

    @pytest.mark.parametrize("bad_order", [-1, 2])
    def test_rejects_invalid_reference_order(self, bad_order):
        with pytest.raises(ValueError, match="reference_order"):
            PlaneRenderConfig(reference_order=bad_order)

    @pytest.mark.parametrize("bad_order", [-1, 2])
    def test_rejects_invalid_annotation_order(self, bad_order):
        with pytest.raises(ValueError, match="annotation_order"):
            PlaneRenderConfig(annotation_order=bad_order)

    def test_rejects_nonpositive_atlas_resolution_um(self):
        with pytest.raises(ValueError, match="atlas_resolution_um"):
            PlaneRenderConfig(atlas_resolution_um=0.0)

    def test_rejects_non_int_cval_annotation(self):
        with pytest.raises(ValueError, match="cval_annotation"):
            PlaneRenderConfig(cval_annotation=0.5)


# ---------------------------------------------------------------------------
# TestRenderReferencePlane
# ---------------------------------------------------------------------------


class TestRenderReferencePlane:
    def test_axis_aligned_identity_pose_renders_DV_constant_plane(self):
        """S §5 Slice 1 invariant 2 / P §1 acceptance bullet 2.

        Under the §3.0 identity-pose convention, a DV-constant atlas
        plane corresponds to the "zero pose, depth = j*res" rendering.
        With an integer-voxel-aligned spec, trilinear sampling reduces
        to exact and the rendered plane equals ``reference[:, j, :].T``.
        """
        bundle = _make_atlas_bundle()
        n_ap, n_dv, n_ml = bundle.shape_zyx
        j_dv = n_dv // 2
        spec = _make_axis_aligned_dv_spec(bundle, j_dv=j_dv, order=1)
        out = render_reference_plane(bundle, spec)
        expected = bundle.reference[:, j_dv, :].T
        assert out.shape == expected.shape == (n_ml, n_ap)
        np.testing.assert_allclose(
            out.astype(np.float64), expected.astype(np.float64), atol=1e-6
        )

    def test_dtype_preserved(self):
        bundle = _make_atlas_bundle()
        assert bundle.reference.dtype == np.uint16
        spec = _make_axis_aligned_dv_spec(bundle, j_dv=10, order=1)
        out = render_reference_plane(bundle, spec)
        assert out.dtype == np.uint16

    def test_out_of_bounds_returns_cval(self):
        bundle = _make_atlas_bundle()
        # Place the entire plane far outside the atlas (AP_um >> n_AP * res).
        res = bundle.resolution_um
        mat = np.eye(4)
        mat[0, 3] = 1e7  # AP offset
        spec = AtlasPlaneSpec(
            out_shape_yx=(6, 8),
            pixel_size_um=res,
            atlas_um_from_grid_4x4=mat,
            order=1,
            cval=7.0,
        )
        out = render_reference_plane(bundle, spec)
        assert out.shape == (6, 8)
        np.testing.assert_array_equal(
            out, np.full((6, 8), 7, dtype=bundle.reference.dtype)
        )

    @pytest.mark.parametrize("out_shape", [(8, 10), (16, 4)])
    def test_shape_matches_spec_out_shape_yx(self, out_shape):
        bundle = _make_atlas_bundle()
        spec = AtlasPlaneSpec(
            out_shape_yx=out_shape,
            pixel_size_um=bundle.resolution_um,
            atlas_um_from_grid_4x4=np.eye(4),
            order=1,
        )
        out = render_reference_plane(bundle, spec)
        assert out.shape == out_shape


# ---------------------------------------------------------------------------
# TestRenderAnnotationPlane
# ---------------------------------------------------------------------------


class TestRenderAnnotationPlane:
    def test_axis_aligned_identity_equals_atlas_slice(self):
        bundle = _make_atlas_bundle()
        n_ap, n_dv, n_ml = bundle.shape_zyx
        j_dv = n_dv // 2
        spec = _make_axis_aligned_dv_spec(
            bundle, j_dv=j_dv, order=0, cval=0.0
        )
        out = render_annotation_plane(bundle, spec)
        expected = bundle.annotation[:, j_dv, :].T
        assert out.dtype == bundle.annotation.dtype
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize(
        "ann_dtype", [np.uint8, np.uint16, np.uint32, np.int32]
    )
    def test_returns_integer_dtype_matching_atlas(self, ann_dtype):
        """S §5 Slice 1 invariant 4 / P §1 acceptance bullet 3."""
        ref, ann = _synthetic_atlas_arrays()
        ann_cast = ann.astype(ann_dtype)
        bundle = _make_atlas_bundle(ref, ann_cast)
        spec = _make_axis_aligned_dv_spec(bundle, j_dv=5, order=0)
        out = render_annotation_plane(bundle, spec)
        assert out.dtype == ann_dtype

    def test_output_values_subset_of_atlas_values(self):
        bundle = _make_atlas_bundle()
        spec = _make_axis_aligned_dv_spec(bundle, j_dv=20, order=0, cval=0.0)
        out = render_annotation_plane(bundle, spec)
        allowed = set(np.unique(bundle.annotation).tolist()) | {0}
        assert set(np.unique(out).tolist()) <= allowed

    def test_rejects_order_not_zero(self):
        bundle = _make_atlas_bundle()
        spec_bad = _make_axis_aligned_dv_spec(bundle, j_dv=5, order=1)
        with pytest.raises(ValueError, match="order == 0"):
            render_annotation_plane(bundle, spec_bad)

    def test_uses_nearest_not_trilinear(self):
        # Build an annotation with a sharp boundary at the AP midline.
        shape = (40, 30, 50)
        ann = np.zeros(shape, dtype=np.uint32)
        ann[shape[0] // 2 :, :, :] = 7
        ref = np.zeros(shape, dtype=np.uint16)
        bundle = AtlasBundle(
            atlas_name="allen_mouse_10um",
            resolution_um=10.0,
            reference=ref,
            annotation=ann,
            lookup_df=pd.DataFrame(),
            shape_zyx=shape,
        )
        # Sample at AP_um corresponding to a fractional voxel index of
        # 19.4 (still on the 0-side of the boundary at voxel 20).
        res = bundle.resolution_um
        ap_um = 19.4 * res
        mat = np.array(
            [
                [0.0, 0.0, 0.0, ap_um],
                [0.0, 0.0, 0.0, 5.0 * res],
                [0.0, res, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        spec = AtlasPlaneSpec(
            out_shape_yx=(shape[2], 1),
            pixel_size_um=res,
            atlas_um_from_grid_4x4=mat,
            order=0,
            cval=0,
        )
        out = render_annotation_plane(bundle, spec)
        # Nearest-neighbour at AP voxel 19 (closer to 19 than 20) → 0.
        assert set(np.unique(out).tolist()) == {0}


# ---------------------------------------------------------------------------
# TestRealBrainGlobeSmokePath — gated on local cache; never downloads.
# ---------------------------------------------------------------------------


def _atlas_is_locally_cached(name: str) -> bool:
    try:
        from pathlib import Path

        import brainglobe_atlasapi.config as cfg

        bg_dir = cfg.get_brainglobe_dir()
    except Exception:
        return False
    if bg_dir is None:
        return False
    p = Path(bg_dir)
    if not p.exists():
        return False
    return any(p.glob(f"{name}_v*"))


_REAL_ATLAS_NAME = "allen_mouse_25um"
_skip_if_no_cached_atlas = pytest.mark.skipif(
    not _atlas_is_locally_cached(_REAL_ATLAS_NAME),
    reason=f"real atlas {_REAL_ATLAS_NAME!r} not present in local BrainGlobe cache",
)


class TestRealBrainGlobeSmokePath:
    @_skip_if_no_cached_atlas
    def test_load_real_atlas_smoke(self):
        _clear_atlas_cache()
        bundle = load_atlas(_REAL_ATLAS_NAME)
        try:
            assert bundle.resolution_um == 25.0
            assert bundle.shape_zyx == bundle.reference.shape
            assert np.issubdtype(bundle.annotation.dtype, np.integer)
        finally:
            _clear_atlas_cache()

    @_skip_if_no_cached_atlas
    def test_render_real_atlas_smoke(self):
        _clear_atlas_cache()
        try:
            bundle = load_atlas(_REAL_ATLAS_NAME)
            spec = AtlasPlaneSpec(
                out_shape_yx=(16, 24),
                pixel_size_um=bundle.resolution_um,
                atlas_um_from_grid_4x4=np.eye(4),
                order=1,
            )
            out = render_reference_plane(bundle, spec)
            assert out.shape == (16, 24)
            assert out.dtype == bundle.reference.dtype
        finally:
            _clear_atlas_cache()


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("allen_mouse_5um", 5),
        ("allen_mouse_10um", 10),
        ("allen_mouse_25um", 25),
        ("allen_mouse_100um", 100),
    ],
)
def test_parse_resolution_from_atlas_name_valid(name, expected):
    assert _parse_resolution_from_atlas_name(name) == expected


@pytest.mark.parametrize(
    "bad_name",
    ["allen_mouse", "allen_mouse_10", "allen_mouse_um", "", "10um_prefix"],
)
def test_parse_resolution_from_atlas_name_invalid(bad_name):
    with pytest.raises(ValueError, match="_<int>um"):
        _parse_resolution_from_atlas_name(bad_name)


@pytest.mark.parametrize(
    "arrays",
    [
        ("reference",),
        ("annotation",),
        ("reference", "annotation"),
        ["annotation", "reference"],
    ],
)
def test_validate_arrays_kwarg_valid(arrays):
    out = _validate_arrays_kwarg(arrays)
    assert isinstance(out, tuple)
    assert set(out) <= {"reference", "annotation"}


@pytest.mark.parametrize(
    "bad",
    [("foo",), ("reference", "foo"), "reference"],  # third is a bare string
)
def test_validate_arrays_kwarg_invalid(bad):
    with pytest.raises(ValueError):
        _validate_arrays_kwarg(bad)


# ---------------------------------------------------------------------------
# M5 — slice geometry + per-slice plane spec
# ---------------------------------------------------------------------------


from iss_preprocess.pipeline.tangential_atlas import (  # noqa: E402
    SliceResidual,
    SliceStackGeometry,
    _compute_kept_slice_z_um,
    build_slice_plane_spec,
    build_slice_stack_geometry,
)


def _build_synthetic_npz(
    tmp_path,
    num_slices=8,
    bad=(),
    canvas_shape=(64, 80),
    chamber="proj/mouse/chamber_01",
):
    slice_numbers = np.arange(1, num_slices + 1, dtype=np.int64)
    rois = np.ones(num_slices, dtype=np.int64)
    data_paths = np.array([chamber] * num_slices, dtype=str)
    overview_files = np.array(["overview.tif"] * num_slices, dtype=str)
    canvas_offset = np.eye(3, dtype=np.float64)
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), (num_slices, 3, 3)).copy()
    out = tmp_path / "global_slice_transforms.npz"
    np.savez(
        out,
        reference_slice=np.int64(slice_numbers[num_slices // 2]),
        slice_numbers=slice_numbers,
        rois=rois,
        data_paths=data_paths,
        overview_files=overview_files,
        canvas_shape_yx=np.array(canvas_shape, dtype=np.int64),
        canvas_offset_matrix=canvas_offset,
        global_from_padded=eye,
        global_from_overview=eye,
        global_from_fullres=eye,
        bad_slices=np.array(list(bad), dtype=np.int64),
        auto_bad_slices=np.array([], dtype=np.int64),
    )
    return out


def _build_synthetic_stack(
    tmp_path,
    slice_numbers=(1, 2, 3, 4),
    *,
    overview_pixel_size_um=5.0,
    section_thickness_um=None,
    absolute_sections=None,
    duplicate_slice_number=None,
    target_shape=(6, 7),
):
    images = np.zeros((len(slice_numbers),) + tuple(target_shape), dtype=np.float32)
    entries = []
    for image_index, sn in enumerate(slice_numbers):
        stored_sn = int(sn)
        if duplicate_slice_number is not None and image_index == len(slice_numbers) - 1:
            stored_sn = int(duplicate_slice_number)
        images[image_index, :, :] = float(stored_sn)
        entry = {
            "slice_number": stored_sn,
            "roi": int(stored_sn),
            "data_path": f"proj/mouse/chamber_{stored_sn:02d}",
            "overview_file": f"overview_sl{stored_sn:03d}.tif",
            "overview_shape_yx": list(target_shape),
            "original_shape_yx": list(target_shape),
            "downsample_ratio": 1.0,
            "overview_pixel_size_um": float(overview_pixel_size_um),
        }
        if section_thickness_um is not None:
            entry["section_thickness_um"] = float(section_thickness_um)
        if absolute_sections is not None:
            entry["absolute_section"] = float(absolute_sections[image_index])
        entries.append(entry)
    manifest = {"target_shape_yx": list(target_shape), "entries": entries}
    path = tmp_path / "unregistered_slices.npz"
    np.savez(
        path,
        images=images,
        default_masks=np.ones_like(images, dtype=np.uint8),
        user_masks=np.ones_like(images, dtype=np.uint8),
        manifest_json=np.array(json.dumps(manifest)),
    )
    return path


def _build_context_transforms(
    tmp_path,
    *,
    slice_numbers=(1, 3, 4),
    bad=(2,),
    canvas_shape=(6, 7),
):
    slice_numbers = np.asarray(slice_numbers, dtype=np.int64)
    rois = slice_numbers.astype(np.int64)
    data_paths = np.array(
        [f"proj/mouse/chamber_{int(sn):02d}" for sn in slice_numbers],
        dtype=str,
    )
    overview_files = np.array(
        [f"overview_sl{int(sn):03d}.tif" for sn in slice_numbers],
        dtype=str,
    )
    eye = np.broadcast_to(
        np.eye(3, dtype=np.float64), (len(slice_numbers), 3, 3)
    ).copy()
    out = tmp_path / "global_slice_transforms.npz"
    np.savez(
        out,
        reference_slice=np.int64(slice_numbers[0]),
        slice_numbers=slice_numbers,
        rois=rois,
        data_paths=data_paths,
        overview_files=overview_files,
        canvas_shape_yx=np.array(canvas_shape, dtype=np.int64),
        canvas_offset_matrix=np.eye(3),
        global_from_padded=eye,
        global_from_overview=eye,
        global_from_fullres=eye,
        bad_slices=np.array(list(bad), dtype=np.int64),
        auto_bad_slices=np.array([], dtype=np.int64),
    )
    return out


class TestSliceResidual:
    def test_happy_path_zero_residual(self):
        r = SliceResidual(slice_number=1)
        assert r.slice_number == 1
        assert r.dx_um == 0.0 and r.dy_um == 0.0
        assert r.dtheta_deg == 0.0 and r.dz_um == 0.0

    @pytest.mark.parametrize("field", ["dx_um", "dy_um", "dtheta_deg"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite(self, field, bad):
        kwargs = dict(slice_number=1, dx_um=0.0, dy_um=0.0, dtheta_deg=0.0, dz_um=0.0)
        kwargs[field] = bad
        with pytest.raises(ValueError, match=field):
            SliceResidual(**kwargs)

    def test_rejects_nonzero_dz(self):
        with pytest.raises(ValueError, match="dz_um"):
            SliceResidual(slice_number=1, dz_um=0.5)

    def test_accepts_numpy_int_slice_number(self):
        r = SliceResidual(slice_number=np.int64(7))
        assert r.slice_number == 7 and isinstance(r.slice_number, int)

    def test_rejects_bool_slice_number(self):
        with pytest.raises(ValueError, match="slice_number"):
            SliceResidual(slice_number=True)


def _make_geometry(M=4, px=5.0, canvas=(64, 80), bad=()):
    sn = np.arange(1, M + 1, dtype=np.int64)
    if bad:
        kept_mask = np.array([s not in set(bad) for s in sn.tolist()], dtype=bool)
        sn_kept = sn[kept_mask]
    else:
        sn_kept = sn
    M2 = sn_kept.size
    return SliceStackGeometry(
        slice_numbers=sn_kept,
        rois=np.ones(M2, dtype=np.int64),
        chambers=tuple(f"chamber_{i}" for i in range(M2)),
        z_um=(sn_kept - sn_kept[0]).astype(np.float64) * 50.0,
        overview_pixel_size_um=px,
        canvas_shape_yx=canvas,
        global_from_overview=np.broadcast_to(np.eye(3), (M2, 3, 3)).copy(),
        bad_slice_numbers=np.array(list(bad), dtype=np.int64),
    )


class TestSliceStackGeometry:
    def test_happy_path(self):
        g = _make_geometry(M=4)
        assert g.slice_numbers.tolist() == [1, 2, 3, 4]
        assert g.z_um.tolist() == [0.0, 50.0, 100.0, 150.0]
        assert g.overview_pixel_size_um == 5.0

    def test_rejects_non_ascending_slice_numbers(self):
        with pytest.raises(ValueError, match="ascending"):
            SliceStackGeometry(
                slice_numbers=np.array([3, 1, 2], dtype=np.int64),
                rois=np.zeros(3, dtype=np.int64),
                chambers=("a", "b", "c"),
                z_um=np.array([0.0, 50.0, 100.0]),
                overview_pixel_size_um=5.0,
                canvas_shape_yx=(64, 80),
                global_from_overview=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
                bad_slice_numbers=np.array([], dtype=np.int64),
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="z_um shape"):
            SliceStackGeometry(
                slice_numbers=np.array([1, 2, 3], dtype=np.int64),
                rois=np.zeros(3, dtype=np.int64),
                chambers=("a", "b", "c"),
                z_um=np.array([0.0, 50.0]),  # wrong length
                overview_pixel_size_um=5.0,
                canvas_shape_yx=(64, 80),
                global_from_overview=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
                bad_slice_numbers=np.array([], dtype=np.int64),
            )

    def test_rejects_bad_in_kept(self):
        with pytest.raises(ValueError, match="disjoint"):
            SliceStackGeometry(
                slice_numbers=np.array([1, 2, 3], dtype=np.int64),
                rois=np.zeros(3, dtype=np.int64),
                chambers=("a", "b", "c"),
                z_um=np.array([0.0, 50.0, 100.0]),
                overview_pixel_size_um=5.0,
                canvas_shape_yx=(64, 80),
                global_from_overview=np.broadcast_to(np.eye(3), (3, 3, 3)).copy(),
                bad_slice_numbers=np.array([2], dtype=np.int64),
            )

    def test_rejects_nonpositive_overview_pixel_size_um(self):
        with pytest.raises(ValueError, match="overview_pixel_size_um"):
            SliceStackGeometry(
                slice_numbers=np.array([1], dtype=np.int64),
                rois=np.zeros(1, dtype=np.int64),
                chambers=("a",),
                z_um=np.array([0.0]),
                overview_pixel_size_um=0.0,
                canvas_shape_yx=(64, 80),
                global_from_overview=np.broadcast_to(np.eye(3), (1, 3, 3)).copy(),
                bad_slice_numbers=np.array([], dtype=np.int64),
            )

    def test_rejects_invalid_global_from_overview_last_row(self):
        gfo = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        gfo[1, 2, 0] = 0.5  # break last-row homogeneity
        with pytest.raises(ValueError, match="last row"):
            SliceStackGeometry(
                slice_numbers=np.array([1, 2], dtype=np.int64),
                rois=np.zeros(2, dtype=np.int64),
                chambers=("a", "b"),
                z_um=np.array([0.0, 50.0]),
                overview_pixel_size_um=5.0,
                canvas_shape_yx=(64, 80),
                global_from_overview=gfo,
                bad_slice_numbers=np.array([], dtype=np.int64),
            )


class TestBuildSliceStackGeometry:
    def test_loads_8_slices_with_constant_thickness(self, tmp_path):
        npz = _build_synthetic_npz(tmp_path, num_slices=8)
        g = build_slice_stack_geometry(
            "ignored_data_path",
            transforms_path=npz,
            section_thickness_um=50.0,
            overview_pixel_size_um=5.0,
        )
        assert g.slice_numbers.tolist() == list(range(1, 9))
        np.testing.assert_allclose(g.z_um, np.arange(8) * 50.0)
        assert g.overview_pixel_size_um == 5.0
        assert g.canvas_shape_yx == (64, 80)

    def test_bad_slice_creates_gap(self, tmp_path):
        npz = _build_synthetic_npz(tmp_path, num_slices=8, bad=[3])
        g = build_slice_stack_geometry(
            "ignored",
            transforms_path=npz,
            section_thickness_um=50.0,
            overview_pixel_size_um=5.0,
        )
        assert g.slice_numbers.tolist() == [1, 2, 4, 5, 6, 7, 8]
        # z_um: slice s → (s - 1) * 50.0
        np.testing.assert_allclose(
            g.z_um, np.array([0, 50, 150, 200, 250, 300, 350], dtype=float)
        )
        diffs = np.diff(g.z_um)
        assert diffs[1] == 100.0  # the gap from kept slice 2 → kept slice 4
        assert g.bad_slice_numbers.tolist() == [3]

    def test_state_bad_slices_override_npz(self, tmp_path):
        npz = _build_synthetic_npz(tmp_path, num_slices=5, bad=[2])
        # state's bad_slices override the npz's
        g = build_slice_stack_geometry(
            "ignored",
            transforms_path=npz,
            state={"bad_slices": [4]},
            section_thickness_um=50.0,
            overview_pixel_size_um=5.0,
        )
        assert g.slice_numbers.tolist() == [1, 2, 3, 5]
        assert g.bad_slice_numbers.tolist() == [4]

    def test_overview_pixel_size_mismatch_raises(self, tmp_path, monkeypatch):
        # build an npz with two distinct chambers, then mock load_ops to
        # report different pixel sizes per chamber.
        slice_numbers = np.array([1, 2, 3, 4], dtype=np.int64)
        rois = np.ones(4, dtype=np.int64)
        data_paths = np.array(
            [
                "proj/mouse/chamber_01",
                "proj/mouse/chamber_01",
                "proj/mouse/chamber_02",
                "proj/mouse/chamber_02",
            ],
            dtype=str,
        )
        eye = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
        npz = tmp_path / "global_slice_transforms.npz"
        np.savez(
            npz,
            reference_slice=np.int64(2),
            slice_numbers=slice_numbers,
            rois=rois,
            data_paths=data_paths,
            overview_files=np.array(["o"] * 4),
            canvas_shape_yx=np.array([64, 80], dtype=np.int64),
            canvas_offset_matrix=np.eye(3),
            global_from_padded=eye,
            global_from_overview=eye,
            global_from_fullres=eye,
            bad_slices=np.array([], dtype=np.int64),
            auto_bad_slices=np.array([], dtype=np.int64),
        )

        def fake_load_ops(path, warn_missing=True):
            return {
                "overview_pixel_size_um": (
                    5.0 if "chamber_01" in str(path) else 10.0
                )
            }

        monkeypatch.setattr("iss_preprocess.io.load.load_ops", fake_load_ops)
        with pytest.raises(ValueError, match="differs"):
            build_slice_stack_geometry(
                "ignored",
                transforms_path=npz,
                section_thickness_um=50.0,
            )

    def test_explicit_z_um_by_slice_overrides_scalar_spacing(self, tmp_path):
        npz = _build_synthetic_npz(tmp_path, num_slices=4, bad=[2])
        g = build_slice_stack_geometry(
            "ignored",
            transforms_path=npz,
            section_thickness_um=50.0,
            overview_pixel_size_um=5.0,
            z_um_by_slice={1: 0.0, 3: 120.0, 4: 160.0},
        )
        assert g.slice_numbers.tolist() == [1, 3, 4]
        np.testing.assert_allclose(g.z_um, [0.0, 120.0, 160.0])

    def test_explicit_z_um_by_slice_requires_kept_slices(self, tmp_path):
        npz = _build_synthetic_npz(tmp_path, num_slices=3)
        with pytest.raises(ValueError, match="missing kept slice"):
            build_slice_stack_geometry(
                "ignored",
                transforms_path=npz,
                overview_pixel_size_um=5.0,
                z_um_by_slice={1: 0.0, 2: 50.0},
            )


class TestTangentialAtlasContext:
    def test_maps_kept_slice_number_to_raw_image_index_with_bad_prior_slice(self, tmp_path):
        stack = _build_synthetic_stack(tmp_path, slice_numbers=(1, 2, 3, 4))
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1, 3, 4), bad=(2,)
        )
        ctx = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
        )
        assert isinstance(ctx, TangentialAtlasContext)
        assert ctx.geometry.slice_numbers.tolist() == [1, 3, 4]
        assert [r.image_index for r in ctx.slice_records] == [0, 2, 3]
        np.testing.assert_allclose(ctx.geometry.z_um, [0.0, 40.0, 60.0])
        assert image_for_tangential_atlas_slice(ctx, 3)[0, 0] == 3.0
        codes = {issue.code for issue in ctx.issues}
        assert "section_thickness_fallback_20um" in codes
        assert "absolute_section_unavailable" in codes
        assert ctx.metadata.section_thickness_um == DEFAULT_TANGENTIAL_ATLAS_SECTION_THICKNESS_UM
        assert ctx.metadata.z_source == "slice_number_spacing"

    def test_duplicate_manifest_slice_number_is_hard_error(self, tmp_path):
        stack = _build_synthetic_stack(
            tmp_path,
            slice_numbers=(1, 2),
            duplicate_slice_number=1,
        )
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1,), bad=()
        )
        with pytest.raises(ValueError, match="duplicate_manifest_slice_number"):
            build_tangential_atlas_context(
                tmp_path,
                stack_path=stack,
                transforms_path=transforms,
            )

    def test_missing_manifest_image_for_active_slice_is_hard_error(self, tmp_path):
        stack = _build_synthetic_stack(tmp_path, slice_numbers=(1, 2))
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1, 3), bad=()
        )
        with pytest.raises(ValueError, match="active_slice_missing_manifest_image"):
            build_tangential_atlas_context(
                tmp_path,
                stack_path=stack,
                transforms_path=transforms,
            )

    def test_image_lookup_rejects_unmapped_slice(self, tmp_path):
        stack = _build_synthetic_stack(tmp_path, slice_numbers=(1, 2, 3))
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1, 3), bad=(2,)
        )
        ctx = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
        )
        with pytest.raises(ValueError, match="not mapped"):
            image_for_tangential_atlas_slice(ctx, 2)

    def test_manifest_absolute_section_drives_physical_z(self, tmp_path):
        stack = _build_synthetic_stack(
            tmp_path,
            slice_numbers=(1, 2, 3, 4),
            section_thickness_um=25.0,
            absolute_sections=(10, 11, 13, 14),
        )
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1, 3, 4), bad=(2,)
        )
        ctx = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
        )
        assert ctx.metadata.section_thickness_source == "manifest.section_thickness_um"
        assert ctx.metadata.z_source == "manifest.absolute_section"
        np.testing.assert_allclose(ctx.geometry.z_um, [0.0, 75.0, 100.0])
        assert not any(issue.severity == "error" for issue in ctx.issues)

    def test_state_z_um_by_slice_drives_geometry(self, tmp_path):
        stack = _build_synthetic_stack(tmp_path, slice_numbers=(1, 2, 3, 4))
        transforms = _build_context_transforms(
            tmp_path, slice_numbers=(1, 3, 4), bad=(2,)
        )
        state = {
            "bad_slices": [2],
            "overview_pixel_size_um": 5.0,
            "section_thickness_um": 20.0,
            "z_um_by_slice": {"1": 0.0, "3": 80.0, "4": 100.0},
        }
        ctx = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
            state=state,
        )
        assert ctx.metadata.z_source == "state.z_um_by_slice"
        np.testing.assert_allclose(ctx.geometry.z_um, [0.0, 80.0, 100.0])

    def test_public_pipeline_re_exports_context_api(self):
        import iss_preprocess.pipeline as pipeline

        assert pipeline.TangentialAtlasIssue is TangentialAtlasIssue
        assert pipeline.build_tangential_atlas_context is build_tangential_atlas_context
        assert pipeline.image_for_tangential_atlas_slice is image_for_tangential_atlas_slice


class TestBuildSlicePlaneSpec:
    def test_zero_pose_returns_DV_constant_at_z(self):
        # Under §3.0, P sends y_tan → -ML. To get a clean DV-constant
        # rendered plane equal to reference[:, j_dv, :].T (no axis
        # flips in the expected), put a y-flip into global_from_overview
        # so the canvas (row) maps to -y_tan, which P then sends to +ML.
        # The chain: col,row → (col*px, row*px, 0, 1) [S]
        #          → (col*px, -row*px, 0, 1) [M_embed with y-flip]
        #          → (col*px, -row*px, z, 1) [T_z]
        #          → (col*px, z, row*px, 1) [P]
        # → atlas voxel (col, z/res, row) = reference[col, j_dv, row].
        bundle = _make_atlas_bundle()  # (60, 80, 100) shape, 10um res
        res = bundle.resolution_um
        n_ap, n_dv, n_ml = bundle.shape_zyx
        j_dv = n_dv // 2
        sn = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        z_um = (sn - 1).astype(np.float64) * (j_dv * res / 2.0)
        yflip = np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        gfo = np.broadcast_to(yflip[None, :, :], (5, 3, 3)).copy()
        geom = SliceStackGeometry(
            slice_numbers=sn,
            rois=np.ones(5, dtype=np.int64),
            chambers=tuple(f"c{i}" for i in range(5)),
            z_um=z_um,
            overview_pixel_size_um=res,
            canvas_shape_yx=(n_ml, n_ap),
            global_from_overview=gfo,
            bad_slice_numbers=np.array([], dtype=np.int64),
        )
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        spec = build_slice_plane_spec(geom, pose, slice_index=2)
        assert isinstance(spec, AtlasPlaneSpec)
        out = render_reference_plane(bundle, spec)
        expected = bundle.reference[:, j_dv, :].T  # (n_ml, n_ap)
        np.testing.assert_allclose(
            out.astype(np.float64), expected.astype(np.float64), atol=1e-6
        )

    def test_kept_list_indexing(self):
        # With bad=[3], geometry.slice_numbers = [1, 2, 4, 5]; slice_index=2
        # corresponds to global slice 4.
        geom = _make_geometry(M=5, bad=(3,))
        assert geom.slice_numbers.tolist() == [1, 2, 4, 5]
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        # slice_index=2 → global slice 4 → z = (4-1)*50 = 150
        spec = build_slice_plane_spec(geom, pose, slice_index=2)
        # The composed matrix's z-translation row should equal
        # pose_to_matrix(...)[:3,:3] @ [0, 0, z, 0] + atlas-frame translation.
        # Just verify the spec is constructed; matrix arithmetic is covered
        # by the zero-pose invariant test above.
        assert isinstance(spec, AtlasPlaneSpec)

    @pytest.mark.parametrize("bad_index", [-1, 4, 100])
    def test_rejects_out_of_range_slice_index(self, bad_index):
        geom = _make_geometry(M=4)
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        with pytest.raises(ValueError, match="slice_index"):
            build_slice_plane_spec(geom, pose, slice_index=bad_index)

    def test_residual_dz_must_be_zero(self):
        # The dataclass guard fires at construction.
        with pytest.raises(ValueError, match="dz_um"):
            SliceResidual(slice_number=1, dz_um=1.0)


# ---------------------------------------------------------------------------
# M6 — RegistrationObjectiveConfig, residual composition, state IO
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402

from iss_preprocess.pipeline.tangential_atlas import (  # noqa: E402
    RegistrationObjectiveConfig,
    TANGENTIAL_ATLAS_STATE_VERSION,
    center_residuals,
    compose_pose_with_residual,
    load_tangential_atlas_state,
    save_tangential_atlas_state,
    validate_residual_against_bounds,
)


class TestRegistrationObjectiveConfig:
    def test_defaults(self):
        c = RegistrationObjectiveConfig()
        assert c.max_dx_um == 250.0 and c.max_dy_um == 250.0
        assert c.max_dtheta_deg == 2.0 and c.max_dz_um == 0.0
        assert c.smoothness_lambda == 0.0
        assert c.centering == "mean_zero"

    @pytest.mark.parametrize(
        "field", ["max_dx_um", "max_dy_um", "max_dtheta_deg", "smoothness_lambda"]
    )
    def test_rejects_negative(self, field):
        kwargs = dict(max_dx_um=10.0, max_dy_um=10.0, max_dtheta_deg=1.0,
                      max_dz_um=0.0, smoothness_lambda=0.0)
        kwargs[field] = -1.0
        with pytest.raises(ValueError, match=field):
            RegistrationObjectiveConfig(**kwargs)

    def test_rejects_nonzero_max_dz_um(self):
        with pytest.raises(ValueError, match="max_dz_um"):
            RegistrationObjectiveConfig(max_dz_um=1.0)

    def test_rejects_invalid_centering(self):
        with pytest.raises(ValueError, match="centering"):
            RegistrationObjectiveConfig(centering="weighted")


class TestValidateResidualAgainstBounds:
    def test_in_bounds_passes(self):
        validate_residual_against_bounds(
            SliceResidual(slice_number=1, dx_um=10.0, dy_um=10.0, dtheta_deg=0.5),
            RegistrationObjectiveConfig(),
        )

    def test_exactly_at_bound_passes(self):
        validate_residual_against_bounds(
            SliceResidual(slice_number=1, dx_um=250.0, dy_um=-250.0, dtheta_deg=2.0),
            RegistrationObjectiveConfig(),
        )

    def test_over_dx_raises(self):
        with pytest.raises(ValueError, match="dx_um"):
            validate_residual_against_bounds(
                SliceResidual(slice_number=1, dx_um=300.0),
                RegistrationObjectiveConfig(),
            )

    def test_over_dtheta_raises(self):
        with pytest.raises(ValueError, match="dtheta_deg"):
            validate_residual_against_bounds(
                SliceResidual(slice_number=1, dtheta_deg=10.0),
                RegistrationObjectiveConfig(),
            )


class TestComposePoseWithResidual:
    def test_zero_residual_matches_build_slice_plane_spec_chain(self):
        geom = _make_geometry(M=3)
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        # At zero residual, compose_pose_with_residual should equal the
        # pose's atlas_from_tangential.
        r = SliceResidual(slice_number=int(geom.slice_numbers[1]))
        out = compose_pose_with_residual(pose, geom, r, slice_index=1)
        np.testing.assert_allclose(out, pose.atlas_from_tangential, atol=1e-12)

    def test_nonzero_residual_shifts_translation_block(self):
        geom = _make_geometry(M=3)
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        r = SliceResidual(slice_number=int(geom.slice_numbers[0]),
                          dx_um=5.0, dy_um=-7.0)
        out = compose_pose_with_residual(pose, geom, r, slice_index=0)
        # The composed matrix applies T_inplane after P; the translation
        # column of the composed 4×4 should be non-zero.
        assert not np.allclose(out[:3, 3], 0.0)

    @pytest.mark.parametrize("bad_index", [-1, 3, 100])
    def test_rejects_out_of_range_slice_index(self, bad_index):
        geom = _make_geometry(M=3)
        pose = StackPose(
            atlas_from_tangential=pose_to_matrix(0, 0, 0, 0, 0, 0),
            yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
            depth_um=0.0, tx_atlas_um=0.0, ty_atlas_um=0.0,
        )
        r = SliceResidual(slice_number=1)
        with pytest.raises(ValueError, match="slice_index"):
            compose_pose_with_residual(pose, geom, r, slice_index=bad_index)


class TestCenterResiduals:
    def test_mean_zero_centers(self):
        r = {
            1: SliceResidual(slice_number=1, dx_um=2.0, dy_um=4.0, dtheta_deg=0.5),
            2: SliceResidual(slice_number=2, dx_um=4.0, dy_um=6.0, dtheta_deg=1.5),
        }
        out = center_residuals(r, rule="mean_zero")
        means = (
            np.mean([v.dx_um for v in out.values()]),
            np.mean([v.dy_um for v in out.values()]),
            np.mean([v.dtheta_deg for v in out.values()]),
        )
        for m in means:
            assert abs(m) < 1e-12

    def test_rule_none_is_identity(self):
        r = {1: SliceResidual(slice_number=1, dx_um=2.0)}
        out = center_residuals(r, rule="none")
        assert out[1].dx_um == 2.0

    def test_empty_input(self):
        assert center_residuals({}) == {}

    def test_rejects_invalid_rule(self):
        with pytest.raises(ValueError, match="rule"):
            center_residuals({}, rule="weighted")


def _make_state_payload():
    P_4x4_list = _P_4x4.tolist()
    return {
        "version": TANGENTIAL_ATLAS_STATE_VERSION,
        "atlas_name": "allen_mouse_10um",
        "atlas_resolution_um": 10.0,
        "stack_path": "/tmp/fake/tangential_volume/unregistered_slices.npz",
        "global_transforms_path":
            "/tmp/fake/tangential_volume/global_slice_transforms.npz",
        "canvas_shape_yx": [64, 80],
        "overview_pixel_size_um": 5.0,
        "bad_slices": [],
        "pose": {
            "atlas_from_tangential_4x4": P_4x4_list,
            "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
            "depth_um": 0.0, "tx_atlas_um": 0.0, "ty_atlas_um": 0.0,
            "rotation_order": "ZYX",
            "axes": {"yaw": "DV", "pitch": "ML", "roll": "AP"},
        },
        "residuals": {
            "1": {"dx_um": 1.0, "dy_um": -2.0, "dtheta_deg": 0.5, "dz_um": 0.0},
        },
        "residual_bounds": {
            "max_dx_um": 250.0, "max_dy_um": 250.0,
            "max_dtheta_deg": 2.0, "max_dz_um": 0.0,
            "smoothness_lambda": 0.0, "centering": "mean_zero",
        },
    }


class TestStateIO:
    def test_round_trip(self, tmp_path):
        payload = _make_state_payload()
        state_path = tmp_path / "tangential_atlas_state.json"
        save_tangential_atlas_state(payload, state_path=state_path)
        loaded = load_tangential_atlas_state(tmp_path, state_path=state_path)
        # Compare the JSON-serialised form (round-trip identity).
        assert loaded["atlas_name"] == payload["atlas_name"]
        assert loaded["residuals"]["1"]["dx_um"] == 1.0
        assert loaded["version"] == TANGENTIAL_ATLAS_STATE_VERSION

    def test_load_accepts_optional_real_data_geometry_fields(self, tmp_path):
        payload = _make_state_payload()
        payload["section_thickness_um"] = 20.0
        payload["z_um_by_slice"] = {"1": 0.0, "3": 40.0}
        state_path = tmp_path / "tangential_atlas_state.json"
        save_tangential_atlas_state(payload, state_path=state_path)
        loaded = load_tangential_atlas_state(tmp_path, state_path=state_path)
        assert loaded["section_thickness_um"] == 20.0
        assert loaded["z_um_by_slice"]["3"] == 40.0

    def test_load_rejects_wrong_version(self, tmp_path):
        payload = _make_state_payload()
        payload["version"] = 99
        state_path = tmp_path / "tangential_atlas_state.json"
        with open(state_path, "w") as f:
            _json.dump(payload, f)
        with pytest.raises(ValueError, match="version"):
            load_tangential_atlas_state(tmp_path, state_path=state_path)

    def test_load_rejects_non_zyx_rotation_order(self, tmp_path):
        payload = _make_state_payload()
        payload["pose"]["rotation_order"] = "XYZ"
        state_path = tmp_path / "tangential_atlas_state.json"
        with open(state_path, "w") as f:
            _json.dump(payload, f)
        with pytest.raises(ValueError, match="rotation_order"):
            load_tangential_atlas_state(tmp_path, state_path=state_path)

    def test_load_rejects_nonrigid_pose(self, tmp_path):
        payload = _make_state_payload()
        # Make the pose 4×4 non-rigid (det != 1).
        payload["pose"]["atlas_from_tangential_4x4"] = (
            (2 * np.eye(4)).tolist()
        )
        # Keep the last row valid so we fail at the rigidity check.
        payload["pose"]["atlas_from_tangential_4x4"][3] = [0, 0, 0, 1]
        state_path = tmp_path / "tangential_atlas_state.json"
        with open(state_path, "w") as f:
            _json.dump(payload, f)
        with pytest.raises(ValueError, match="rotation"):
            load_tangential_atlas_state(tmp_path, state_path=state_path)

    def test_load_rejects_nonzero_dz(self, tmp_path):
        payload = _make_state_payload()
        payload["residuals"]["1"]["dz_um"] = 0.5
        state_path = tmp_path / "tangential_atlas_state.json"
        with open(state_path, "w") as f:
            _json.dump(payload, f)
        with pytest.raises(ValueError, match="dz_um"):
            load_tangential_atlas_state(tmp_path, state_path=state_path)

    def test_save_makes_bak_on_overwrite(self, tmp_path):
        payload = _make_state_payload()
        state_path = tmp_path / "tangential_atlas_state.json"
        save_tangential_atlas_state(payload, state_path=state_path)
        # Save again — the previous file should move to .bak.
        save_tangential_atlas_state(payload, state_path=state_path)
        assert (tmp_path / "tangential_atlas_state.json.bak").exists()


# ---------------------------------------------------------------------------
# M7 — dz_um (G2-gated) + objective functions
# ---------------------------------------------------------------------------


from iss_preprocess.pipeline.tangential_atlas import (  # noqa: E402
    assert_monotonic_spacing,
    monotonic_dz_from_raw,
    slice_objective,
)


class TestMonotonicDzFromRaw:
    def test_zero_input_zero_output(self):
        out = monotonic_dz_from_raw(np.zeros(8))
        np.testing.assert_allclose(out, np.zeros(8), atol=1e-12)

    def test_strictly_increasing_under_base_plus_dz(self):
        rng = np.random.default_rng(0)
        raw = rng.standard_normal(10)
        dz = monotonic_dz_from_raw(raw)
        base = np.arange(10, dtype=np.float64) * 50.0
        effective = base + dz
        assert np.all(np.diff(effective) > 0)

    def test_empty_input(self):
        out = monotonic_dz_from_raw(np.zeros(0))
        assert out.shape == (0,)

    def test_single_element_input(self):
        out = monotonic_dz_from_raw(np.array([3.5]))
        np.testing.assert_array_equal(out, np.zeros(1))


class TestAssertMonotonicSpacing:
    def test_default_gate_returns_empty(self):
        geom = _make_geometry(M=5)
        residuals = {
            int(sn): SliceResidual(slice_number=int(sn))
            for sn in geom.slice_numbers.tolist()
        }
        warnings = assert_monotonic_spacing(geom, residuals)
        assert warnings == []

    def test_warns_on_crossing_when_gate_open(self):
        # Manually break the SliceResidual invariant by building a non-dataclass
        # stand-in: use a SimpleNamespace with dz_um attribute, mimicking the
        # frozen dataclass surface but allowing dz_um != 0.
        from types import SimpleNamespace

        geom = _make_geometry(M=3)  # z_um = [0, 50, 100]
        residuals = {
            int(geom.slice_numbers[0]): SimpleNamespace(slice_number=1, dz_um=0.0),
            int(geom.slice_numbers[1]): SimpleNamespace(slice_number=2, dz_um=100.0),
            int(geom.slice_numbers[2]): SimpleNamespace(slice_number=3, dz_um=0.0),
        }
        # With dz=[0, 100, 0] applied, effective_z = [0, 150, 100] → crosses.
        bounds_open = RegistrationObjectiveConfig()
        # Force the gate open by constructing a new bounds object with
        # max_dz_um > 0 (cannot via dataclass — G2 gate). Bypass via __dict__
        # set on a fresh instance? RegistrationObjectiveConfig is frozen.
        # Use a mock-like object instead.
        bounds = SimpleNamespace(max_dz_um=1000.0)
        warnings = assert_monotonic_spacing(geom, residuals, bounds=bounds)
        assert len(warnings) >= 1
        assert "non-monotonic" in warnings[0]


class TestSliceObjective:
    def test_ncc_self_correlation_is_one(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((16, 16)).astype(np.float64)
        assert abs(slice_objective(x, x, mode="ncc") - 1.0) < 1e-6

    def test_ncc_negative_correlation(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((16, 16))
        assert abs(slice_objective(x, -x, mode="ncc") + 1.0) < 1e-6

    def test_mi_nonneg_and_self_high(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal((32, 32))
        y = rng.standard_normal((32, 32))
        mi_self = slice_objective(x, x, mode="mi")
        mi_ind = slice_objective(x, y, mode="mi")
        assert mi_self >= 0 and mi_ind >= 0
        assert mi_self > mi_ind

    def test_edge_self_match_perfect(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((16, 16))
        assert abs(slice_objective(x, x, mode="edge") - 1.0) < 1e-6

    def test_mask_is_respected(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((16, 16))
        y = rng.standard_normal((16, 16))
        mask = np.zeros_like(x, dtype=bool)
        mask[2:6, 2:6] = True
        # NCC over the masked region should differ from NCC over the whole.
        full = slice_objective(x, y, mode="ncc")
        sub = slice_objective(x, y, mode="ncc", mask=mask)
        assert full != sub

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            slice_objective(np.zeros((4, 4)), np.zeros((5, 5)), mode="ncc")

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            slice_objective(np.zeros((4, 4)), np.zeros((4, 4)), mode="bogus")


# ---------------------------------------------------------------------------
# M8 — refine_residuals (G1-gated) + write_qc_report
# ---------------------------------------------------------------------------


from iss_preprocess.pipeline.tangential_atlas import (  # noqa: E402
    refine_residuals,
    write_qc_report,
)


def _make_m8_state_and_geometry(bundle):
    """Build a small synthetic state + geometry suitable for M8 tests."""
    res = bundle.resolution_um
    n_ap, n_dv, n_ml = bundle.shape_zyx
    j_dv = n_dv // 2
    sn = np.array([1, 2, 3], dtype=np.int64)
    z_um = (sn - 1).astype(np.float64) * (j_dv * res / 2.0)
    yflip = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    gfo = np.broadcast_to(yflip[None, :, :], (3, 3, 3)).copy()
    geom = SliceStackGeometry(
        slice_numbers=sn,
        rois=np.ones(3, dtype=np.int64),
        chambers=tuple(f"c{i}" for i in range(3)),
        z_um=z_um,
        overview_pixel_size_um=res,
        canvas_shape_yx=(n_ml, n_ap),
        global_from_overview=gfo,
        bad_slice_numbers=np.array([], dtype=np.int64),
    )
    P4 = _P_4x4.tolist()
    state = {
        "version": TANGENTIAL_ATLAS_STATE_VERSION,
        "atlas_name": "allen_mouse_10um",
        "atlas_resolution_um": float(res),
        "stack_path": "/tmp/fake/tangential_volume/unregistered_slices.npz",
        "global_transforms_path": "/tmp/fake/tangential_volume/global_slice_transforms.npz",
        "canvas_shape_yx": [n_ml, n_ap],
        "overview_pixel_size_um": float(res),
        "bad_slices": [],
        "pose": {
            "atlas_from_tangential_4x4": P4,
            "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
            "depth_um": 0.0, "tx_atlas_um": 0.0, "ty_atlas_um": 0.0,
            "rotation_order": "ZYX",
            "axes": {"yaw": "DV", "pitch": "ML", "roll": "AP"},
        },
        "residuals": {
            "1": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
            "2": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
            "3": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
        },
        "residual_bounds": {
            "max_dx_um": 250.0, "max_dy_um": 250.0,
            "max_dtheta_deg": 2.0, "max_dz_um": 0.0,
            "smoothness_lambda": 0.0, "centering": "mean_zero",
        },
    }
    return state, geom


class TestRefineResidualsGate:
    def test_default_call_raises_g1(self):
        bundle = _make_atlas_bundle()
        state, geom = _make_m8_state_and_geometry(bundle)
        with pytest.raises(RuntimeError, match="G1"):
            refine_residuals(state, geom, bundle)

    def test_g1_approved_converges_to_zero_on_synthetic_match(self):
        # On a synthetic perfect-match setup (target = atlas plane at zero
        # residual), the optimum is (0, 0, 0); refine_residuals should
        # return residuals near zero.
        bundle = _make_atlas_bundle()
        state, geom = _make_m8_state_and_geometry(bundle)
        out = refine_residuals(state, geom, bundle, g1_approved=True, maxiter=10)
        assert "residuals" in out
        for sn_str, r in out["residuals"].items():
            assert abs(r["dx_um"]) < 1.0
            assert abs(r["dy_um"]) < 1.0
            assert abs(r["dtheta_deg"]) < 0.2


class TestWriteQcReport:
    def test_writes_overlays_and_residuals(self, tmp_path):
        bundle = _make_atlas_bundle()
        state, geom = _make_m8_state_and_geometry(bundle)
        qc = write_qc_report(tmp_path, state, atlas=bundle, geometry=geom)
        assert "overlay_dir" in qc and "residuals_plot" in qc
        overlay_dir = Path(qc["overlay_dir"])
        residuals_path = Path(qc["residuals_plot"])
        assert overlay_dir.exists()
        assert residuals_path.exists()
        pngs = sorted(overlay_dir.glob("sl*.png"))
        assert len(pngs) == 3  # one per kept slice

    def test_spacing_warnings_empty_at_default(self, tmp_path):
        bundle = _make_atlas_bundle()
        state, geom = _make_m8_state_and_geometry(bundle)
        qc = write_qc_report(tmp_path, state, atlas=bundle, geometry=geom)
        assert qc["spacing_warnings"] == []

    def test_context_backed_qc_uses_real_images_and_writes_summary(self, tmp_path):
        volume_dir = tmp_path / "tangential_volume"
        volume_dir.mkdir()
        stack = _build_synthetic_stack(
            volume_dir,
            slice_numbers=(1, 2, 3, 4),
            target_shape=(6, 7),
        )
        transforms = _build_context_transforms(
            volume_dir,
            slice_numbers=(1, 3, 4),
            bad=(2,),
            canvas_shape=(6, 7),
        )
        state = _make_state_payload()
        state["stack_path"] = str(stack)
        state["global_transforms_path"] = str(transforms)
        state["canvas_shape_yx"] = [6, 7]
        state["bad_slices"] = [2]
        state["state_path"] = str(tmp_path / "tangential_atlas" / "tangential_atlas_state.json")
        state["section_thickness_um"] = 20.0
        state["z_um_by_slice"] = {"1": 0.0, "3": 40.0, "4": 60.0}
        state["residuals"] = {
            "1": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
            "3": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
            "4": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
        }
        context = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
            state=state,
        )
        qc = write_qc_report(tmp_path, state, atlas=_make_atlas_bundle(), context=context)
        overlay_dir = Path(qc["overlay_dir"])
        assert overlay_dir == tmp_path / "figures" / "tangential_atlas" / "overlays"
        assert len(sorted(overlay_dir.glob("sl*.png"))) == 3
        summary_path = Path(qc["qc_summary"])
        assert summary_path.exists()
        summary = _json.loads(summary_path.read_text())
        assert summary["state_path"] == state["state_path"]
        assert summary["per_slice"][1]["slice_number"] == 3
        assert summary["per_slice"][1]["image_index"] == 2
        assert "objective_scores" in summary["per_slice"][1]
        assert summary["metadata"]["z_source"] == "state.z_um_by_slice"

    def test_write_tangential_atlas_rasters(self, tmp_path):
        volume_dir = tmp_path / "tangential_volume"
        volume_dir.mkdir()
        stack = _build_synthetic_stack(
            volume_dir,
            slice_numbers=(1, 2, 3),
            target_shape=(5, 6),
        )
        transforms = _build_context_transforms(
            volume_dir,
            slice_numbers=(1, 3),
            bad=(2,),
            canvas_shape=(5, 6),
        )
        state = _make_state_payload()
        state["stack_path"] = str(stack)
        state["global_transforms_path"] = str(transforms)
        state["canvas_shape_yx"] = [5, 6]
        state["bad_slices"] = [2]
        state["section_thickness_um"] = 20.0
        state["z_um_by_slice"] = {"1": 0.0, "3": 40.0}
        state["residuals"] = {
            "1": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
            "3": {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0},
        }
        context = build_tangential_atlas_context(
            tmp_path,
            stack_path=stack,
            transforms_path=transforms,
            state=state,
        )
        out = write_tangential_atlas_rasters(
            tmp_path,
            state,
            context=context,
            atlas=_make_atlas_bundle(),
        )
        assert out == tmp_path / "tangential_atlas" / "tangential_atlas_rasters.npz"
        data = np.load(out)
        assert data["slice_numbers"].tolist() == [1, 3]
        assert data["annotation"].shape == (2, 5, 6)
        assert data["atlas_coordinates_um"].shape == (2, 5, 6, 3)


# ---------------------------------------------------------------------------
# M4 — Spots-To-Atlas Export.
# ---------------------------------------------------------------------------


from iss_preprocess.pipeline.tangential_atlas import (  # noqa: E402
    register_spots_to_tangential_atlas,
)


def _make_spots_df(rows, include_is_bad_slice=True):
    """Build a synthetic spots DataFrame mirroring `register_spots_to_global_volume`.

    `rows` is an iterable of dicts; missing keys default to sensible values.
    """
    records = []
    for i, r in enumerate(rows):
        rec = {
            "x": float(r.get("x", 0.0)),
            "y": float(r.get("y", 0.0)),
            "x_global": r.get("x_global", np.nan),
            "y_global": r.get("y_global", np.nan),
            "z_global_um": r.get("z_global_um", np.nan),
            "slice_number": int(r["slice_number"]),
            "roi": int(r.get("roi", 1)),
            "data_path": str(r.get("data_path", "proj/mouse/chamber_01")),
            "tag": str(r.get("tag", f"row{i}")),
        }
        if include_is_bad_slice:
            rec["is_bad_slice"] = bool(r.get("is_bad_slice", False))
        records.append(rec)
    return pd.DataFrame(records)


def _write_spots_pickle(tmp_path, df, name="barcode_round_spots_global.pkl"):
    path = tmp_path / name
    df.to_pickle(path)
    return path


def _build_m4_transforms(
    tmp_path, *, slice_numbers=(1, 3), bad=(2,), canvas_shape=(5, 6)
):
    """Like ``_build_context_transforms`` but with a y-flip in ``global_from_overview``.

    The y-flip keeps projected ML coordinates non-negative under the
    canonical zero pose, so synthetic spots land inside the M3 atlas.
    """
    slice_numbers = np.asarray(slice_numbers, dtype=np.int64)
    rois = slice_numbers.astype(np.int64)
    data_paths = np.array(
        [f"proj/mouse/chamber_{int(sn):02d}" for sn in slice_numbers], dtype=str
    )
    overview_files = np.array(
        [f"overview_sl{int(sn):03d}.tif" for sn in slice_numbers], dtype=str
    )
    yflip = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    gfo = np.broadcast_to(yflip[None, :, :], (len(slice_numbers), 3, 3)).copy()
    eye = np.broadcast_to(
        np.eye(3, dtype=np.float64), (len(slice_numbers), 3, 3)
    ).copy()
    out = tmp_path / "global_slice_transforms.npz"
    np.savez(
        out,
        reference_slice=np.int64(slice_numbers[0]),
        slice_numbers=slice_numbers,
        rois=rois,
        data_paths=data_paths,
        overview_files=overview_files,
        canvas_shape_yx=np.array(canvas_shape, dtype=np.int64),
        canvas_offset_matrix=np.eye(3),
        global_from_padded=eye,
        global_from_overview=gfo,
        global_from_fullres=eye,
        bad_slices=np.array(list(bad), dtype=np.int64),
        auto_bad_slices=np.array([], dtype=np.int64),
    )
    return out


def _build_m4_context(tmp_path, slice_numbers=(1, 3), bad=(2,), canvas_shape=(5, 6)):
    """Return (context, stack_path, transforms_path, state) for M4 tests."""
    volume_dir = tmp_path / "tangential_volume"
    volume_dir.mkdir(exist_ok=True)
    stack = _build_synthetic_stack(
        volume_dir,
        slice_numbers=tuple(sorted(set(list(slice_numbers) + list(bad)))),
        target_shape=canvas_shape,
    )
    transforms = _build_m4_transforms(
        volume_dir,
        slice_numbers=slice_numbers,
        bad=bad,
        canvas_shape=canvas_shape,
    )
    state = _make_state_payload()
    state["stack_path"] = str(stack)
    state["global_transforms_path"] = str(transforms)
    state["canvas_shape_yx"] = list(canvas_shape)
    state["bad_slices"] = list(bad)
    state["section_thickness_um"] = 20.0
    state["z_um_by_slice"] = {
        str(int(sn)): float(i) * 20.0 for i, sn in enumerate(slice_numbers)
    }
    state["residuals"] = {
        str(int(sn)): {"dx_um": 0.0, "dy_um": 0.0, "dtheta_deg": 0.0, "dz_um": 0.0}
        for sn in slice_numbers
    }
    context = build_tangential_atlas_context(
        tmp_path,
        stack_path=stack,
        transforms_path=transforms,
        state=state,
    )
    return context, stack, transforms, state


def _expected_atlas_um(context, state, sn, x_global, y_global):
    """Project a single canvas-pixel coordinate to atlas microns for ground truth."""
    from iss_preprocess.pipeline.tangential_atlas import (
        SliceResidual,
        build_slice_plane_spec,
    )

    pose_dict = state["pose"]
    pose = StackPose(
        atlas_from_tangential=np.asarray(
            pose_dict["atlas_from_tangential_4x4"], dtype=float
        ),
        yaw_deg=float(pose_dict.get("yaw_deg", 0.0)),
        pitch_deg=float(pose_dict.get("pitch_deg", 0.0)),
        roll_deg=float(pose_dict.get("roll_deg", 0.0)),
        depth_um=float(pose_dict.get("depth_um", 0.0)),
        tx_atlas_um=float(pose_dict.get("tx_atlas_um", 0.0)),
        ty_atlas_um=float(pose_dict.get("ty_atlas_um", 0.0)),
    )
    sn = int(sn)
    slice_index = int(np.where(context.geometry.slice_numbers == sn)[0][0])
    residual = SliceResidual(slice_number=sn)
    spec = build_slice_plane_spec(
        context.geometry, pose, residual=residual, slice_index=slice_index
    )
    pt = np.array([float(x_global), float(y_global), 0.0, 1.0])
    out = spec.atlas_um_from_grid_4x4 @ pt
    return out[:3]


def _make_atlas_bundle_with_acronyms():
    """Atlas bundle whose lookup_df has an `acronym` column."""
    reference, annotation = _synthetic_atlas_arrays()
    return AtlasBundle(
        atlas_name="allen_mouse_10um",
        resolution_um=10.0,
        reference=reference,
        annotation=annotation,
        lookup_df=pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "acronym": ["root", "AreaA", "AreaB", "AreaC"],
            }
        ),
        shape_zyx=tuple(int(d) for d in reference.shape),
    )


class TestRegisterSpotsToTangentialAtlas:
    def test_callable_is_importable(self):
        assert callable(register_spots_to_tangential_atlas)

    def test_bad_global_coordinate_unit_raises(self, tmp_path):
        context, _, _, state = _build_m4_context(tmp_path)
        spots_path = _write_spots_pickle(
            tmp_path, _make_spots_df([{"slice_number": 1, "x_global": 1.0, "y_global": 1.0}])
        )
        with pytest.raises(ValueError, match="global_coordinate_unit"):
            register_spots_to_tangential_atlas(
                tmp_path,
                spots_path=spots_path,
                context=context,
                atlas=_make_atlas_bundle_with_acronyms(),
                global_coordinate_unit="furlongs",
            )

    def test_missing_spots_file_raises(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        with pytest.raises(FileNotFoundError, match="global spots table"):
            register_spots_to_tangential_atlas(
                tmp_path,
                spots_path=tmp_path / "does_not_exist.pkl",
                context=context,
                atlas=_make_atlas_bundle_with_acronyms(),
            )

    def test_projects_known_canvas_pixel_to_atlas_microns_with_zero_pose(
        self, tmp_path
    ):
        context, _, _, state = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [
                {"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "a"},
                {"slice_number": 3, "x_global": 4.0, "y_global": 3.0, "tag": "b"},
            ]
        )
        spots_path = _write_spots_pickle(tmp_path, df)
        out = register_spots_to_tangential_atlas(
            tmp_path, spots_path=spots_path, context=context, atlas=atlas
        )
        result = pd.read_pickle(out).set_index("tag")
        exp_a = _expected_atlas_um(context, state, 1, 2.0, 1.0)
        exp_b = _expected_atlas_um(context, state, 3, 4.0, 3.0)
        np.testing.assert_allclose(
            result.loc["a", ["AP_um", "DV_um", "ML_um"]].to_numpy(dtype=float),
            exp_a,
            atol=1e-9,
        )
        np.testing.assert_allclose(
            result.loc["b", ["AP_um", "DV_um", "ML_um"]].to_numpy(dtype=float),
            exp_b,
            atol=1e-9,
        )

    def test_global_coordinate_unit_um_matches_pixel_input(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        px = float(context.geometry.overview_pixel_size_um)
        df_pixel = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "a"}]
        )
        df_um = _make_spots_df(
            [
                {
                    "slice_number": 1,
                    "x_global": 2.0 * px,
                    "y_global": 1.0 * px,
                    "tag": "a",
                }
            ]
        )
        out_pixel = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df_pixel, name="pixel.pkl"),
            context=context,
            atlas=atlas,
            output_name="pixel_atlas.pkl",
        )
        out_um = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df_um, name="um.pkl"),
            context=context,
            atlas=atlas,
            output_name="um_atlas.pkl",
            global_coordinate_unit="um",
        )
        rp = pd.read_pickle(out_pixel).iloc[0]
        ru = pd.read_pickle(out_um).iloc[0]
        np.testing.assert_allclose(
            [rp.AP_um, rp.DV_um, rp.ML_um], [ru.AP_um, ru.DV_um, ru.ML_um], atol=1e-9
        )

    def test_annotation_lookup_returns_atlas_label(self, tmp_path):
        context, _, _, state = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "a"}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        row = pd.read_pickle(out).iloc[0]
        exp = _expected_atlas_um(context, state, 1, 2.0, 1.0)
        voxel = np.round(exp / atlas.resolution_um).astype(int)
        expected_area = int(atlas.annotation[voxel[0], voxel[1], voxel[2]])
        assert int(row["area_id"]) == expected_area
        expected_acronym = atlas.lookup_df.set_index("id")["acronym"].get(
            expected_area
        )
        assert row["area_acronym"] == expected_acronym
        assert bool(row["atlas_valid"]) is True
        assert row["atlas_warning"] == ""

    def test_acronym_column_present_but_null_when_lookup_df_lacks_acronym(
        self, tmp_path
    ):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle()  # default has no `acronym` column
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "a"}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        result = pd.read_pickle(out)
        assert "area_acronym" in result.columns
        assert result["area_acronym"].isna().all()

    def test_outside_atlas_rows_marked_invalid(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        # AP = 200 * px (5) = 1000um -> voxel 100 > n_AP=60 -> outside.
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 200.0, "y_global": 1.0, "tag": "out"}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        row = pd.read_pickle(out).iloc[0]
        assert np.isfinite(row["AP_um"]) and np.isfinite(row["ML_um"])
        assert pd.isna(row["area_id"])
        assert bool(row["atlas_valid"]) is False
        assert row["atlas_warning"] == "outside_atlas"

    def test_bad_slice_rows_kept_with_invalid_atlas(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [
                {"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "good"},
                {
                    "slice_number": 2,
                    "x_global": np.nan,
                    "y_global": np.nan,
                    "is_bad_slice": True,
                    "tag": "bad",
                },
            ]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        result = pd.read_pickle(out).set_index("tag")
        bad = result.loc["bad"]
        assert pd.isna(bad["AP_um"]) and pd.isna(bad["DV_um"]) and pd.isna(bad["ML_um"])
        assert pd.isna(bad["area_id"])
        assert bool(bad["atlas_valid"]) is False
        assert bad["atlas_warning"] == "bad_slice"
        # good row still projected
        assert bool(result.loc["good", "atlas_valid"]) is True

    def test_unmapped_slice_rows_kept_with_invalid_atlas(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [
                {"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "good"},
                {"slice_number": 99, "x_global": 2.0, "y_global": 1.0, "tag": "ghost"},
            ]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        result = pd.read_pickle(out).set_index("tag")
        ghost = result.loc["ghost"]
        assert pd.isna(ghost["AP_um"])
        assert pd.isna(ghost["area_id"])
        assert bool(ghost["atlas_valid"]) is False
        assert ghost["atlas_warning"] == "slice_not_in_state"

    def test_include_bad_slices_false_drops_bad_rows(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [
                {"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "good"},
                {
                    "slice_number": 2,
                    "x_global": np.nan,
                    "y_global": np.nan,
                    "is_bad_slice": True,
                    "tag": "bad",
                },
            ]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
            include_bad_slices=False,
        )
        result = pd.read_pickle(out)
        assert "bad" not in set(result["tag"])
        assert "good" in set(result["tag"])

    def test_output_columns_present_and_typed(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0, "tag": "a"}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        result = pd.read_pickle(out)
        for col in [
            "AP_um",
            "DV_um",
            "ML_um",
            "area_id",
            "area_acronym",
            "atlas_valid",
            "atlas_warning",
        ]:
            assert col in result.columns
        # original columns preserved
        for col in ["x", "y", "x_global", "y_global", "slice_number", "roi", "tag"]:
            assert col in result.columns
        assert result["atlas_valid"].dtype == bool
        assert isinstance(result["area_id"].dtype, pd.Int64Dtype)

    def test_output_path_default_under_tangential_atlas(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        assert (
            out
            == tmp_path / "tangential_atlas" / "barcode_round_spots_atlas.pkl"
        )
        assert out.exists()

    def test_explicit_output_name_honoured(self, tmp_path):
        context, _, _, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0}]
        )
        out = register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df, name="custom_in.pkl"),
            context=context,
            atlas=atlas,
            output_name="custom_out.pkl",
        )
        assert out == tmp_path / "tangential_atlas" / "custom_out.pkl"
        assert out.exists()

    def test_existing_volume_artifacts_not_modified(self, tmp_path):
        import os

        context, stack, transforms, _ = _build_m4_context(tmp_path)
        atlas = _make_atlas_bundle_with_acronyms()
        df = _make_spots_df(
            [{"slice_number": 1, "x_global": 2.0, "y_global": 1.0}]
        )
        before_stack = os.path.getmtime(stack)
        before_tx = os.path.getmtime(transforms)
        register_spots_to_tangential_atlas(
            tmp_path,
            spots_path=_write_spots_pickle(tmp_path, df),
            context=context,
            atlas=atlas,
        )
        assert os.path.getmtime(stack) == before_stack
        assert os.path.getmtime(transforms) == before_tx


from pathlib import Path  # noqa: E402
