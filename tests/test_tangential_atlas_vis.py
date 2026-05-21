"""Tests for ``iss_preprocess.vis.tangential_atlas`` and
``iss_preprocess.vis.tangential_atlas_napari`` (M4).

Headless: matplotlib forced to Agg, no live napari instantiation.
"""

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.ndimage


def _make_slice_and_atlas_planes(shape=(64, 80), seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(shape).astype(np.float32)
    slice_image = scipy.ndimage.gaussian_filter(raw, sigma=2.0)
    raw2 = rng.standard_normal(shape).astype(np.float32)
    atlas_reference = scipy.ndimage.gaussian_filter(raw2, sigma=2.5)
    quantiles = np.quantile(atlas_reference, [0.25, 0.5, 0.75])
    annotation = np.zeros(shape, dtype=np.uint32)
    annotation[atlas_reference > quantiles[0]] = 1
    annotation[atlas_reference > quantiles[1]] = 2
    annotation[atlas_reference > quantiles[2]] = 3
    return slice_image, atlas_reference, annotation


class TestOverlayAtlasOnSlice:
    def test_returns_figure_headless(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        slice_image, ref, _ = _make_slice_and_atlas_planes()
        fig = overlay_atlas_on_slice(slice_image, ref)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_handles_no_annotation(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        slice_image, ref, _ = _make_slice_and_atlas_planes()
        fig = overlay_atlas_on_slice(slice_image, ref, atlas_annotation_plane=None)
        # No contour collections added when annotation is None.
        for ax in fig.axes:
            assert len(ax.collections) == 0
        plt.close(fig)

    def test_handles_annotation_draws_contours(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        slice_image, ref, ann = _make_slice_and_atlas_planes()
        fig = overlay_atlas_on_slice(slice_image, ref, atlas_annotation_plane=ann)
        # Panels 0 (overlay) and 1 (slice) each get one contour artist.
        assert len(fig.axes[0].collections) >= 1
        assert len(fig.axes[1].collections) >= 1
        # Panel 2 (atlas alone) is left without a contour overlay.
        assert len(fig.axes[2].collections) == 0
        plt.close(fig)

    def test_rejects_mismatched_shapes(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        slice_image, _, _ = _make_slice_and_atlas_planes(shape=(64, 80))
        ref_wrong, _, _ = _make_slice_and_atlas_planes(shape=(32, 40))
        with pytest.raises(ValueError, match="does not match"):
            overlay_atlas_on_slice(slice_image, ref_wrong)

    @pytest.mark.parametrize(
        "bad_contrast",
        [(0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0), (0.0, float("inf"))],
    )
    def test_rejects_invalid_contrast(self, bad_contrast):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        slice_image, ref, _ = _make_slice_and_atlas_planes()
        with pytest.raises(ValueError, match="contrast"):
            overlay_atlas_on_slice(slice_image, ref, contrast=bad_contrast)

    def test_rejects_non_2d_inputs(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        with pytest.raises(ValueError, match="2-D"):
            overlay_atlas_on_slice(
                np.zeros((4, 5, 6), dtype=np.float32),
                np.zeros((4, 5), dtype=np.float32),
            )


class TestReviewTangentialAtlasNapari:
    def test_module_imports_without_napari(self, monkeypatch):
        """The module itself must not import napari at load time."""
        # Drop the napari modules so the next import would trigger a fresh
        # import.
        for name in list(sys.modules):
            if name == "napari" or name.startswith("napari."):
                monkeypatch.delitem(sys.modules, name, raising=False)

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "napari" or name.startswith("napari."):
                raise ImportError("simulated missing napari")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Drop the tangential napari module too so we re-import under the
        # patched __import__.
        monkeypatch.delitem(
            sys.modules,
            "iss_preprocess.vis.tangential_atlas_napari",
            raising=False,
        )
        import iss_preprocess.vis.tangential_atlas_napari as mod  # noqa: F401

        assert callable(mod.review_tangential_atlas_napari)

    def test_review_tangential_atlas_napari_raises_friendly_import_error(
        self, monkeypatch
    ):
        """Calling the reviewer when napari is unavailable should raise
        ImportError with a clear message."""
        from iss_preprocess.vis import tangential_atlas_napari as mod

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "napari":
                raise ImportError("simulated missing napari")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ImportError, match="napari"):
            mod.review_tangential_atlas_napari("/tmp/does_not_exist.npz")

    def test_helper_uses_slice_number_mapping_not_kept_index(self):
        from iss_preprocess.vis import tangential_atlas_napari as mod

        images = np.zeros((4, 3, 3), dtype=np.float32)
        images[0] = 1.0
        images[1] = 2.0
        images[2] = 3.0
        images[3] = 4.0
        ctx = SimpleNamespace(
            images=images,
            slice_records=(
                SimpleNamespace(slice_number=1, kept_index=0, image_index=0, roi=1, chamber="c1", z_um=0.0),
                SimpleNamespace(slice_number=3, kept_index=1, image_index=2, roi=3, chamber="c3", z_um=40.0),
            ),
            metadata=SimpleNamespace(
                overview_pixel_size_um=5.0,
                section_thickness_um=20.0,
                z_source="slice_number_spacing",
            ),
            issues=(
                SimpleNamespace(
                    severity="warning",
                    code="section_thickness_fallback_20um",
                    slice_number=None,
                ),
            ),
        )

        selected = mod._slice_image_for_index(ctx, 1)
        assert selected[0, 0] == 3.0
        text = mod._slice_metadata_text(ctx, 1)
        assert "slice_number=3" in text
        assert "kept_index=1" in text
        assert "image_index=2" in text
        assert "section_thickness_fallback_20um" in text

    def test_annotation_border_helper(self):
        from iss_preprocess.vis import tangential_atlas_napari as mod

        annotation = np.zeros((8, 8), dtype=np.int64)
        annotation[2:6, 2:6] = 5
        border = mod._annotation_border(annotation)
        assert border.shape == annotation.shape
        assert border.dtype == np.uint8
        assert border.sum() > 0


class TestTangentialAtlasWorkflowNotebook:
    def test_notebook_skeleton_exists_and_has_expected_cells(self):
        path = Path("notebooks/tangential_atlas_workflow.ipynb")
        assert path.exists()
        nb = json.loads(path.read_text())
        cells = nb["cells"]
        source = "\n".join("".join(cell.get("source", [])) for cell in cells)
        assert "mouse_path" in source
        assert "overview_pixel_size_um" in source
        assert "section_thickness_um = 20.0" in source
        assert "build_tangential_atlas_context" in source
        assert "review_tangential_atlas_napari" in source
        assert "load_tangential_atlas_state" in source
        assert "render_reference_plane" in source
        assert "render_annotation_plane" in source
        assert "write_qc_report" in source
        assert "write_tangential_atlas_rasters" in source
        assert "register_spots_to_tangential_atlas" in source


class TestTangentialAtlasStub:
    def test_overlay_atlas_on_slice_importable(self):
        from iss_preprocess.vis.tangential_atlas import overlay_atlas_on_slice

        assert callable(overlay_atlas_on_slice)

    def test_widget_stub_raises_not_implemented(self):
        from iss_preprocess.vis.tangential_atlas import (
            review_tangential_atlas_widget,
        )

        with pytest.raises(NotImplementedError, match="napari"):
            review_tangential_atlas_widget()

    def test_module_imports_without_ipywidgets(self, monkeypatch):
        """Acceptance bullet 3 — import-safe in environments without ipywidgets."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "ipywidgets" or name.startswith("ipywidgets."):
                raise ImportError("simulated missing ipywidgets")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.delitem(
            sys.modules, "iss_preprocess.vis.tangential_atlas", raising=False
        )
        import iss_preprocess.vis.tangential_atlas as mod  # noqa: F401

        assert callable(mod.overlay_atlas_on_slice)
