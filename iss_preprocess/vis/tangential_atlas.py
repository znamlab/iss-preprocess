"""Matplotlib + ipywidgets surface for the tangential atlas reviewer.

M4 (S §5 Slice 2) ships only the headless three-panel overlay and an
import-safe ipywidgets stub. The napari reviewer lives in the sibling
module ``vis.tangential_atlas_napari`` (lazy-import). The full
ipywidgets implementation is deferred.
"""

import numpy as np
from skimage.segmentation import find_boundaries

from .volume_registration import (
    OVERLAP_FIXED_CMAP,
    OVERLAP_MOVING_CMAP,
    _intensity_rgba,
)

__all__ = [
    "overlay_atlas_on_slice",
    "review_tangential_atlas_widget",
]


def _annotation_boundary_mask(annotation_plane: np.ndarray) -> np.ndarray:
    """Return a boolean ``(H, W)`` mask of integer-label boundaries."""
    if annotation_plane.ndim != 2:
        raise ValueError(
            f"annotation_plane must be 2-D, got shape {annotation_plane.shape!r}"
        )
    if not np.issubdtype(annotation_plane.dtype, np.integer):
        raise ValueError(
            "annotation_plane must have an integer dtype, "
            f"got {annotation_plane.dtype!r}"
        )
    return find_boundaries(annotation_plane, mode="inner").astype(bool)


def overlay_atlas_on_slice(
    slice_image: np.ndarray,
    atlas_reference_plane: np.ndarray,
    atlas_annotation_plane: np.ndarray = None,
    *,
    contrast: tuple = None,
    border_color: str = "lime",
):
    """Three-panel matplotlib overlay of a slice + its atlas plane.

    Parameters
    ----------
    slice_image : np.ndarray
        ``(H, W)`` overview of one slice.
    atlas_reference_plane : np.ndarray
        ``(H, W)`` atlas reference rendered at the same pose; same
        shape as ``slice_image``.
    atlas_annotation_plane : np.ndarray, optional
        ``(H, W)`` integer label raster; if given, area-border
        contours are drawn on the blended and slice panels.
    contrast : tuple of float, optional
        ``(vmin, vmax)`` for the slice; the atlas uses its own
        min/max for windowing.
    border_color : str
        Matplotlib colour for the annotation borders.

    Returns
    -------
    matplotlib.figure.Figure
        Three-panel figure: panel 0 = slice + atlas overlay,
        panel 1 = slice alone (gray), panel 2 = atlas alone (gray).
    """
    import matplotlib.pyplot as plt

    if slice_image.ndim != 2:
        raise ValueError(
            f"slice_image must be 2-D, got shape {slice_image.shape!r}"
        )
    if atlas_reference_plane.ndim != 2:
        raise ValueError(
            "atlas_reference_plane must be 2-D, got shape "
            f"{atlas_reference_plane.shape!r}"
        )
    if slice_image.shape != atlas_reference_plane.shape:
        raise ValueError(
            f"slice_image shape {slice_image.shape} does not match "
            f"atlas_reference_plane shape {atlas_reference_plane.shape}"
        )
    if atlas_annotation_plane is not None:
        if atlas_annotation_plane.shape != slice_image.shape:
            raise ValueError(
                f"atlas_annotation_plane shape {atlas_annotation_plane.shape} "
                f"does not match slice shape {slice_image.shape}"
            )

    if contrast is not None:
        if (
            not isinstance(contrast, tuple)
            or len(contrast) != 2
            or not all(np.isfinite(v) for v in contrast)
            or contrast[0] >= contrast[1]
        ):
            raise ValueError(
                "contrast must be a (vmin, vmax) tuple with vmin < vmax "
                f"and both finite, got {contrast!r}"
            )
        slice_vmin, slice_vmax = float(contrast[0]), float(contrast[1])
    else:
        slice_vmin = float(np.nanmin(slice_image))
        slice_vmax = float(np.nanmax(slice_image))

    fig = plt.figure(figsize=(8, 14))
    grid = fig.add_gridspec(3, 1)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[2, 0]),
    ]

    axes[0].imshow(
        _intensity_rgba(
            slice_image,
            OVERLAP_FIXED_CMAP,
            vmin=slice_vmin,
            vmax=slice_vmax,
            alpha_scale=1.35,
        )
    )
    axes[0].imshow(
        _intensity_rgba(
            atlas_reference_plane,
            OVERLAP_MOVING_CMAP,
            alpha_scale=1.35,
        )
    )
    axes[0].set_title("Slice + atlas overlay")

    axes[1].imshow(slice_image, cmap="gray", vmin=slice_vmin, vmax=slice_vmax)
    axes[1].set_title("Slice")

    axes[2].imshow(atlas_reference_plane, cmap="gray")
    axes[2].set_title("Atlas reference")

    if atlas_annotation_plane is not None:
        boundaries = _annotation_boundary_mask(atlas_annotation_plane).astype(float)
        axes[0].contour(
            boundaries, levels=[0.5], colors=[border_color], linewidths=0.8
        )
        axes[1].contour(
            boundaries, levels=[0.5], colors=[border_color], linewidths=0.8
        )

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def review_tangential_atlas_widget(*args, **kwargs):
    """ipywidgets reviewer — deferred in v1.

    Mirrors the dual-vis precedent (``vis/volume_registration.py``
    pairs with ``vis/volume_registration_napari.py``). The
    matplotlib + ipywidgets reviewer is intentionally not
    implemented in v1; use
    :func:`iss_preprocess.vis.tangential_atlas_napari.review_tangential_atlas_napari`
    instead.
    """
    raise NotImplementedError(
        "review_tangential_atlas_widget is deferred in v1; use "
        "iss_preprocess.vis.tangential_atlas_napari."
        "review_tangential_atlas_napari for manual interaction "
        "(requires napari)."
    )
