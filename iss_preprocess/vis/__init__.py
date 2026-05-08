from .vis import *


def review_pairwise_registrations_widget(*args, **kwargs):
    """Lazy import wrapper to avoid pulling pipeline imports at vis package import time."""
    from .volume_registration import review_pairwise_registrations_widget as _widget

    return _widget(*args, **kwargs)


def save_pairwise_overlap_plots(*args, **kwargs):
    """Lazy import wrapper for exporting pairwise registration overlay PNGs."""
    from .volume_registration import save_pairwise_overlap_plots as _save

    return _save(*args, **kwargs)


def review_pairwise_registrations_napari(*args, **kwargs):
    """Lazy import wrapper for the napari-based pairwise registration reviewer.

    Requires the optional `napari` extra: `pip install 'iss-preprocess[napari]'`.
    """
    from .volume_registration_napari import (
        review_pairwise_registrations_napari as _napari,
    )

    return _napari(*args, **kwargs)
