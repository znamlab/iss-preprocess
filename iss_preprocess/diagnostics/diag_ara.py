import cv2
import matplotlib.pyplot as plt
import numpy as np

import iss_preprocess as iss
from iss_preprocess.io import load_ops
from iss_preprocess.pipeline import ara_registration as ara_registration


def plot_ara_registration(data_path, roi, reference_prefix="genes_round_1_1"):
    """Overlay reference image to ARA borders

    Args:
        data_path (str): Relative path to data
        roi (int): ROI number to plot
        reference_prefix (str, optional): Image to use as reference. Defaults to
            "genes_round_1_1".

    Raises:
        ImportError: If `cricksaw_analysis` is not installed

    Returns:
        plt.Figure: Reference to the figure created.
    """
    try:
        from cricksaw_analysis import atlas_utils
    except ImportError:
        raise ImportError("`plot_registration requires `cricksaw_analysis")
    area_ids = ara_registration.make_area_image(
        data_path=data_path, roi=roi, atlas_size=10, full_scale=False
    )
    reg_metadata = ara_registration.load_registration_reference_metadata(
        data_path, roi=roi
    )

    ops = load_ops(data_path)
    stitched_stack = iss.pipeline.stitch_registered(
        data_path,
        reference_prefix,
        roi=roi,
        channels=ops["ref_ch"],
    ).astype(np.single)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    new_shape = reg_metadata["new_shape"]
    downsampled = cv2.resize(stitched_stack, (new_shape[1], new_shape[0]))
    ax.imshow(
        downsampled,
        extent=(0, new_shape[1], 0, new_shape[0]),
        vmax=np.quantile(downsampled, 0.99),
        vmin=0,
        origin="lower",
        cmap="gray",
    )
    atlas_utils.plot_borders_and_areas(
        ax,
        area_ids,
        border_kwargs=dict(colors="purple", alpha=0.6, linewidths=0.1),
    )
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
