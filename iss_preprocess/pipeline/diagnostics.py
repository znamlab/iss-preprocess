"""
Module containing diagnostic plots to make sure steps of the pipeline run smoothly

The functions in here do not compute anything useful, but create figures
"""
from pathlib import Path
import numpy as np
from flexiznam.config import PARAMETERS
from ..io import load_stack, get_roi_dimensions, load_ops
from ..vis import (
    plot_correction_images,
    plot_tilestats_distributions,
    plot_matrix_difference,
)


def check_illumination_correction(
    data_path,
    grand_averages=("barcode_round", "genes_round"),
    plot_tilestats=True,
    verbose=True,
):
    """Check if illumination correction average look reasonable

    Args:
        data_path (str): Relative path to data folder
        grand_averages (list, optional): List of grand averages to plot.
            Defaults to ("barcode_round", "genes_round")
        plot_titlestats (bool, optional): Plot a figure of tilestats change. Defaults
            to True
        verbose (bool, optional): Print info about progress. Defaults to True
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    average_dir = processed_path / data_path / "averages"
    figure_folder = processed_path / data_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    correction_images = dict()
    distributions = dict()

    for fname in average_dir.glob("*average.tif"):
        correction_images[fname.name.replace("_average.tif", "")] = load_stack(fname)
    for fname in average_dir.glob("*_tilestats.npy"):
        distributions[fname.name.replace("_tilestats.npy", "")] = np.load(fname)
    if verbose:
        print(
            f"Found {len(correction_images)} averages"
            + f" and {len(distributions)} tilestats"
        )

    plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
    if plot_tilestats:
        plot_tilestats_distributions(
            data_path, distributions, grand_averages, figure_folder
        )


def reg_to_ref_estimation(data_path, prefix, rois=None):
    """Plot estimation of shifts/angle for registration to ref

    Compare raw measures to ransac

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        rois (list): List of ROIs to process. If None, will either use ops["use_rois"]
            if it is defined, or all ROIs otherwise. Defaults to None
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    reg_dir = processed_path / data_path / "reg"
    figure_folder = processed_path / data_path / "figures"

    ndims = get_roi_dimensions(data_path)
    ops = load_ops(data_path)
    if rois is not None:
        ndims = ndims[np.in1d(ndims[:, 0], rois)]
    elif "use_rois" in ops:
        ndims = ndims[np.in1d(ndims[:, 0], ops["use_rois"])]
    figs = {}
    for roi, *ntiles in ndims:
        raw = np.zeros([3, *ntiles])
        corrected = np.zeros([3, *ntiles])
        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                data = np.load(reg_dir / f"tforms_to_ref_{prefix}_{roi}_{ix}_{iy}.npz")
                raw[:2, ix, iy] = data["shifts"]
                raw[2, ix, iy] = data["angles"]
                data = np.load(
                    reg_dir / f"tforms_corrected_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                corrected[:2, ix, iy] = data["shifts"]
                corrected[2, ix, iy] = data["angles"]
        fig = plot_matrix_difference(
            raw=raw,
            corrected=corrected,
            col_labels=["Shift x", "Shift y", "Angle"],
            line_labels=["Raw", "Corrected", "Difference"],
        )
        fig.suptitle(f"Registration to reference. {prefix} ROI {roi}")
        fig.savefig(
            figure_folder / f"registration_to_ref_estimation_{prefix}_roi{roi}.png"
        )
        figs[roi] = fig
    return fig
