"""
Module containing diagnostic plots to make sure steps of the pipeline run smoothly

The functions in here do not compute anything useful, but create figures
"""
from pathlib import Path
import numpy as np
from flexiznam.config import PARAMETERS
from ..io import load_stack, load_ops
from ..vis import plot_correction_images, plot_tilestats_distributions


def check_illumination_correction(
    data_path, grand_averages=("barcode_round", "genes_round"), verbose=True
):
    """Check if illumination correction average look reasonable

    Args:
        data_path (str): Relative path to data folder
        grand_averages (list): List of grand averages to plot.
            Defaults to ("barcode_round", "genes_round")
        verbose (bool): Print info about progress. Defaults to True
    """
    ops = load_ops(data_path)
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
        print(f"Found {len(correction_images)} averages" +
         f" and {len(distributions)} tilestats")
    
    plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
    plot_tilestats_distributions(
        distributions, grand_averages, figure_folder, camera_order=ops['camera_order']
    )
