"""
Module containing diagnostic plots to make sure steps of the pipeline run smoothly

The functions in here do not compute anything useful, but create figures
"""
from pathlib import Path
import flexiznam as flz
from ..io import load_stack
from ..vis import plot_correction_images


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
    processed = Path(flz.PARAMETERS["data_root"]["processed"])
    average_dir = processed / data_path / "averages"
    figure_folder = processed / data_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    correction_images = dict()

    for fname in average_dir.glob("*average.tif"):
        correction_images[fname.name.replace("_average.tif", "")] = load_stack(fname)
    if verbose:
        print(f"Found {len(correction_images)} averages")
    correction_images.keys()
    plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
