"""Functions accessory to the pipeline.

This module contains functions that are useful for checking that the pipeline works
but that are not required per se to run it."""


import matplotlib.pyplot as plt
from znamutils import slurm_it
from ..vis.diagnostics import plot_correction_images, plot_tilestats_distributions, plot_spot_sign_image
from ..io import load_ops, load_stack, get_roi_dimensions, get_processed_path, get_zprofile


import numpy as np


def _get_some_tiles(data_path, prefix, tile_coords=None):
    """Get some tiles to check registration

    If `tile_coords` is None, will select 10 tiles. If `ops` has a `xx_ref_tiles`
    matching prefix, these will be part of the 10 tiles. The remaining tiles will be
    selected randomly.

    Args:
        data_path (str): Relative path to data folder
        prefix (str): Prefix of the images to load
        tile_coords (list, optional): List of tile coordinates to process. If None, will
            select 10 tiles. Defaults to None.

    Returns:
        list: List of tile coordinates
    """
    ops = load_ops(data_path)
    # get stack registered between channel and rounds
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    if tile_coords is None:
        # check if ops has a ref tile
        if f"{prefix.split('_')[0]}_ref_tiles" in ops:
            tile_coords = ops[f"{prefix.split('_')[0]}_ref_tiles"]
            nrandom = 10 - len(tile_coords)
        else:
            tile_coords = []
            nrandom = 10
        # select random tiles
        if nrandom > 0:
            for i in range(nrandom):
                # pick a roi randomly
                roi = np.random.choice(roi_dims[:, 0])
                # pick a tile inside that roi
                ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0]
                tile_coords.append([roi, *np.random.randint(0, ntiles)])
    elif isinstance(tile_coords[0], int):
        tile_coords = [tile_coords]
    return tile_coords


def check_spot_sign_image(data_path):
    """Plot the average spot sign image and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    spot_image = np.load(processed_path / "spot_sign_image.npy")
    plot_spot_sign_image(spot_image)
    plt.savefig(figure_folder / "spot_sign_image.png")


@slurm_it(conda_env="iss-preprocess")
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
    processed_path = get_processed_path(data_path)
    average_dir = processed_path / "averages"
    figure_folder = processed_path / "figures"
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


def plot_zfocus(data_path, prefix, rois=None, verbose=True):
    """Plot the z-focus of a ROI

    Args:
        data_path (str): Relative path to data folder
        prefix (str): Prefix of the images to load.
        rois (int | list, optional): ROI to plot. If None, will plot all ROIs. Defaults 
        to None.
        verbose (bool, optional): Print info about progress. Defaults to True.
        

    """
    roi_dims = get_roi_dimensions(data_path, prefix=prefix)
    if rois is None:
        rois = roi_dims[:, 0]
    elif isinstance(rois, int):
        rois = [rois]

    # collect all data    
    for roi in rois:
        if verbose:
            print(f"Plotting ROI {roi}")
        nx, ny = roi_dims[roi_dims[:, 0] == roi, 1:][0]
        zfocus = np.zeros((nx, ny, 4))
        zprofiles = None
        for tile_x in range(nx):
            for tile_y in range(ny):
                zprof = get_zprofile(data_path, prefix, (roi, tile_x, tile_y))
                # use std proj, max was too noisy
                zprof = zprof["std"]
                if zprofiles is None:
                    zprofiles = np.zeros((nx, ny, *zprof.shape))
                zprofiles[tile_x, tile_y, :] = zprof
                zfocus[tile_x, tile_y, :] = np.argmax(zprof, axis=1)
        
    # find out of focus tiles
    nz = zfocus.shape[-1]
    correct = (zfocus > 4) & (zfocus < nz - 4)
    
    aspect = nx / ny
    fig, axs = plt.subplots(nx, ny, figsize=(12, 12 * aspect))
