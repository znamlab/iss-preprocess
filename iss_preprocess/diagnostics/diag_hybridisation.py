import numpy as np

from iss_preprocess import vis
from iss_preprocess.io import get_processed_path, load_metadata


def check_hybridisation_setup(data_path, prefixes):
    """Plot the hybridisation spot clusters scatter plots and bleedthrough matrices

    Args:
        data_path (str): Relative path to data folder
        prefixes (list): Prefix of the acquisition to check

    """
    processed_path = get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    if prefixes is None:
        metadata = load_metadata(data_path)
        prefixes = metadata["hybridisation"].keys()
    for hyb_round in prefixes:
        reference_hyb_spots = np.load(
            processed_path / f"{hyb_round}_cluster_means.npz", allow_pickle=True
        )
        figs = vis.plot_clusters(
            [reference_hyb_spots["cluster_means"]],
            reference_hyb_spots["spot_colors"],
            [reference_hyb_spots["cluster_inds"]],
        )
        for fig in figs:
            fig.savefig(figure_folder / f"{hyb_round}_{fig.get_label()}.png")
