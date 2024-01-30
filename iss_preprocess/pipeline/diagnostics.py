"""
Module containing diagnostic plots to make sure steps of the pipeline run smoothly

The functions in here do not compute anything useful, but create figures
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_pdf import PdfPages
from flexiznam.config import PARAMETERS
from znamutils import slurm_it
import iss_preprocess as iss
from iss_preprocess.pipeline import sequencing
from iss_preprocess import vis


@slurm_it(conda_env="iss-preprocess", module_list=["FFmpeg"])
def check_ref_tile_registration(data_path, prefix="genes_round"):
    """Plot the reference tile registration and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"

    target_folder.mkdir(exist_ok=True, parents=True)

    ops = iss.io.load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    print("Loading and registering sequencing tile")
    reg_stack, bad_pixels = sequencing.load_and_register_sequencing_tile(
        data_path,
        filter_r=False,
        correct_channels=False,
        correct_illumination=False,
        corrected_shifts="reference",
        tile_coors=ops["ref_tile"],
        suffix=ops[f"{prefix.split('_')[0]}_projection"],
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=None,
    )

    # compute vmax based on round 0
    vmaxs = np.percentile(reg_stack[..., 0], 99.99, axis=(0, 1))
    vmins = np.percentile(reg_stack[..., 0], 0.01, axis=(0, 1))
    center = np.array(reg_stack.shape[:2]) // 2
    view = np.array([center - 200, center + 200]).T
    channel_colors = ([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1])

    print("Animating")
    vis.animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"initial_ref_tile_registration_{prefix}.mp4",
        vmax=vmaxs,
        vmin=vmins,
        extent=(view[0], view[1]),
        channel_colors=channel_colors,
    )

    print("Static figure")
    stack = reg_stack[:, :, np.argsort(ops["camera_order"]), :]
    nrounds = stack.shape[3]
    
    def round_image(iround):
        vmax = np.percentile(stack[view[0,0]:view[0,1], view[1,0]:view[1,1], :, iround], 99.99, axis=(0, 1))
        vmin = np.percentile(stack[view[0,0]:view[0,1], view[1,0]:view[1,1], :, iround], 0.01, axis=(0, 1))
        return iss.vis.to_rgb(
            stack[view[0,0]:view[0,1], view[1,0]:view[1,1], :, iround],
            channel_colors,
            vmin=vmin,
            vmax=vmax,
        )

    # Make the smallest rectangle that contains `nrounds` axes
    nrows = int(np.sqrt(nrounds))
    ncols = int(np.ceil(nrounds / nrows))
    fig = plt.figure(figsize=(3.5 * ncols, 3.2 * nrows))
    for iround in range(nrounds):
        ax = fig.add_subplot(nrows, ncols, iround + 1)
        ax.imshow(round_image(iround))
        ax.axis("off")
        ax.set_title(f"Round {iround}")
    iss.vis.add_bases_legend(channel_colors)
    fig.tight_layout()
    fig.savefig(target_folder / f"initial_ref_tile_registration_{prefix}.png")
    print(f"Saved to {target_folder / f'initial_ref_tile_registration_{prefix}.mp4'}")


def check_shift_correction(
    data_path, prefix="genes_round", roi_dimension_prefix="genes_round_1_1"
):
    """Plot the shift correction and save it in the figures folder

    Compare the ransac output to the tile-by-tile shifts and plot
    matrix of differences

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"

    target_folder.mkdir(exist_ok=True, parents=True)

    reg_dir = processed_path / data_path / "reg"
    figure_folder = processed_path / data_path / "figures" / "registration"

    ndims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
    if "use_rois" in ops:
        ndims = ndims[np.in1d(ndims[:, 0], ops["use_rois"])]
    nc = len(ops["camera_order"])
    nr = ops[f"{prefix}s"]

    # Now plot them.
    def get_data(which, roi, nr):
        if nr > 1:
            raw = np.zeros([nc, nr, 3, *ntiles]) + np.nan
            corrected = np.zeros([nc, nr, 3, *ntiles]) + np.nan
        else:
            raw = np.zeros([nc, 3, *ntiles]) + np.nan
            corrected = np.zeros([nc, 3, *ntiles]) + np.nan
        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                try:
                    data = np.load(reg_dir / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz")
                    raw[..., :2, ix, iy] = data[f"shifts_{which}_channels"]
                    raw[..., 2, ix, iy] = data[f"angles_{which}_channels"]
                except FileNotFoundError:
                    pass
                data = np.load(
                    reg_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                corrected[..., :2, ix, iy] = data[f"shifts_{which}_channels"]
                corrected[..., 2, ix, iy] = data[f"angles_{which}_channels"]
        return raw, corrected

    # For "within channels" we plot the shifts for each channel and each round
    fig = plt.figure(figsize=(3 * nr * 2, 2 * 3 * nc))
    for roi, *ntiles in ndims:
        raw, corrected = get_data("within", roi, nr=nr)
        corr_feature = ["Shift x", "Shift y"]
        fig.clear()
        axes = fig.subplots(nrows=nc * 3, ncols=nr * 2)
        for c in range(nc):
            for ifeat, feat in enumerate(corr_feature):
                raw_to_plot = raw[c, :, ifeat, ...]
                corr_to_plot = corrected[c, :, ifeat, ...]
                iss.vis.plot_matrix_difference(
                    raw=raw_to_plot,
                    corrected=corr_to_plot,
                    col_labels=[f"Round {i} {feat}" for i in np.arange(nr)],
                    range_min=[5 if ifeat < 2 else 0.1] * nr,
                    range_max=[10 if ifeat < 2 else 1] * nr,
                    axes=axes[c * 3 : (c + 1) * 3, ifeat * nr : (ifeat + 1) * nr],
                    line_labels=("Raw", f"CHANNEL {c}\nCorrected", "Difference"),
                )
        fig_title = f"{prefix} Correct shift within channels\n"
        fig_title += f"ROI {roi}"
        fig.suptitle(fig_title)
        fig.subplots_adjust(
            wspace=0.15, hspace=0, bottom=0.01, top=0.95, right=0.95, left=0.1
        )

        fname = fig_title.lower().replace(" ", "_").replace("\n", "_")
        fig.savefig(figure_folder / (fname + ".png"))

    # now do "between channels"
    nrois = len(ndims)
    fig = plt.figure(figsize=(3 * 3 * nc, 2 * 2 * nrois))
    axes = fig.subplots(nrows=nc * 3, ncols=nrois * 2)
    for ir, (roi, *ntiles) in enumerate(ndims):
        raw, corrected = get_data("between", roi, nr=1)
        corr_feature = ["Shift x", "Shift y"]
        for ifeat, feat in enumerate(corr_feature):
            raw_to_plot = raw[:, ifeat, ...]
            corr_to_plot = corrected[:, ifeat, ...]
            iss.vis.plot_matrix_difference(
                raw=raw_to_plot,
                corrected=corr_to_plot,
                col_labels=[f"Channel {i} {feat}" for i in np.arange(nc)],
                range_min=[1 if ifeat < 2 else 0.1] * nc,
                range_max=[5 if ifeat < 2 else 1] * nc,
                axes=axes[ir * 3 : (ir + 1) * 3, ifeat * nc : (ifeat + 1) * nc],
                line_labels=("Raw", f"ROI {ir}\nCorrected", "Difference"),
            )
    fig_title = f"{prefix} Correct shift between channels"
    fig.suptitle(fig_title)
    fig.subplots_adjust(
        wspace=0.15, hspace=0, bottom=0.01, top=0.95, right=0.95, left=0.1
    )
    fname = fig_title.lower().replace(" ", "_").replace("\n", "_")
    fig.savefig(figure_folder / (fname + ".png"))


def check_sequencing_tile_registration(data_path, tile_coords, prefix="genes_round"):
    """Plot the a mp4 of registered tile and save it in the figures folder

    This will load the data after ransac correction

    Args:
        data_path (str): Relative path to data folder
        prefix (str, optional): Prefix of the images to load. Defaults to "genes_round".
    """
    processed_path = iss.io.get_processed_path(data_path)
    target_folder = processed_path / "figures" / "registration"

    target_folder.mkdir(exist_ok=True, parents=True)

    ops = iss.io.load_ops(data_path)
    nrounds = ops[f"{prefix}s"]

    # get stack registered between channel and rounds
    reg_stack, bad_pixels = sequencing.load_and_register_sequencing_tile(
        data_path,
        filter_r=False,
        correct_channels=False,
        correct_illumination=False,
        corrected_shifts=True,
        tile_coors=tile_coords,
        suffix=ops[f"{prefix.split('_')[0]}_projection"],
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=None,
    )

    # compute vmax based on round 0
    vmaxs = np.quantile(reg_stack[..., 0], 0.9999, axis=(0, 1))
    center = np.array(reg_stack.shape[:2]) // 2
    view = np.array([center - 200, center + 200]).T

    tilename = "_".join([str(x) for x in tile_coords])
    vis.animate_sequencing_rounds(
        reg_stack,
        savefname=target_folder / f"registration_tile{tilename}_{prefix}.mp4",
        vmax=vmaxs,
        extent=(view[0], view[1]),
        channel_colors=([1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]),
    )


def check_hybridisation_setup(data_path):
    """Plot the hybridisation spot clusters scatter plots and bleedthrough matrices

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    metadata = iss.io.load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        reference_hyb_spots = np.load(
            processed_path / f"{hyb_round}_cluster_means.npz", allow_pickle=True
        )
        figs = iss.vis.plot_clusters(
            [reference_hyb_spots["cluster_means"]],
            reference_hyb_spots["spot_colors"],
            [reference_hyb_spots["cluster_inds"]],
        )
        for fig in figs:
            fig.savefig(figure_folder / f"{hyb_round}_{fig.get_label()}.png")


def check_barcode_calling(data_path):
    """Plot the barcode cluster scatter plots and cluster means and save them in the
    figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    reference_barcode_spots = np.load(
        processed_path / "reference_barcode_spots.npz", allow_pickle=True
    )
    cluster_means = np.load(processed_path / "barcode_cluster_means.npy")
    figs = iss.vis.plot_clusters(
        cluster_means,
        reference_barcode_spots["spot_colors"],
        reference_barcode_spots["cluster_inds"],
    )
    for fig in figs:
        fig.savefig(figure_folder / f"barcode_{fig.get_label()}.png")


def check_omp_setup(data_path):
    """Plot the OMP setup, including clustering of reference gene spots and
    gene templates, and save them in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    reference_gene_spots = np.load(
        processed_path / "reference_gene_spots.npz", allow_pickle=True
    )
    omp_stat = np.load(processed_path / "gene_dict.npz", allow_pickle=True)
    nrounds = reference_gene_spots["spot_colors"].shape[0]
    figs = iss.vis.plot_clusters(
        omp_stat["cluster_means"],
        reference_gene_spots["spot_colors"],
        reference_gene_spots["cluster_inds"],
    )
    figs.append(
        iss.vis.plot_gene_templates(
            omp_stat["gene_dict"],
            omp_stat["gene_names"],
            iss.call.BASES,
            nrounds=nrounds,
        )
    )
    for fig in figs:
        fig.savefig(figure_folder / f"omp_{fig.get_label()}.png")


def check_spot_sign_image(data_path):
    """Plot the average spot sign image and save it in the figures folder

    Args:
        data_path (str): Relative path to data folder

    """
    processed_path = iss.io.get_processed_path(data_path)
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    spot_image = np.load(processed_path / "spot_sign_image.npy")
    iss.vis.plot_spot_sign_image(spot_image)
    plt.savefig(figure_folder / "spot_sign_image.png")


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
    processed_path = iss.io.get_processed_path(data_path)
    average_dir = processed_path / "averages"
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    correction_images = dict()
    distributions = dict()

    for fname in average_dir.glob("*average.tif"):
        correction_images[fname.name.replace("_average.tif", "")] = iss.io.load_stack(
            fname
        )
    for fname in average_dir.glob("*_tilestats.npy"):
        distributions[fname.name.replace("_tilestats.npy", "")] = np.load(fname)
    if verbose:
        print(
            f"Found {len(correction_images)} averages"
            + f" and {len(distributions)} tilestats"
        )

    iss.vis.plot_correction_images(
        correction_images, grand_averages, figure_folder, verbose=True
    )
    if plot_tilestats:
        iss.vis.plot_tilestats_distributions(
            data_path, distributions, grand_averages, figure_folder
        )


def reg_to_ref_estimation(
    data_path, prefix, rois=None, roi_dimension_prefix="genes_round_1_1"
):
    """Plot estimation of shifts/angle for registration to ref

    Compare raw measures to ransac

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        rois (list): List of ROIs to process. If None, will either use ops["use_rois"]
            if it is defined, or all ROIs otherwise. Defaults to None
        roi_dimension_prefix (str, optional): prefix to load roi dimension. Defaults to
            "genes_round_1_1"

    """
    processed_path = iss.io.get_processed_path(data_path)
    reg_dir = processed_path / "reg"
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], ops["use_rois"])]
    figs = {}
    roi_dims[:, 1:] = roi_dims[:, 1:] + 1
    for roi, *ntiles in roi_dims:
        raw = np.zeros([3, *ntiles]) + np.nan
        corrected = np.zeros([3, *ntiles]) + np.nan
        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                try:
                    data = np.load(
                        reg_dir / f"tforms_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                    )
                    raw[:2, ix, iy] = data["shifts"]
                    raw[2, ix, iy] = data["angles"]
                except FileNotFoundError:
                    pass
                data = np.load(
                    reg_dir / f"tforms_corrected_to_ref_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                corrected[:2, ix, iy] = data["shifts"]
                corrected[2, ix, iy] = data["angles"]
        fig = iss.vis.plot_matrix_difference(
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


def check_tile_shifts(
    data_path, prefix, rois=None, roi_dimension_prefix="genes_round_1_1"
):
    """Plot estimation of shifts/angle for registration to ref

    Compare raw measures to ransac

    Args:
        data_path (str): Relative path to data
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        rois (list): List of ROIs to process. If None, will either use ops["use_rois"]
            if it is defined, or all ROIs otherwise. Defaults to None
        roi_dimension_prefix (str, optional): prefix to load roi dimension. Defaults to
            "genes_round_1_1"

    """
    processed_path = iss.io.get_processed_path(data_path)
    reg_dir = processed_path / "reg"
    figure_folder = processed_path / "figures"
    figure_folder.mkdir(exist_ok=True)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix=roi_dimension_prefix)
    ops = iss.io.load_ops(data_path)
    if rois is not None:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], rois)]
    elif "use_rois" in ops:
        roi_dims = roi_dims[np.in1d(roi_dims[:, 0], ops["use_rois"])]
    roi_dims[:, 1:] = roi_dims[:, 1:] + 1
    figs = {}
    data = np.load(reg_dir / f"tforms_{prefix}_{roi_dims[0,0]}_0_0.npz")
    nchannels = data["shifts_within_channels"].shape[0]
    nrounds = data["shifts_within_channels"].shape[1]
    for roi, *ntiles in roi_dims:
        shifts_within_channels_raw = np.zeros([nchannels, nrounds, 2, *ntiles]) + np.nan
        shifts_within_channels_corrected = shifts_within_channels_raw.copy()
        shifts_between_channels_raw = np.zeros([nchannels, 2, *ntiles]) + np.nan
        shifts_between_channels_corrected = shifts_between_channels_raw.copy()

        for ix in range(ntiles[0]):
            for iy in range(ntiles[1]):
                try:
                    data = np.load(reg_dir / f"tforms_{prefix}_{roi}_{ix}_{iy}.npz")
                    shifts_within_channels_raw[:, :, :, ix, iy] = data[
                        "shifts_within_channels"
                    ]
                    shifts_between_channels_raw[:, :, ix, iy] = data[
                        "shifts_between_channels"
                    ]
                except FileNotFoundError:
                    pass
                data = np.load(
                    reg_dir / f"tforms_corrected_{prefix}_{roi}_{ix}_{iy}.npz"
                )
                shifts_within_channels_corrected[:, :, :, ix, iy] = data[
                    "shifts_within_channels"
                ]
                shifts_between_channels_corrected[:, :, ix, iy] = data[
                    "shifts_between_channels"
                ]
        # create a PDF for each roi
        with PdfPages(figure_folder / f"tile_shifts_{prefix}_roi{roi}.pdf") as pdf:
            for ch in range(nchannels):
                for dim in range(2):
                    fig = iss.vis.plot_matrix_difference(
                        raw=shifts_within_channels_raw[ch, :, dim, :, :],
                        corrected=shifts_within_channels_corrected[ch, :, dim, :, :],
                        col_labels=[f"round {i}" for i in range(nrounds)],
                        range_min=np.ones(nrounds) * 5,
                    )
                    fig.suptitle(f"Dim {dim} shifts. {prefix} ROI {roi} channel {ch}")
                    pdf.savefig(fig)
                    figs[roi] = fig
    return figs
