from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from znamutils import slurm_it

from iss_preprocess.call.call import extract_traces_somata
from iss_preprocess.image.utils import highpass_stack
from iss_preprocess.io.load import load_metadata
from iss_preprocess.pipeline.segment import find_edge_touching_masks
from iss_preprocess.vis.vis import plot_clusters


from ..call import (
    BASES,
    extract_spots,
    get_cluster_means,
)
from ..call.omp import barcode_spots_dot_product, make_gene_templates, run_omp
from ..call.spot_shape import (
    apply_symmetry,
    detect_spots_by_shape,
    find_gene_spots,
    get_spot_shape,
)
from ..image import (
    compute_distribution,
    filter_stack,
)
from ..io import get_processed_path, load_ops, load_sequencing_rounds, write_stack
from ..segment import detect_isolated_spots
from .register import load_and_register_sequencing_tile, load_register_and_fill_tile, load_register_and_fill_tile_mask


@slurm_it(conda_env="iss-preprocess")
def setup_barcode_calling(data_path):
    """Detect spots and compute cluster means

    Args:
        data_path (str): Relative path to data

    Returns:
        cluster_means (list): A list with Nrounds elements. Each a Nch x Ncl (square
            because N channels is equal to N clusters) array of cluster means,
            normalised by round 0 intensity
        all_spots (pandas.DataFrame): All detected spots.

    """
    # TODO: move most of this to a pipeline.py function
    ops = load_ops(data_path)
    print("detecting barcode spots")
    all_spots, _ = get_reference_spots(data_path, prefix="barcode")
    cluster_means, spot_colors, cluster_inds = get_cluster_means(
        all_spots,
        score_thresh=ops["barcode_cluster_score_thresh"],
        initial_cluster_mean=np.array(ops["initial_cluster_means"]),
    )
    processed_path = get_processed_path(data_path)
    np.savez(
        processed_path / "reference_barcode_spots.npz",
        spot_colors=spot_colors,
        cluster_inds=cluster_inds,
    )
    np.save(processed_path / "barcode_cluster_means.npy", cluster_means)

    # check_barcode_calling(data_path)
    print("barcode calling setup complete")
    return cluster_means, all_spots

@slurm_it(conda_env="iss-preprocess")
def setup_soma_barcode_calling(data_path, reload=False):
    """Detect spots and compute cluster means

    Args:
        data_path (str): Relative path to data

    Returns:
        cluster_means (list): A list with Nrounds elements. Each a Nch x Ncl (square
            because N channels is equal to N clusters) array of cluster means,
            normalised by round 0 intensity
        all_spots (pandas.DataFrame): All detected spots.

    """

    ops = load_ops(data_path)
    reference_tiles = ops["barcode_soma_reference_tiles"]
    all_tile_dfs = []
    processed_path = get_processed_path(data_path)
    reloaded = None
    # reload 
    if reload:
        try:
            reloaded = pd.read_pickle(processed_path / "somata_traces_df.pkl")
        except FileNotFoundError:
            print("Could not find saved tile traces, starting from scratch.")

    for tile_coors in reference_tiles:
    # load stack and masks
        # max
        tile_id = "_".join(map(str, tile_coors))
        if reloaded is not None and tile_id in reloaded["tile_of_origin"].values:
            all_tile_dfs.append(reloaded[reloaded["tile_of_origin"] == tile_id])
            continue

        max_stack, bad_pixels = load_register_and_fill_tile(
            data_path,
            tile_coors,
            filter_r=None,
            prefix="barcode_round",
            suffix="max",
            nrounds=ops["barcode_rounds"],
            correct_channels="round1_only",
            corrected_shifts=ops["corrected_shifts"],
            correct_illumination=True,
            reference_prefix = ops["reference_prefix"],
            specific_rounds=None,
            edge=10,
            mid=5,
            zero_fill_output=True,
        )

        max_stack[bad_pixels, ...] = 0 # however there shouldn't be any bad pixels left

        # Reorder channels (assumes max_stack is (H, W, C, R))
        cam_order = np.argsort(ops["camera_order"])
        max_stack = max_stack[:, :, cam_order, :]

        # High-pass filter the stack to enhance somata
        filtered_stack = highpass_stack(max_stack, cutoff=15.0, order=2, pad=100)

        masks = load_register_and_fill_tile_mask(
            data_path,
            tile_coors,
            prefix=ops["segmentation_acquisition"],
            reference_prefix=ops["reference_prefix"],
            corrected_shifts=ops["corrected_shifts"],
        )

        # before barcode calling remove edge masks again (these will be covered on adjacent tiles anyways).
        masks, _ = find_edge_touching_masks(masks, border_width=10)


        # extract traces for each mask
        tile_traces_df = extract_traces_somata(filtered_stack, masks)

        # Filter + annotate (avoid SettingWithCopy)
        tile_traces_df = (
            tile_traces_df
            .loc[tile_traces_df["std"] > ops["soma_std_threshold_clustering"]]
            .copy()
        )
        tile_id = "_".join(map(str, tile_coors))
        tile_traces_df["tile_of_origin"] = tile_id

        all_tile_dfs.append(tile_traces_df)

    traces_df = pd.concat(all_tile_dfs, axis=0, ignore_index=True) if all_tile_dfs else pd.DataFrame()

    # run kmeans on traces to get cluster means
    cluster_means = []
    initial_cluster_mean = np.array(ops["initial_cluster_means"])   
    cluster_means, spot_colors, cluster_inds = get_cluster_means(
        traces_df,
        score_thresh=ops["somata_cluster_score_thresh"],
        initial_cluster_mean=initial_cluster_mean,
    )

    # save somata clustering
    np.save(processed_path / "somata_barcode_cluster_means.npy", cluster_means)
    np.savez(
        processed_path / "somata_reference_barcodes.npz",
        spot_colors=spot_colors,
        cluster_inds=cluster_inds,
    )
    # save traces_df
    traces_df.to_pickle(processed_path / "somata_traces_df.pkl")

    # diagnostics
    figure_folder = processed_path / "figures" / "cells"
    figure_folder.mkdir(exist_ok=True)
    reference_barcode_spots = np.load(
        processed_path / "somata_reference_barcodes.npz", allow_pickle=True
    )

    cluster_means = np.load(processed_path / "somata_barcode_cluster_means.npy")
    figs = plot_clusters(
        cluster_means,
        reference_barcode_spots["spot_colors"],
        reference_barcode_spots["cluster_inds"],
    )
    for fig in figs:
        fig.savefig(figure_folder / f"barcode_{fig.get_label()}.png")

def basecall_tile(data_path, tile_coors, save_spots=True):
    """Detect and basecall barcodes for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
        save_spots (bool, optional): Whether to save the detected spots. Used to run
            without erasing during diagnostics. Defaults to True.

    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    cluster_means = np.load(processed_path / "barcode_cluster_means.npy")

    print(f"Loading and registering tile {tile_coors}")
    # check ops for whether to fill tile with neighbours?
    if ops["fill_with_neighbours"]:
        print("Filling missing pixels with neighbouring tiles")
        ref_round = ops["ref_round"] + 1  # rounds are 1-indexed in filenames
        reference_prefix = f"barcode_round_{ref_round}_1"
        stack, bad_pixels = load_register_and_fill_tile(
            data_path,
            tile_coors,
            filter_r=ops["filter_r"],
            prefix="barcode_round",
            suffix=ops["barcode_projection"],
            nrounds=ops["barcode_rounds"],
            correct_channels=ops["barcode_correct_channels"],
            corrected_shifts=ops["corrected_shifts"],
            correct_illumination=True,
            reference_prefix=reference_prefix,
            specific_rounds=None, 
            edge=10,
            mid=5,
            zero_fill_output=False,
        )
    else:
        print("Not filling missing pixels with neighbouring tiles")
        stack, bad_pixels = load_and_register_sequencing_tile(
            data_path,
            tile_coors,
            filter_r=ops["filter_r"],
            prefix="barcode_round",
            suffix=ops["barcode_projection"],
            nrounds=ops["barcode_rounds"],
            correct_channels=ops["barcode_correct_channels"],
            corrected_shifts=ops["corrected_shifts"],
            correct_illumination=True,
        )

    stack = stack[:, :, np.argsort(ops["camera_order"]), :]

    spot_sign_image = load_spot_sign_image(data_path, ops["spot_shape_threshold"])
    print(f"Detecting spots in tile {tile_coors}")

    basecalling_proj_across_rounds = ops.get(f"basecalling_proj_across_rounds", None)
    if basecalling_proj_across_rounds is not None:
        stack_for_proj = stack[:, :, :, basecalling_proj_across_rounds]
    else:
        stack_for_proj = stack.copy()

    if ops.get(f"basecalling_proj_type", "std") == "mean":
        detect_image = np.nanmean(
            stack_for_proj, axis=(2, 3)
        )
    elif ops.get(f"basecalling_proj_type", "std") == "std":
        detect_image = np.nanstd(
            stack_for_proj, axis=(2, 3)
        )
    else:
        raise ValueError("basecalling_proj_type must be 'mean' or 'std'")
    #always use mean for scoring
    score_image = np.nanmean(
            stack_for_proj, axis=(2, 3)
        )

    spots = detect_spots_by_shape(
        detect_image,
        spot_sign_image,
        threshold=ops["barcode_detection_threshold_basecalling"],
        rho=ops["barcode_spot_rho"],
        score_image=score_image,
    )
    print(f"Extracting spots in tile {tile_coors}")
    extract_spots(spots, stack, ops["spot_extraction_radius"])
    if len(spots) == 0:
        print(f"No spots detected in tile {tile_coors}")
        col2add = [
            "sequence",
            "scores",
            "mean_score",
            "bases",
            "dot_product_score",
            "mean_intensity",
        ]
        spots = pd.DataFrame(columns=spots.columns.tolist() + col2add)
    else:
        x = np.stack(spots["trace"], axis=2)
        x = np.nan_to_num(x)

        cluster_inds = []
        top_score = []

        print(f"Basecalling tile {tile_coors}")
        # TODO: perhaps we should apply background correction before basecalling?
        for iround in range(ops["barcode_rounds"]):
            this_round_means = cluster_means[iround] / np.linalg.norm(
                cluster_means[iround], axis=1, keepdims=True
            )
            x_norm = x[iround, :, :].T / np.linalg.norm(
                x[iround, :, :].T, axis=1, keepdims=True
            )

            # should be Spots x Channels matrix @ Channels x Clusters matrix
            score = x_norm @ this_round_means.T
            cluster_ind = np.argmax(score, axis=1)
            top_score.append(score[np.arange(x_norm.shape[0]), cluster_ind])
            cluster_ind[np.isnan(score).any(axis=1)] = cluster_means[iround].shape[0]
            cluster_inds.append(cluster_ind)

        sequences = np.stack(cluster_inds, axis=1)
        print("Adding quality metrics to spots")
        spots["sequence"] = [seq for seq in sequences]
        scores = np.stack(top_score, axis=1)
        spots["scores"] = [s for s in scores]
        spots["mean_score"] = np.nanmean(scores, axis=1)
        bases = np.hstack([BASES, ["N"]])
        spots["bases"] = ["".join(bases[seq]) for seq in spots["sequence"]]
        spots["dot_product_score"] = barcode_spots_dot_product(spots, cluster_means)
        spots["mean_intensity"] = [np.mean(np.abs(trace)) for trace in spots["trace"]]
    if save_spots:
        save_dir = processed_path / "spots"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving spots to {save_dir}")
        spots.to_pickle(
            save_dir
            / f"barcode_round_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
        )
    print(f"Basecalling complete for tile {tile_coors}")
    return stack, spot_sign_image, spots

@slurm_it(conda_env="iss-preprocess")
def basecall_somata_tile(data_path, tile_coors, save_basecalled_somata=True):
    """Detect and basecall barcodes for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
        save_basecalled_somata (bool, optional): Whether to save the basecalled somata. Used to run
            without erasing during diagnostics. Defaults to True.
        use_slurm (bool, optional): Submit job to slurm. Defaults to True.
    Returns:
        filtered_stack (numpy.ndarray): The high-pass filtered registered stack for the tile.
        masks (numpy.ndarray): The segmentation masks for the tile.
        tile_traces_df (pandas.DataFrame): DataFrame containing the extracted traces and basecalling results for each detected soma.
    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    cluster_means = np.load(processed_path / "somata_barcode_cluster_means.npy")

    masks = load_register_and_fill_tile_mask(
        data_path,
        tile_coors,
        prefix=ops["segmentation_acquisition"],
        reference_prefix=ops["reference_prefix"],
        corrected_shifts=ops["corrected_shifts"],
    )

    # before barcode calling remove edge masks again (these will be covered on adjacent tiles anyways).
    masks, _ = find_edge_touching_masks(masks, border_width=10)

    # check any masks in the tile ie any pixel above 0 in the mask?
    if np.nansum(masks) == 0:
        print(f"No masks detected in tile {tile_coors}")
        col2add = [
            "label", 
            "centroid-0", 
            "centroid-1", 
            "area",
            "trace",
            "std",
            "sequence",
            "scores",
            "mean_score",
            "bases",
            "dot_product_score",
            "mean_intensity",
            "roi",
            "tilex",
            "tiley",
        ]
        tile_traces_df = pd.DataFrame(columns=col2add)
        if save_basecalled_somata:
            save_dir = processed_path / "cells"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving somata to {save_dir}")
            tile_traces_df.to_pickle(
                save_dir
                / f"barcode_round_somata_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
            )
        return None, None, tile_traces_df

    max_stack, bad_pixels = load_register_and_fill_tile(
        data_path,
        tile_coors,
        filter_r=None,
        prefix="barcode_round",
        suffix="max",
        nrounds=ops["barcode_rounds"],
        correct_channels="round1_only",
        corrected_shifts=ops["corrected_shifts"],
        correct_illumination=True,
        reference_prefix = ops["reference_prefix"],
        specific_rounds=None,
        edge=10,
        mid=5,
        zero_fill_output=True,
    )

    max_stack[bad_pixels, ...] = 0 # however there shouldn't be any bad pixels left

    # Reorder channels (assumes max_stack is (H, W, C, R))
    cam_order = np.argsort(ops["camera_order"])
    max_stack = max_stack[:, :, cam_order, :]

    # High-pass filter the stack to enhance somata
    filtered_stack = highpass_stack(max_stack, cutoff=15.0, order=2, pad=100)


    tile_traces_df = extract_traces_somata(filtered_stack, masks)

    if len(tile_traces_df) == 0:
        print(f"No spots detected in tile {tile_coors}")
        col2add = [
            "sequence",
            "scores",
            "mean_score",
            "bases",
            "dot_product_score",
            "mean_intensity",
            "roi",
            "tilex",
            "tiley",
        ]
        tile_traces_df = pd.DataFrame(columns=tile_traces_df.columns.tolist() + col2add)
    else:
        x = np.stack(tile_traces_df["trace"], axis=2)
        x = np.nan_to_num(x)

        cluster_inds = []
        top_score = []

        print(f"Basecalling tile {tile_coors}")
        for iround in range(ops["barcode_rounds"]):
            this_round_means = cluster_means[iround] / np.linalg.norm(
                cluster_means[iround], axis=1, keepdims=True
            )
            x_norm = x[iround, :, :].T / np.linalg.norm(
                x[iround, :, :].T, axis=1, keepdims=True
            )

            # should be Spots x Channels matrix @ Channels x Clusters matrix
            score = x_norm @ this_round_means.T
            cluster_ind = np.argmax(score, axis=1)
            top_score.append(score[np.arange(x_norm.shape[0]), cluster_ind])
            cluster_ind[np.isnan(score).any(axis=1)] = cluster_means[iround].shape[0]
            cluster_inds.append(cluster_ind)

        sequences = np.stack(cluster_inds, axis=1)
        print("Adding quality metrics to spots")
        tile_traces_df["sequence"] = [seq for seq in sequences]
        scores = np.stack(top_score, axis=1)
        tile_traces_df["scores"] = [s for s in scores]
        tile_traces_df["mean_score"] = np.nanmean(scores, axis=1)
        bases = np.hstack([BASES, ["N"]])
        tile_traces_df["bases"] = ["".join(bases[seq]) for seq in tile_traces_df["sequence"]]
        tile_traces_df["dot_product_score"] = barcode_spots_dot_product(tile_traces_df, cluster_means)
        tile_traces_df["mean_intensity"] = [np.mean(np.abs(trace)) for trace in tile_traces_df["trace"]]
        tile_traces_df["roi"] = tile_coors[0]
        tile_traces_df["tilex"] = tile_coors[1]
        tile_traces_df["tiley"] = tile_coors[2]

    if save_basecalled_somata:
        save_dir = processed_path / "cells"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving somata to {save_dir}")
        tile_traces_df.to_pickle(
            save_dir
            / f"barcode_round_somata_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
        )
    print(f"Basecalling complete for tile {tile_coors}")
    return filtered_stack, masks, tile_traces_df

@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "1:00:00", "mem": "8GB"})
def setup_omp(data_path, force_redo=False):
    """Prepare variables required to run the OMP algorithm. Finds isolated spots using
    STD across rounds and channels. Detected spots are then used to determine the
    bleedthrough matrix using scaled k-means.

    Args:
        data_path (str): Relative path to data.
        force_redo (bool, optional): Whether to redo the setup. Defaults to False.

    Returns:
        numpy.ndarray: N x M dictionary, where N = R * C and M is the
            number of genes.
        list: gene names.
        float: norm shift for the OMP algorithm, estimated as median norm of all pixels.

    """
    # TODO: move most of this to a pipeline.py function?
    print("setting up OMP")
    ops = load_ops(data_path)
    processed_path = get_processed_path(data_path)
    targets = [
        processed_path / "reference_gene_spots.npz",
        processed_path / "gene_dict.npz",
    ]
    if all([target.exists() for target in targets]) and not force_redo:
        print("Gene dictionary already exists. Skipping setup.")
        return
    print("detecting reference gene spots")
    all_spots, norm_shifts = get_reference_spots(data_path, prefix="genes")
    cluster_means, spot_colors, cluster_inds = get_cluster_means(
        all_spots,
        initial_cluster_mean=np.array(ops["initial_cluster_means"]),
        score_thresh=ops["genes_cluster_score_thresh"],
    )
    np.savez(
        processed_path / "reference_gene_spots.npz",
        spot_colors=spot_colors,
        cluster_inds=cluster_inds,
    )
    print(f'Saved cluster means to {processed_path / "reference_gene_spots.npz"}')
    codebook = pd.read_csv(
        Path(__file__).parent.parent / "call" / ops["codebook"],
        header=None,
        names=["gii", "seq", "gene"],
    )
    gene_dict, gene_names = make_gene_templates(cluster_means, codebook)
    norm_shift = np.min(norm_shifts)
    np.savez(
        processed_path / "gene_dict.npz",
        gene_dict=gene_dict,
        gene_names=gene_names,
        norm_shift=norm_shift,
        cluster_means=cluster_means,
    )
    print(f'Saved gene dictionary to {processed_path / "gene_dict.npz"}')
    # check_omp_setup(data_path)
    return gene_dict, gene_names, norm_shift


def get_reference_spots(data_path, prefix="genes"):
    """Load the reference spots for the given dataset.

    Internal function for setup_omp and setup_barcode_calling.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Short prefix, either 'genes' or 'barcode'. Defaults to
            'genes'.

    Returns:
        pandas.DataFrame: Detected spots.
        list: Normalisation shifts.

    """
    ops = load_ops(data_path)
    all_spots = []
    norm_shifts = []
    for ref_tile in ops[f"{prefix}_ref_tiles"]:
        print(f"detecting spots in tile {ref_tile}")
        stack, bad_pixels = load_and_register_sequencing_tile(
            data_path,
            ref_tile,
            filter_r=ops["filter_r"],
            prefix=f"{prefix}_round",
            suffix=ops[f"{prefix}_projection"],
            nrounds=ops[f"{prefix}_rounds"],
            correct_channels=ops[f"{prefix}_correct_channels"],
            corrected_shifts=ops["corrected_shifts"],
            correct_illumination=False,
        )
        stack[bad_pixels, :, :] = 0
        stack = stack[:, :, np.argsort(ops["camera_order"]), :]

        basecalling_proj_across_rounds = ops.get(f"basecalling_proj_across_rounds", None)
        if basecalling_proj_across_rounds is not None:
            stack_for_proj = stack[:, :, :, basecalling_proj_across_rounds]

        if ops.get(f"basecalling_proj_type", "std") == "std":
            proj_image = np.nanstd(
                stack_for_proj, axis=(2, 3)
            )
        else:
            proj_image = np.nanmean(
                stack_for_proj, axis=(2, 3)
            )

        spots = detect_isolated_spots(
            proj_image,
            detection_threshold=ops[f"{prefix}_detection_threshold"],
            isolation_threshold=ops[f"{prefix}_isolation_threshold"],
        )

        extract_spots(spots, stack, ops["spot_extraction_radius"])
        all_spots.append(spots)
        norm_shift = np.sqrt(np.median(np.sum(stack**2, axis=(2, 3))))
        norm_shifts.append(norm_shift)

    all_spots = pd.concat(all_spots, ignore_index=True)
    return all_spots, norm_shifts


@slurm_it(conda_env="iss-preprocess")
def estimate_channel_correction(
    data_path, prefix="genes_round", nrounds=7, fit_norm_factors=False
):
    """Compute grayscale value distribution and normalisation factors

    Each `correction_tiles` of `ops` is filtered before being used to compute the
    distribution of pixel values.
    Normalisation factor to equalise these distribution across channels and rounds are
    defined as `ops["correction_quantile"]` of the distribution.

    Args:
        data_path (str or Path): Relative path to the data folder
        prefix (str, optional): Folder name prefix, before round number. Defaults
            to "genes_round".
        nrounds (int, optional): Number of rounds. Defaults to 7.

    Returns:
        pixel_dist (np.array): A 65536 x Nch x Nrounds distribution of grayscale values
            for filtered stacks
        norm_factors (np.array) A Nch x Nround array of normalisation factors

    """
    ops = load_ops(data_path)
    nch = len(ops["camera_order"])
    if nrounds is None:
        nrounds = ops[f"{prefix.split('_')[0]}_rounds"]

    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch, nrounds))
    if prefix == "genes_round":
        projection = ops["genes_projection"]
    elif prefix == "barcode_round":
        projection = ops["barcode_projection"]
    else:
        raise ValueError("prefix must be 'genes_round' or 'barcode_round'")
    corr_tiles = ops.get("correction_tiles", None)
    if corr_tiles is None:
        print("No correction tiles specified - using ref tiles")
        corr_tiles = ops[f"{prefix.split('_')[0]}_ref_tiles"]
        assert corr_tiles is not None, "No ref tiles specified"

    for tile in corr_tiles:
        print(f"counting pixel values for roi {tile[0]}, tile {tile[1]}, {tile[2]}")
        try:
            stack = load_sequencing_rounds(
                data_path, tile, suffix=projection, prefix=prefix, nrounds=nrounds
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Tile {tile} not found. Is ops['correction_tiles'] correct?"
            )
        stack = filter_stack(
            stack,
            r1=ops["filter_r"][0],
            r2=ops["filter_r"][1],
        )
        stack[stack < 0] = 0
        for iround in range(nrounds):
            pixel_dist[:, :, iround] += compute_distribution(
                stack[:, :, :, iround], max_value=max_val
            )

    cumulative_pixel_dist = np.cumsum(pixel_dist, axis=0)
    cumulative_pixel_dist = cumulative_pixel_dist / cumulative_pixel_dist[-1, :, :]
    norm_factors_raw = np.zeros((nch, nrounds))
    for iround in range(nrounds):
        for ich in range(nch):
            norm_factors_raw[ich, iround] = np.argmax(
                cumulative_pixel_dist[:, ich, iround] > ops["correction_quantile"]
            )

    if fit_norm_factors:
        x_ch = np.repeat(np.arange(nch)[:, np.newaxis], nrounds, axis=1)
        x_round = np.repeat(np.arange(nrounds)[np.newaxis, :], nch, axis=0)
        channels_encoding = (
            OneHotEncoder().fit_transform(x_ch.flatten()[:, np.newaxis]).todense()
        )
        x = np.asarray(np.hstack((x_round.flatten()[:, np.newaxis], channels_encoding)))

        mdl = LinearRegression(fit_intercept=False).fit(
            x, np.log(norm_factors_raw.flatten()[:, np.newaxis])
        )
        norm_factors_fit = np.exp(mdl.predict(x))
        norm_factors_fit = np.reshape(norm_factors_fit, norm_factors_raw.shape)
    else:
        norm_factors_fit = norm_factors_raw

    save_path = get_processed_path(data_path) / f"correction_{prefix}.npz"
    np.savez(
        save_path,
        pixel_dist=pixel_dist,
        norm_factors=norm_factors_fit,
        norm_factors_raw=norm_factors_raw,
    )
    print(f"Saved pixel distribution and normalisation factors to {save_path}")
    print(f"plotting normalisation factors across rounds and channels")
    metadata = load_metadata(data_path)
    # pairwise matching hues for each channel
    channel_order = metadata["camera_order"]
    color = ['red', 'green', 'cyan', 'magenta']
    color = [color[i-1] for i in channel_order]

    plt.figure(figsize=(16, 8))
    for ch in range(0, norm_factors_raw.shape[0]):
        # ch = rank_order - 1  # zero-based
        plt.plot(norm_factors_fit[ch,:], label=f'Norm factors channel {ch}', linestyle='dashed', color=color[ch])
        plt.plot(norm_factors_raw[ch,:], label=f'Raw norm factors channel {ch}', color=color[ch])
        plt.xlabel('Round')
        # show each round on x
        plt.xticks(range(norm_factors_raw.shape[1]), range(1, norm_factors_raw.shape[1]+1))
        plt.ylabel('Normalization Factor')
        plt.legend()
    plt.title(f'Normalization Factors per Channel - {data_path}')
    plt.savefig(get_processed_path(data_path) / "figures" / f'normalization_factors_{prefix}.png')

    return pixel_dist, norm_factors_fit, norm_factors_raw


def compute_spot_sign_image(data_path, prefix="genes_round"):
    """Compute the reference spot sign image to use in spot calling. Save it to
    the processed data folder.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional):  Prefix of the sequencing read to use.
            Defaults to "genes_round".

    """
    ops = load_ops(data_path)
    processed_path = get_processed_path(data_path)
    total_spots = 0
    images = []
    for tile in ops["genes_ref_tiles"]:
        g, _ = run_omp_on_tile(
            data_path, ops["ref_tile"], ops, save_stack=False, prefix=prefix
        )
        spot_sign_image, n_spots = get_spot_shape(
            g, spot_xy=7, neighbor_filter_size=9, neighbor_threshold=15
        )
        images.append(spot_sign_image)
        total_spots += n_spots
    spot_sign_image = np.sum(np.stack(images, axis=2), axis=2) / total_spots
    spot_sign_image = apply_symmetry(spot_sign_image)
    np.save(processed_path / "spot_sign_image.npy", spot_sign_image)
    # TODO: move this call to a pipeline.py function
    # check_spot_sign_image(data_path)


def load_spot_sign_image(data_path, threshold, return_raw_image=False):
    """Load the reference spot sign image to use in spot calling. First, check
    if the spot sign image has been computed for the current dataset and use it
    if available. Otherwise, use the spot sign image saved in the repo.

    Args:
        data_path (str): Relative path to data.
        threshold (float): Absolute value threshold used to binarize the spot
            sign image.
        return_raw_image (bool, optional): Whether to return the raw spot sign
            image. Defaults to False.

    Returns:
        numpy.ndarray: Spot sign image after thresholding, containing -1, 0, or 1s.

    """
    processed_path = get_processed_path(data_path)
    spot_image_path = processed_path / "spot_sign_image.npy"
    if spot_image_path.exists():
        spot_sign_image = np.load(spot_image_path)
    else:
        print("No spot sign image for this dataset - using default.")
        spot_sign_image = np.load(
            Path(__file__).parent.parent / "call/spot_sign_image.npy"
        )
    if return_raw_image:
        return spot_sign_image

    spot_sign_image[np.abs(spot_sign_image) < threshold] = 0
    return spot_sign_image


def run_omp_on_tile(data_path, tile_coors, ops, save_stack=False, prefix="genes_round"):
    """
    Run OMP on a tile and return the results.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of the tile to process.
        ops (dict): Dictionary of parameters.
        save_stack (bool, optional): Whether to save the registered stack.
            Defaults to False.
        prefix (str, optional): Prefix of the sequencing read to use.
            Defaults to "genes_round".

    Returns:
        numpy.ndarray: OMP results.
        dict: Dictionary of OMP parameters.

    """
    processed_path = get_processed_path(data_path)

    stack, bad_pixels = load_and_register_sequencing_tile(
        data_path,
        tile_coors,
        suffix=ops["genes_projection"],
        correct_channels=ops["genes_correct_channels"],
        prefix=prefix,
        corrected_shifts=ops["corrected_shifts"],
        nrounds=ops["genes_rounds"],
        correct_illumination=True,
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), :]

    if save_stack:
        save_dir = processed_path / "reg"
        save_dir.mkdir(parents=True, exist_ok=True)
        stack_path = (
            save_dir / f"tile_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.tif"
        )
        write_stack(stack.copy(), stack_path, bigtiff=True)

    omp_stat = np.load(processed_path / "gene_dict.npz", allow_pickle=True)
    g, _, _ = run_omp(
        stack,
        omp_stat["gene_dict"],
        tol=ops["omp_threshold"],
        weighted=True,
        refit_background=True,
        alpha=ops["omp_alpha"],
        beta_squared=ops["omp_beta_squared"],
        norm_shift=omp_stat["norm_shift"],
        max_comp=ops["omp_max_genes"],
        min_intensity=ops["omp_min_intensity"],
    )

    for igene in range(g.shape[2]):
        g[bad_pixels, igene] = 0

    return g, omp_stat


def detect_genes_on_tile(data_path, tile_coors, save_stack=False, prefix="genes_round"):
    """Apply the OMP algorithm to unmix spots in a given tile using the saved
    gene dictionary and settings saved in `ops.yml`. Then detect gene spots in
    the resulting gene maps.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of tile to load: ROI, Xpos, Ypos.
        save_stack (bool, optional): Whether to save registered and preprocessed images.
            Defaults to False.
        prefix (str, optional): Prefix of the sequencing read to analyse.
            Defaults to "genes_round".

    """
    ops = load_ops(data_path)
    g, omp_stat = run_omp_on_tile(
        data_path, tile_coors, ops, save_stack=save_stack, prefix=prefix
    )

    spot_sign_image = load_spot_sign_image(data_path, ops["spot_shape_threshold"])
    gene_spots, gene_coefficients = find_gene_spots(
        g,
        spot_sign_image,
        gene_names=omp_stat["gene_names"],
        rho=ops["genes_spot_rho"],
        spot_score_threshold=ops["genes_spot_score_threshold"],
    )

    for df, gene in zip(gene_spots, omp_stat["gene_names"]):
        df["gene"] = gene
    save_dir = get_processed_path(data_path) / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_spots).to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
    gene_coefficients.to_pickle(
        save_dir
        / f"{prefix}_coefficients_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
