from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from znamutils import slurm_it

import iss_preprocess as iss

from ..call import (
    BASES,
    apply_symmetry,
    barcode_spots_dot_product,
    detect_spots_by_shape,
    extract_spots,
    find_gene_spots,
    get_cluster_means,
    get_spot_shape,
    make_gene_templates,
    run_omp,
)
from ..image import (
    apply_illumination_correction,
    compute_distribution,
    filter_stack,
)
from ..io import (
    load_ops,
    load_tile_by_coors,
    write_stack,
)
from ..reg import (
    align_channels_and_rounds,
    generate_channel_round_transforms,
)
from ..segment import detect_isolated_spots


def load_sequencing_rounds(
    data_path,
    tile_coors=(1, 0, 0),
    nrounds=7,
    suffix="max",
    prefix="genes_round",
    specific_rounds=None,
):
    """Load processed tile images across rounds

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        nrounds (int, optional): Number of rounds to load. Used only if
            `specific_rounds` is None. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to 'fstack'.
        prefix (str, optional): the folder name prefix, before round number. Defaults
            to "genes_round"
        specific_round (list, optional): if not None, specify which rounds must be
            loaded and ignores `nrounds`. Defaults to None

    Returns:
        numpy.ndarray: X x Y x channels x rounds stack.

    """
    if specific_rounds is None:
        specific_rounds = np.arange(nrounds) + 1

    ims = []
    for iround in specific_rounds:
        dirname = f"{prefix}_{iround}_1"
        ims.append(
            load_tile_by_coors(
                data_path, tile_coors=tile_coors, suffix=suffix, prefix=dirname
            )
        )
    return np.stack(ims, axis=3)


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
    ops = load_ops(data_path)
    all_spots, _ = get_reference_spots(data_path, prefix="barcode")
    cluster_means, spot_colors, cluster_inds = get_cluster_means(
        all_spots,
        score_thresh=ops["barcode_cluster_score_thresh"],
        initial_cluster_mean=np.array(ops["initial_cluster_means"]),
    )
    processed_path = iss.io.get_processed_path(data_path)
    np.savez(
        processed_path / "reference_barcode_spots.npz",
        spot_colors=spot_colors,
        cluster_inds=cluster_inds,
    )
    np.save(processed_path / "barcode_cluster_means.npy", cluster_means)
    iss.pipeline.check_barcode_calling(data_path)
    return cluster_means, all_spots


def basecall_tile(data_path, tile_coors, save_spots=True):
    """Detect and basecall barcodes for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
        save_spots (bool, optional): Whether to save the detected spots. Used to run
            without erasing during diagnostics. Defaults to True.

    """
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    cluster_means = np.load(processed_path / "barcode_cluster_means.npy")

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
    stack[bad_pixels, :, :] = 0
    spot_sign_image = load_spot_sign_image(data_path, ops["spot_shape_threshold"])
    spots = detect_spots_by_shape(
        np.mean(stack, axis=(2, 3)),
        spot_sign_image,
        threshold=ops["barcode_detection_threshold_basecalling"],
        rho=ops["barcode_spot_rho"],
    )
    extract_spots(spots, stack, ops["spot_extraction_radius"])
    if len(spots) == 0:
        print(f"No spots detected in tile {tile_coors}")
        return stack, spot_sign_image, spots
    x = np.stack(spots["trace"], axis=2)
    cluster_inds = []
    top_score = []

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
        cluster_inds.append(cluster_ind)
        top_score.append(score[np.arange(x_norm.shape[0]), cluster_ind])

    mean_score = np.mean(np.stack(top_score, axis=1), axis=1)
    sequences = np.stack(cluster_inds, axis=1)
    spots["sequence"] = [seq for seq in sequences]
    scores = np.stack(top_score, axis=1)
    spots["scores"] = [s for s in scores]
    spots["mean_score"] = mean_score
    spots["bases"] = ["".join(BASES[seq]) for seq in spots["sequence"]]
    spots["dot_product_score"] = barcode_spots_dot_product(spots, cluster_means)
    spots["mean_intensity"] = [np.mean(np.abs(trace)) for trace in spots["trace"]]
    if save_spots:
        save_dir = processed_path / "spots"
        save_dir.mkdir(parents=True, exist_ok=True)
        spots.to_pickle(
            save_dir
            / f"barcode_round_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
        )
    return stack, spot_sign_image, spots


@slurm_it(conda_env="iss-preprocess")
def setup_omp(data_path):
    """Prepare variables required to run the OMP algorithm. Finds isolated spots using
    STD across rounds and channels. Detected spots are then used to determine the
    bleedthrough matrix using scaled k-means.

    Args:
        data_path (str): Relative path to data.

    Returns:
        numpy.ndarray: N x M dictionary, where N = R * C and M is the
            number of genes.
        list: gene names.
        float: norm shift for the OMP algorithm, estimated as median norm of all pixels.

    """
    ops = load_ops(data_path)
    processed_path = iss.io.get_processed_path(data_path)
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
    iss.pipeline.check_omp_setup(data_path)
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
        spots = detect_isolated_spots(
            np.mean(stack, axis=(2, 3)),
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
    nch = len(ops["black_level"])

    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch, nrounds))
    if prefix == "genes_round":
        projection = ops["genes_projection"]
    else:
        projection = ops["barcode_projection"]
    for tile in ops["correction_tiles"]:
        print(f"counting pixel values for roi {tile[0]}, tile {tile[1]}, {tile[2]}")
        stack = filter_stack(
            load_sequencing_rounds(
                data_path, tile, suffix=projection, prefix=prefix, nrounds=nrounds
            ),
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

    save_path = iss.io.get_processed_path(data_path) / f"correction_{prefix}.npz"
    np.savez(
        save_path,
        pixel_dist=pixel_dist,
        norm_factors=norm_factors_fit,
        norm_factors_raw=norm_factors_raw,
    )
    return pixel_dist, norm_factors_fit, norm_factors_raw


def load_and_register_sequencing_tile(
    data_path,
    tile_coors=(1, 0, 0),
    prefix="genes_round",
    suffix="max",
    filter_r=(2, 4),
    correct_channels=False,
    corrected_shifts="best",
    correct_illumination=False,
    nrounds=7,
    specific_rounds=None,
    zero_bad_pixels=False,
):
    """Load sequencing tile and align channels. Optionally, filter, correct
    illumination and channel brightness.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, options): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1, 0, 0).
        prefix (str, optional): Prefix of the sequencing round.
            Defaults to "genes_round".
        suffix (str, optional): Filename suffix corresponding to the z-projection
            to use. Defaults to "fstack".
        filter_r (tuple, optional): Inner and out radius for the hanning filter.
            If `False`, stack is not filtered. Defaults to (2, 4).
        correct_channels (bool or str, optional): Whether to normalize channel
            brightness. If 'round1_only', normalise by round 1 correction factor,
            otherwise, if True use all norm_factors. Defaults to False.
        corrected_shifts (str, optional): Which shift to use. One of `reference`,
            `single_tile`, `ransac`, or `best`. Defaults to 'best'.
        correct_illumination (bool, optional): Whether to correct vignetting.
            Defaults to False.
        nrounds (int, optional): Number of sequencing rounds to load. Used only if
            specific_rounds is None. Defaults to 7.
        specific_rounds (list, optional): if not None, specifies which rounds must be
            loaded and ignores `nrounds`. Defaults to None
        zero_bad_pixels (bool, optional): Whether to zero bad pixels. Defaults to False.

    Returns:
        numpy.ndarray: X x Y x Nch x len(specific_rounds) or Nrounds image stack.
        numpy.ndarray: X x Y boolean mask, identifying bad pixels that we were not imaged
            for all channels and rounds (due to registration offsets) and should be discarded
            during analysis.

    """
    if specific_rounds is None:
        specific_rounds = np.arange(nrounds) + 1
    elif isinstance(specific_rounds, int):
        specific_rounds = [specific_rounds]
    # ensure we have an array
    specific_rounds = np.asarray(specific_rounds, dtype=int)
    assert specific_rounds.min() > 0, "rounds must be strictly positive integers"
    valid_shifts = ["reference", "single_tile", "ransac", "best"]
    assert corrected_shifts in valid_shifts, (
        f"unknown shift correction method, must be one of {valid_shifts}",
    )

    processed_path = iss.io.get_processed_path(data_path)
    stack = load_sequencing_rounds(
        data_path,
        tile_coors,
        suffix=suffix,
        prefix=prefix,
        nrounds=nrounds,
        specific_rounds=specific_rounds,
    )
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)

    ops = iss.io.load_ops(data_path)
    tforms = get_channel_round_shifts(data_path, prefix, tile_coors, corrected_shifts)
    tforms = generate_channel_round_transforms(
        tforms["angles_within_channels"],
        tforms["shifts_within_channels"],
        tforms["matrix_between_channels"],
        stack.shape[:2],
        align_channels=ops["align_channels"],
        ref_ch=ops["ref_ch"],
    )
    tforms = tforms[:, specific_rounds - 1]
    stack = align_channels_and_rounds(stack, tforms)

    bad_pixels = np.any(np.isnan(stack), axis=(2, 3))
    if zero_bad_pixels:
        stack[np.isnan(stack)] = 0

    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
        mask = np.ones((filter_r[1] * 2 + 1, filter_r[1] * 2 + 1))
        bad_pixels = binary_dilation(bad_pixels, mask)
    if correct_channels:
        correction_path = processed_path / f"correction_{prefix}.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        if correct_channels == "round1_only":
            stack = stack / norm_factors[np.newaxis, np.newaxis, :, 0, np.newaxis]
        else:
            stack = stack / norm_factors[np.newaxis, np.newaxis, :, specific_rounds - 1]

    return stack, bad_pixels


def get_channel_round_shifts(data_path, prefix, tile_coors, corrected_shifts):
    """Load the channel and round shifts for a given tile and sequencing acquisition.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Prefix of the sequencing round.
        tile_coors (tuple): Coordinates of the tile to process.
        corrected_shifts (str): Which shift to use. One of `reference`, `single_tile`,
            `ransac`, or `best`.

    Returns:
        np.ndarray: Array of channel and round shifts.

    """
    processed_path = iss.io.get_processed_path(data_path)
    tile_name = f"{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}"
    match corrected_shifts:
        case "reference":
            tforms_fname = f"tforms_{prefix}.npz"
            tforms_path = processed_path
        case "single_tile":
            tforms_fname = f"tforms_{prefix}_{tile_name}.npz"
            tforms_path = processed_path / "reg"
        case "ransac":
            tforms_fname = f"tforms_corrected_{prefix}_{tile_name}.npz"
            tforms_path = processed_path / "reg"
        case "best":
            tforms_fname = f"tforms_best_{prefix}_{tile_name}.npz"
            tforms_path = processed_path / "reg"
        case _:
            raise ValueError(f"unknown shift correction method: {corrected_shifts}")
    tforms = np.load(tforms_path / tforms_fname, allow_pickle=True)
    return tforms


def compute_spot_sign_image(data_path, prefix="genes_round"):
    """Compute the reference spot sign image to use in spot calling. Save it to
    the processed data folder.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional):  Prefix of the sequencing read to use.
            Defaults to "genes_round".

    """
    ops = load_ops(data_path)
    processed_path = iss.io.get_processed_path(data_path)
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
    iss.pipeline.check_spot_sign_image(data_path)


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
    processed_path = iss.io.get_processed_path(data_path)
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
    processed_path = iss.io.get_processed_path(data_path)

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
    gene_spots = find_gene_spots(
        g,
        spot_sign_image,
        rho=ops["genes_spot_rho"],
        spot_score_threshold=ops["genes_spot_score_threshold"],
    )

    for df, gene in zip(gene_spots, omp_stat["gene_names"]):
        df["gene"] = gene
    save_dir = iss.io.get_processed_path(data_path) / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_spots).to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
