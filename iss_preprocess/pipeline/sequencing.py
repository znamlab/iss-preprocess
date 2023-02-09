import numpy as np
import pandas as pd
from flexiznam.config import PARAMETERS
from pathlib import Path
from skimage.morphology import binary_dilation
from ..image import (
    filter_stack,
    apply_illumination_correction,
    compute_distribution,
)
from ..reg import (
    align_channels_and_rounds,
    generate_channel_round_transforms,
)
from ..io import (
    write_stack,
    load_tile_by_coors,
    load_ops,
)
from ..segment import detect_isolated_spots
from ..call import (
    extract_spots,
    make_gene_templates,
    run_omp,
    find_gene_spots,
    detect_spots_by_shape,
    get_cluster_means,
    barcode_spots_dot_product,
    BASES,
)


# AB: LGTM 10/03/23
def load_sequencing_rounds(
    data_path, tile_coors=(1, 0, 0), nrounds=7, suffix="fstack", prefix="round"
):
    """Load processed tile images across rounds

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        nrounds (int, optional): Number of rounds to load. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to 'fstack'.
        prefix (str, optional): the folder name prefix, before round number. Defaults
            to "round"

    Returns:
        numpy.ndarray: X x Y x channels x rounds stack.

    """
    ims = []
    for iround in range(nrounds):
        dirname = f"{prefix}_{iround+1}_1"
        ims.append(
            load_tile_by_coors(
                data_path, tile_coors=tile_coors, suffix=suffix, prefix=dirname
            )
        )
    return np.stack(ims, axis=3)


def setup_barcode_calling(
    data_path, score_thresh=0.5, spot_size=2, correct_channels=False
):
    """Detect spot and compute cluster means

    Args:
        data_path (str): Relative path to data
        score_thresh (float, optional): score_thresh argument for get_cluster_mean.
            Defaults to 0.5.
        spot_size (int, optional): Size of the spots in pixels. Defaults to 2.
        correct_channels (bool, optional): Correct intensity difference across channel.
            True to normalise all rounds individually. `round1_only` to normalise all
            rounds to round1 correction. False to remove correction. Defaults to False.

    Returns:
        cluster_means (list): A list with Nrounds elements. Each a Nch x Ncl (square
            because N channels is equal to N clusters) array of cluster means,
            normalised by round 0 intensity
        all_spots (pandas.DataFrame): All detected spots.
    """
    ops = load_ops(data_path)
    all_spots = []
    for ref_tile in ops["barcode_ref_tiles"]:
        print(f"detecting spots in tile {ref_tile}")
        stack, _ = load_and_register_tile(
            data_path,
            ref_tile,
            filter_r=ops["filter_r"],
            prefix="barcode_round",
            suffix=ops["projection"],
            nrounds=ops["barcode_rounds"],
            correct_channels=correct_channels,
            corrected_shifts=True,
            correct_illumination=False,
        )
        stack = stack[:, :, np.argsort(ops["camera_order"]), :]
        spots = detect_isolated_spots(
            np.std(stack, axis=(2, 3)),
            detection_threshold=ops["barcode_detection_threshold"],
            isolation_threshold=ops["barcode_isolation_threshold"],
        )
        spots["size"] = np.ones(len(spots)) * spot_size
        extract_spots(spots, stack)
        all_spots.append(spots)
    all_spots = pd.concat(all_spots, ignore_index=True)
    cluster_means = get_cluster_means(all_spots, vis=True, score_thresh=score_thresh)
    return cluster_means, all_spots


def basecall_tile(data_path, tile_coors):
    """Detect and basecall barcodes for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    cluster_means = np.load(processed_path / data_path / "barcode_cluster_means.npy")
    nrounds = cluster_means.shape[0]

    stack, bad_pixels = load_and_register_tile(
        data_path,
        tile_coors,
        filter_r=ops["filter_r"],
        prefix="barcode_round",
        suffix=ops["projection"],
        nrounds=nrounds,
        correct_channels=ops["barcode_correct_channels"],
        corrected_shifts=True,
        correct_illumination=True,
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), :]
    stack[bad_pixels, :, :] = 0
    spot_sign_image = load_spot_sign_image(data_path, ops["spot_threshold"])
    spots = detect_spots_by_shape(
        np.mean(stack, axis=(2, 3)),
        spot_sign_image,
        threshold=ops["barcode_detection_threshold_basecalling"],
        rho=ops["barcode_spot_rho"],
    )
    # TODO: size should probably be set inside detect spots?
    spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
    extract_spots(spots, stack)
    x = np.stack(spots["trace"], axis=2)
    cluster_inds = []
    top_score = []

    nrounds = x.shape[0]
    # TODO: perhaps we should apply background correction before basecalling?
    for iround in range(nrounds):
        this_round_means = cluster_means[iround]
        this_round_means = this_round_means / np.linalg.norm(this_round_means, axis=1)
        x_norm = (
            x[iround, :, :].T / np.linalg.norm(x[iround, :, :].T, axis=1)[:, np.newaxis]
        )
        # should be Spots x Channels matrix @ Channels x Clusters matrix
        score = x_norm @ this_round_means.T
        cluster_ind = np.argmax(score, axis=1)
        cluster_inds.append(cluster_ind)
        top_score.append(np.squeeze(score[np.arange(x_norm.shape[0]), cluster_ind]))

    mean_score = np.mean(np.stack(top_score, axis=1), axis=1)
    sequences = np.stack(cluster_inds, axis=1)
    spots["sequence"] = [seq for seq in sequences]
    scores = np.stack(top_score, axis=1)
    spots["scores"] = [s for s in scores]
    spots["mean_score"] = mean_score
    spots["bases"] = ["".join(BASES[seq]) for seq in spots["sequence"]]
    spots["dot_product_score"] = barcode_spots_dot_product(spots, cluster_means)
    spots["mean_intensity"] = [np.mean(np.abs(trace)) for trace in spots["trace"]]
    save_dir = processed_path / data_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    spots.to_pickle(
        save_dir
        / f"barcode_round_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )


def setup_omp(
    stack,
    codebook_name="codebook_83gene_pool.csv",
    detection_threshold=40,
    isolation_threshold=30,
):
    """Prepare variables required to run the OMP algorithm. Finds isolated spots using
    STD across rounds and channels. Detected spots are then used to determine the
    bleedthrough matrix using scaled k-means.

    Args:
        stack (numpy.ndarray): X x Y x C x R image stack.
        codebook_name (str): filename of the codebook to use.
        detection_threshold (float): spot detection threshold to find spots for
            training the bleedthrough matrix.
        isolation_threshold (float): threshold used to selected isolated spots
            based on average values in the annulus around each spot.

    Returns:
        numpy.ndarray: N x M dictionary, where N = R * C and M is the
            number of genes.
        list: gene names.
        float: norm shift for the OMP algorithm, estimated as median norm of all pixels.

    """
    spots = detect_isolated_spots(
        np.std(stack, axis=(2, 3)),
        detection_threshold=detection_threshold,
        isolation_threshold=isolation_threshold,
    )

    extract_spots(spots, stack)
    cluster_means = get_cluster_means(spots, vis=True)
    codebook = pd.read_csv(
        Path(__file__).parent.parent / "call" / codebook_name,
        header=None,
        names=["gii", "seq", "gene"],
    )
    gene_dict, unique_genes = make_gene_templates(cluster_means, codebook, vis=True)

    norm_shift = np.sqrt(np.median(np.sum(stack**2, axis=(2, 3))))
    return gene_dict, unique_genes, norm_shift


# AB: LGTM
def estimate_channel_correction(data_path, prefix="genes_round", nrounds=7):
    """Compute grayscale value distribution and normalisation factors

    Each `correction_tiles` of `ops` is filtered before being used to compute the
    distribution of pixel values.
    Normalisation factor to equalise these distribution across channels and rounds are
    defined as `ops["correction_quantile"]` of the distribution.

    Args:
        data_path (str or Path): Relative path to the data folder
        prefix (str, optional): Folder name prefix, before round number. Defaults
            to "round".
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

    for tile in ops["correction_tiles"]:
        print(f"counting pixel values for roi {tile[0]}, tile {tile[1]}, {tile[2]}")
        stack = filter_stack(
            load_sequencing_rounds(
                data_path,
                tile,
                suffix=ops["projection"],
                prefix=prefix,
                nrounds=nrounds,
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
    norm_factors = np.zeros((nch, nrounds))
    for iround in range(nrounds):
        for ich in range(nch):
            norm_factors[ich, iround] = np.argmax(
                cumulative_pixel_dist[:, ich, iround] > ops["correction_quantile"]
            )
    return pixel_dist, norm_factors


def load_and_register_tile(
    data_path,
    tile_coors=(1, 0, 0),
    prefix="genes_round",
    suffix="fstack",
    filter_r=(2, 4),
    correct_channels=False,
    corrected_shifts=True,
    correct_illumination=False,
    nrounds=7,
):
    """Load sequencing tile and align channels. Optionally, filter, correct
    illumination and channel brightness.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple, options): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1, 0, 0).
        prefix (str, optional): Prefix of the hybridisation round.
            Defaults to "hybridisation_1_1".
        suffix (str, optional): Filename suffix corresponding to the z-projection
            to use. Defaults to "fstack".
        filter_r (tuple, optional): Inner and out radius for the hanning filter.
            If `False`, stack is not filtered. Defaults to (2, 4).
        correct_channels (bool, optional): Whether to normalize channel brightness.
            Defaults to False.
        corrected_shifts (bool, optional): Whether to use corrected shifts estimated
            by robust regression across tiles. Defaults to True.
        correct_illumination (bool, optional): Whether to correct vignetting.
            Defaults to False.
        nrounds (int, optional): Number of sequencing rounds to load. Defaults to 7.

    Returns:
        numpy.ndarray: X x Y x Nch x Nrounds image stack.
        numpy.ndarray: X x Y boolean mask, identifying bad pixels that we were not imaged
            for all channels and rounds (due to registration offsets) and should be discarded
            during analysis.
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    stack = load_sequencing_rounds(
        data_path, tile_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)

    if corrected_shifts:
        tforms_fname = f"tforms_corrected_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
        tforms_path = processed_path / data_path / "reg" / tforms_fname
    else:
        tforms_fname = f"tforms_{prefix}.npz"
        tforms_path = processed_path / data_path / tforms_fname
    tforms = np.load(tforms_path, allow_pickle=True)
    tforms = generate_channel_round_transforms(
        tforms["angles_within_channels"],
        tforms["shifts_within_channels"],
        tforms["scales_between_channels"],
        tforms["angles_between_channels"],
        tforms["shifts_between_channels"],
        stack.shape[:2],
    )
    stack = align_channels_and_rounds(stack, tforms)

    bad_pixels = np.any(np.isnan(stack), axis=(2, 3))
    stack[np.isnan(stack)] = 0
    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
        mask = np.ones((filter_r[1] * 2 + 1, filter_r[1] * 2 + 1))
        bad_pixels = binary_dilation(bad_pixels, mask)
    if correct_channels:
        correction_path = processed_path / data_path / f"correction_{prefix}.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        if correct_channels == "round1_only":
            stack = stack / norm_factors[np.newaxis, np.newaxis, :, 0, np.newaxis]
        else:
            stack = stack / norm_factors[np.newaxis, np.newaxis, :, :nrounds]

    return stack, bad_pixels


def load_spot_sign_image(data_path, threshold):
    """Load the reference spot sign image to use in spot calling. First, check
    if the spot sign image has been computed for the current dataset and use it
    if available. Otherwise, use the spot sign image saved in the repo.

    Args:
        data_path (str): Relative path to data.
        threshold (float): Absolute value threshold used to binarize the spot
            sign image.

    Returns:
        numpy.ndarray: Spot sign image after thresholding, containing -1, 0, or 1s.
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    spot_image_path = processed_path / data_path / "spot_sign_image.npy"
    if spot_image_path.exists():
        spot_sign_image = np.load(spot_image_path)
    else:
        print("No spot sign image for this dataset - using default.")
        spot_sign_image = np.load(
            Path(__file__).parent.parent / "call/spot_signimage.npy"
        )
    spot_sign_image[np.abs(spot_sign_image) < threshold] = 0
    return spot_sign_image


def run_omp_on_tile(
    data_path,
    tile_coors,
    save_stack=False,
    correct_channels=False,
    prefix="genes_round",
):
    """Apply the OMP algorithm to unmix spots in a given tile using the saved
    gene dictionary and settings saved in `ops.npy`. Then detect gene spots in
    the resulting gene maps.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of tile to load: ROI, Xpos, Ypos.
        save_stack (bool, optional): Whether to save registered and preprocessed images.
            Defaults to False.
        correct_channels (bool or str, optional): Whether to apply channel normalization.
            If not False, can specify normalization method, e.g. "round1_only". Defaults to False.
        prefix (str, optional): Prefix of the sequencing read to analyse.
            Defaults to "genes_round".
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)

    stack, bad_pixels = load_and_register_tile(
        data_path,
        tile_coors,
        suffix=ops["projection"],
        correct_channels=correct_channels,
        prefix=prefix,
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), :]

    if save_stack:
        save_dir = processed_path / data_path / "reg"
        save_dir.mkdir(parents=True, exist_ok=True)
        stack_path = (
            save_dir / f"tile_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.tif"
        )
        write_stack(stack.copy(), stack_path, bigtiff=True)

    omp_stat = np.load(processed_path / data_path / "gene_dict.npz", allow_pickle=True)
    g, _, _ = run_omp(
        stack,
        omp_stat["gene_dict"],
        tol=ops["omp_threshold"],
        weighted=True,
        refit_background=True,
        alpha=ops["omp_alpha"],
        norm_shift=omp_stat["norm_shift"],
        max_comp=ops["omp_max_genes"],
        min_intensity=ops["omp_min_intensity"],
    )

    for igene in range(g.shape[2]):
        g[bad_pixels, igene] = 0

    spot_sign_image = load_spot_sign_image(data_path, ops["spot_threshold"])
    gene_spots = find_gene_spots(
        g,
        spot_sign_image,
        rho=ops["spot_rho"],
        spot_score_threshold=ops["spot_threshold"],
    )

    for df, gene in zip(gene_spots, omp_stat["gene_names"]):
        df["gene"] = gene
    save_dir = processed_path / data_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_spots).to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
