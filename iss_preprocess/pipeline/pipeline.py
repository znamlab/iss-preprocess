import subprocess, shlex
import warnings
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from flexiznam.config import PARAMETERS
from pathlib import Path
from skimage.morphology import binary_dilation
from ..image import (
    filter_stack,
    apply_illumination_correction,
    tilestats_and_mean_image,
    compute_distribution,
)
from ..reg import (
    align_channels_and_rounds,
    generate_channel_round_transforms,
    apply_corrections,
)
from ..io import (
    write_stack,
    load_tile_by_coors,
    load_metadata,
    load_hyb_probes_metadata,
    load_ops,
)
from ..segment import detect_isolated_spots, detect_spots
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
from ..coppafish import scaled_k_means


def setup_hyb_spot_calling(data_path, score_thresh=0, vis=True):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    metadata = load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        cluster_means, _, genes = hyb_spot_cluster_means(
            data_path, hyb_round, score_thresh=score_thresh
        )
        if vis:
            plt.figure()
            plt.imshow(cluster_means)
            plt.title(hyb_round)
            plt.yticks(ticks=range(cluster_means.shape[0]), labels=genes)
        save_path = processed_path / data_path / f"{hyb_round}_cluster_means.npz"
        np.savez(save_path, cluster_means=cluster_means, genes=genes)


def hyb_spot_cluster_means(
    data_path,
    prefix,
    score_thresh=0,
    init_spot_colors=np.array([[0, 1, 0, 0], [0, 0, 0, 1]]),
):
    ops = load_ops(data_path)

    nch = len(ops["black_level"])
    metadata = load_metadata(data_path=data_path)
    hyb_probes = load_hyb_probes_metadata()
    init_spot_colors = []
    genes = []
    for probe in metadata["hybridisation"][prefix]["probes"]:
        this_probe = np.zeros(nch)
        this_probe[hyb_probes[probe]["channel"] - 1] = 1
        init_spot_colors.append(this_probe)
        genes.append(hyb_probes[probe]["target"])
    init_spot_colors = np.array(init_spot_colors)

    all_spots = []
    for ref_tile in ops["barcode_ref_tiles"]:
        print(f"detecting spots in tile {ref_tile}")
        stack, _ = load_and_register_hyb_tile(
            data_path,
            tile_coors=ref_tile,
            prefix=prefix,
            correct_channels=ops["hybridisation_correct_channels"],
        )
        spots = detect_spots(
            np.max(stack, axis=2), threshold=ops["hyb_spot_detection_threshold"]
        )
        stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
        spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
        extract_spots(spots, stack)
        all_spots.append(spots)

    all_spots = pd.concat(all_spots, ignore_index=True)
    x = np.stack(all_spots["trace"], axis=2)

    cluster_means, _, _, _, _, _ = scaled_k_means(
        x[0, :, :].T, init_spot_colors, score_thresh=score_thresh
    )
    return cluster_means, all_spots, genes


def extract_hyb_spots_all(data_path):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ops = load_ops(data_path)
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    metadata = load_metadata(data_path)
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "extract_hyb_spots.sh"
    )

    for hyb_round in metadata["hybridisation"].keys():
        for roi in roi_dims[use_rois, :]:
            args = f"--export=DATAPATH={data_path},ROI={roi[0]},PREFIX={hyb_round}"
            args = args + f" --output={Path.home()}/slurm_logs/iss_hyb_spots_%j.out"
            command = f"sbatch {args} {script_path}"
            print(command)
            subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )


def extract_hyb_spots_roi(data_path, prefix, roi):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            extract_hyb_spots_tile(data_path, (roi, ix, iy), prefix)


def extract_hyb_spots_tile(data_path, tile_coors, prefix):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    clusters = np.load(
        processed_path / data_path / f"{prefix}_cluster_means.npz", allow_pickle=True
    )
    print(f"detecting spots in tile {tile_coors}")
    stack, _ = load_and_register_hyb_tile(
        data_path,
        tile_coors=tile_coors,
        prefix=prefix,
        suffix=ops["hybridisation_projection"],
        correct_illumination=True,
        correct_channels=ops["hybridisation_correct_channels"],
    )
    spots = detect_spots(
        np.max(stack, axis=2), threshold=ops["hyb_spot_detection_threshold"]
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
    spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
    extract_spots(spots, stack)
    x = np.stack(spots["trace"], axis=2)
    x_norm = x[0, :, :].T / np.linalg.norm(x[0, :, :].T, axis=1)[:, np.newaxis]
    score = x_norm @ clusters["cluster_means"].T
    cluster_ind = np.argmax(score, axis=1)
    spots["cluster"] = cluster_ind
    spots["gene"] = clusters["genes"][cluster_ind]
    spots["score"] = np.squeeze(score[np.arange(x_norm.shape[0]), cluster_ind])
    spots["mean_intensity"] = [np.max(trace) for trace in spots["trace"]]

    save_dir = processed_path / data_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    spots.to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )


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
        stack, bad_pixels = load_and_register_tile(
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
    """Prepare variables required to run the OMP algorithm.

    Args:
        stack (numpy.ndarray): X x Y x C x R image stack.

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


def save_roi_dimensions(data_path, prefix):
    rois_list = get_roi_dimensions(data_path, prefix)

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    (processed_path / data_path).mkdir(parents=True, exist_ok=True)

    np.save(processed_path / data_path / "roi_dims.npy", rois_list)


def get_roi_dimensions(data_path, prefix):
    """Find imaging ROIs and determine their dimensions.

    Args:
        data_path (str): path to data in the raw data directory.
        prefix (str): directory and file name prefix, e.g. 'round_01_1'

    """
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    data_dir = raw_path / data_path / prefix
    fnames = [p.name for p in data_dir.glob("*.tif")]
    pattern = rf"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d).ome.tif"
    matcher = re.compile(pattern=pattern)
    tile_coors = np.stack(
        [np.array(matcher.match(fname).groups(), dtype=int) for fname in fnames]
    )

    rois = np.unique(tile_coors[:, 0])
    roi_list = []
    for roi in rois:
        roi_list.append(
            [
                roi,
                np.max(tile_coors[tile_coors[:, 0] == roi, 1]),
                np.max(tile_coors[tile_coors[:, 0] == roi, 2]),
            ]
        )

    return roi_list


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


def estimate_channel_correction_hybridisation(data_path):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    nch = len(ops["black_level"])

    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch))
    metadata = load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        for tile in ops["correction_tiles"]:
            print(
                f"counting pixel values for {hyb_round}, roi {tile[0]}, tile {tile[1]}, {tile[2]}"
            )
            stack = filter_stack(
                load_tile_by_coors(
                    data_path,
                    tile_coors=tile,
                    suffix=ops["projection"],
                    prefix=hyb_round,
                ),
                r1=ops["filter_r"][0],
                r2=ops["filter_r"][1],
            )
            stack[stack < 0] = 0
            pixel_dist += compute_distribution(stack, max_value=max_val)

        cumulative_pixel_dist = np.cumsum(pixel_dist, axis=0)
        cumulative_pixel_dist = cumulative_pixel_dist / cumulative_pixel_dist[-1, :]
        norm_factors = np.zeros(nch)
        for ich in range(nch):
            norm_factors[ich] = np.argmax(
                cumulative_pixel_dist[:, ich] > ops["correction_quantile"]
            )

        save_path = processed_path / data_path / f"correction_{hyb_round}.npz"
        np.savez(save_path, pixel_dist=pixel_dist, norm_factors=norm_factors)


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
        norm_factors (np.array) A Nch x Nround array of normalising factors
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


def load_and_register_hyb_tile(
    data_path,
    tile_coors=(0, 0, 0),
    prefix="hybridisation_1_1",
    suffix="fstack",
    filter_r=(2, 4),
    correct_illumination=False,
    correct_channels=False,
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    tforms_fname = (
        f"tforms_corrected_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
    )
    tforms = np.load(
        processed_path / data_path / "reg" / tforms_fname, allow_pickle=True
    )
    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
    )
    stack = apply_corrections(
        stack, tforms["scales"], tforms["angles"], tforms["shifts"], cval=np.nan
    )
    bad_pixels = np.any(np.isnan(stack), axis=(2))
    stack[np.isnan(stack)] = 0
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)
    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
        mask = np.ones((filter_r[1] * 2 + 1, filter_r[1] * 2 + 1))
        bad_pixels = binary_dilation(bad_pixels, mask)
    if correct_channels:
        correction_path = processed_path / data_path / f"correction_{prefix}.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        stack = stack / norm_factors[np.newaxis, np.newaxis, :]
    return stack, bad_pixels


def load_and_register_tile(
    data_path,
    tile_coors=(0, 0, 0),
    prefix="genes_round",
    suffix="fstack",
    filter_r=(2, 4),
    correct_channels=False,
    corrected_shifts=True,
    correct_illumination=False,
    nrounds=7,
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    if corrected_shifts:
        tforms_fname = f"tforms_corrected_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
        tforms_path = processed_path / data_path / "reg" / tforms_fname
    else:
        tforms_fname = f"tforms_{prefix}.npz"
        tforms_path = processed_path / data_path / tforms_fname
    tforms = np.load(tforms_path, allow_pickle=True)

    stack = load_sequencing_rounds(
        data_path, tile_coors, suffix=suffix, prefix=prefix, nrounds=nrounds
    )

    tforms = generate_channel_round_transforms(
        tforms["angles_within_channels"],
        tforms["shifts_within_channels"],
        tforms["scales_between_channels"],
        tforms["angles_between_channels"],
        tforms["shifts_between_channels"],
        stack.shape[:2],
    )
    stack = align_channels_and_rounds(stack, tforms)
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)
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
            stack = stack / norm_factors[np.newaxis, np.newaxis, :, :]

    return stack, bad_pixels


def batch_process_tiles(data_path, script, additional_args=""):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    script_path = str(Path(__file__).parent.parent.parent / "scripts" / f"{script}.sh")
    ops = load_ops(data_path)
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                args = (
                    f"--export=DATAPATH={data_path},ROI={roi[0]},TILEX={ix},TILEY={iy}"
                )
                args = args + additional_args
                args = args + f" --output={Path.home()}/slurm_logs/iss_{script}_%j.out"
                command = f"sbatch {args} {script_path}"
                print(command)
                subprocess.Popen(
                    shlex.split(command),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )


def load_spot_sign_image(data_path, threshold):
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


def create_single_average(
    data_path,
    subfolder,
    subtract_black,
    prefix_filter=None,
    suffix=None,
    combine_tilestats=False,
):
    """Create normalised average of all tifs in a single folder.

    If prefix_filter is not None, the output will be "{prefix_filter}_average.tif",
    otherwise it will be "{folder_path.name}_average.tif"

    Other arguments are read from `ops`:
        average_clip_value: Value to clip images before averaging.
        normalise: Normalise output maximum to one.

    Args:
        data_path (str): Path to the acquisition folder, relative to `projects` folder
        subfolder (str): subfolder in folder_path containing the tifs to average.
        subtract_black (bool): Subtract black level (read from `ops`)
        prefix_filter (str, optional): prefix name to filter tifs. Only file starting
            with `prefix` will be averaged. Defaults to None.
        suffix (str, optional): suffix to filter tifs. Defaults to None
        combine_tilestats (bool, optional): Compute new tilestats distribution of
            averaged images if True, combine pre-existing tilestats into one otherwise.
            Defaults to False

    Returns:
        np.array: Average image
        np.array: Distribution of pixel values
    """

    print("Creating single average")
    print("  Args:")
    print(f"    data_path={data_path}")
    print(f"    subfolder={subfolder}")
    print(f"    subtract_black={subtract_black}")
    print(f"    prefix_filter={prefix_filter}")
    print(f"    suffix={suffix}")
    print(f"    combine_tilestats={combine_tilestats}", flush=True)

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    if prefix_filter is None:
        target_file = f"{subfolder}_average.tif"
    else:
        target_file = f"{prefix_filter}_average.tif"
    target_stats = target_file.replace("_average.tif", "_tilestats.npy")
    target_file = processed_path / data_path / "averages" / target_file
    target_stats = processed_path / data_path / "averages" / target_stats
    # ensure the directory exists for first average.
    target_file.parent.mkdir(exist_ok=True)

    black_level = ops["black_level"] if subtract_black else 0

    av_image, tilestats = tilestats_and_mean_image(
        processed_path / data_path / subfolder,
        prefix=prefix_filter,
        black_level=black_level,
        max_value=ops["average_clip_value"],
        verbose=True,
        median_filter=ops["average_median_filter"],
        normalise=True,
        suffix=suffix,
        combine_tilestats=combine_tilestats,
    )
    write_stack(av_image, target_file, bigtiff=False, dtype="float", clip=False)
    np.save(target_stats, tilestats)
    print(f"Average saved to {target_file}, tilestats to {target_stats}", flush=True)
    return av_image, tilestats


def create_all_single_averages(
    data_path,
    todo=("genes_rounds", "barcode_rounds", "fluorescence", "hybridisation"),
):
    """Average all tiffs in each folder and then all folders by acquisition type

    Args:
        data_path (str): Path to data, relative to project.
        todo (tuple): type of acquisition to process. Default to `("genes_rounds",
            "barcode_rounds", "fluorescence", "hybridisation")`
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    metadata = load_metadata(data_path)

    # Collect all folder names
    to_average = []
    for kind in todo:
        if kind.endswith("rounds"):
            folders = [f"{kind[:-1]}_{acq + 1}_1" for acq in range(metadata[kind])]
            to_average.extend(folders)
        elif kind in ("fluorescence", "hybridisation"):
            to_average.extend(list(metadata[kind].keys()))
        else:
            raise IOError(
                f"Unknown type of acquisition: {kind}.\n"
                + "Valid types are 'XXXXX_rounds', 'fluorescence', 'hybridisation'"
            )

    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "create_single_average.sh"
    )
    for folder in to_average:
        data_folder = processed_path / data_path
        if not data_folder.is_dir():
            warnings.warn("{0} does not exists. Skipping".format(data_folder / folder))
            continue
        export_args = dict(
            DATAPATH=data_path,
            SUBFOLDER=folder,
            SUFFIX=ops["projection"],
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_create_single_average_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_create_single_average_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def create_grand_averages(
    data_path,
    prefix_todo=("genes_round", "barcode_round"),
):
    """Average single acquisition averages into grand average

    Args:
        data_path (str): Path to the folder, relative to `projects` folder
        prefix_todo (tuple, optional): List of str, names of the tifs to average.
            Defaults to ("genes_round", "barcode_round").
    """
    subfolder = "averages"
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "create_grand_average.sh"
    )
    for kind in prefix_todo:
        export_args = dict(
            DATAPATH=data_path,
            SUBFOLDER=subfolder,
            PREFIX=kind,
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_create_grand_average_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_create_grand_average_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
