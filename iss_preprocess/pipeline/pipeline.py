import subprocess, shlex
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..image import filter_stack, apply_illumination_correction
from ..reg import (
    align_channels_and_rounds,
    generate_channel_round_transforms,
    apply_corrections,
)
from ..io import write_stack, load_tile_by_coors, load_metadata
from ..segment import detect_isolated_spots, detect_spots
from ..call import (
    extract_spots,
    make_gene_templates,
    run_omp,
    find_gene_spots,
    detect_spots_by_shape,
    get_cluster_means,
    rois_to_array,
    barcode_spots_dot_product,
    BASES,
)
from ..coppafish import scaled_k_means


def setup_hyb_spot_calling(data_path, score_thresh=0, vis=True):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    metadata = load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        cluster_means, _ = hyb_spot_cluster_means(
            data_path, hyb_round, score_thresh=score_thresh
        )
        if vis:
            plt.figure()
            plt.imshow(cluster_means)
            plt.title(hyb_round)
        save_path = processed_path / data_path / f"{hyb_round}_cluster_means.npy"
        np.save(save_path, cluster_means)


def hyb_spot_cluster_means(
    data_path,
    prefix,
    score_thresh=0,
    init_spot_colors=np.array([[0, 1, 0, 0], [0, 0, 0, 1]]),
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    rois = []
    for ref_tile in ops["barcode_ref_tiles"]:
        print(f"detecting spots in tile {ref_tile}")
        stack, _ = load_and_register_hyb_tile(
            data_path, tile_coors=ref_tile, prefix=prefix
        )
        spots = detect_spots(
            np.max(stack, axis=2), threshold=ops["hyb_spot_detection_threshold"]
        )
        stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
        spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
        rois.extend(extract_spots(spots, np.moveaxis(stack, 2, 3)))
    x = rois_to_array(rois, normalize=False)
    cluster_means, _, _, _, _, _ = scaled_k_means(
        x[0, :, :].T, init_spot_colors, score_thresh=score_thresh
    )
    return cluster_means, rois


def extract_hyb_spots_tile(data_path, tile_coors, prefix):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    cluster_means = np.load(processed_path / data_path / f"{prefix}_cluster_means.npy")
    print(f"detecting spots in tile {tile_coors}")
    stack, _ = load_and_register_hyb_tile(
        data_path, tile_coors=tile_coors, prefix=prefix, correct_illumination=True
    )
    spots = detect_spots(
        np.max(stack, axis=2), threshold=ops["hyb_spot_detection_threshold"]
    )
    stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
    spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
    spot_rois = extract_spots(spots, np.moveaxis(stack, 2, 3))
    x = rois_to_array(spot_rois, normalize=False)
    spots["trace"] = [roi.trace for roi in spot_rois]
    x_norm = x[0, :, :].T / np.linalg.norm(x[0, :, :].T, axis=1)[:, np.newaxis]
    score = x_norm @ cluster_means.T
    cluster_ind = np.argmax(score, axis=1)
    spots["cluster"] = cluster_ind
    spots["score"] = np.squeeze(score[np.arange(x_norm.shape[0]), cluster_ind])
    save_dir = processed_path / data_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    spots.to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )


def setup_barcode_calling(
    data_path, score_thresh=0.5, spot_size=2, correct_channels=False
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    rois = []
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
        rois.extend(extract_spots(spots, np.moveaxis(stack, 2, 3)))
    cluster_means = get_cluster_means(rois, vis=True, score_thresh=score_thresh)
    return cluster_means, rois


def basecall_tile(data_path, tile_coors):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
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
    spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
    spot_rois = extract_spots(spots, np.moveaxis(stack, 2, 3))
    x = rois_to_array(spot_rois, normalize=False)
    spots["trace"] = [roi.trace for roi in spot_rois]
    cluster_inds = []
    top_score = []

    nrounds = x.shape[0]
    for iround in range(nrounds):
        this_round_means = cluster_means[iround]
        this_round_means = this_round_means / np.linalg.norm(this_round_means, axis=1)
        x_norm = (
            x[iround, :, :].T / np.linalg.norm(x[iround, :, :].T, axis=1)[:, np.newaxis]
        )
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
    stack = np.moveaxis(stack, 2, 3)
    spots = detect_isolated_spots(
        np.std(stack, axis=(2, 3)),
        detection_threshold=detection_threshold,
        isolation_threshold=isolation_threshold,
    )

    rois = extract_spots(spots, stack)
    cluster_means = get_cluster_means(rois, vis=True)
    codebook = pd.read_csv(
        Path(__file__).parent.parent / "call" / codebook_name,
        header=None,
        names=["gii", "seq", "gene"],
    )
    gene_dict, unique_genes = make_gene_templates(cluster_means, codebook, vis=True)

    norm_shift = np.sqrt(
        np.median(
            np.sum(
                np.reshape(stack, (stack.shape[0], stack.shape[1], -1)) ** 2,
                axis=2,
            )
        )
    )
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
    pattern = f"{prefix}_MMStack_(\d*)-Pos(\d\d\d)_(\d\d\d).ome.tif"
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
        prefix (str, optional): the folder name prefix, before round number. Defaults to "round"

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


def estimate_channel_correction(data_path, prefix="round", nrounds=7):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
    stack = load_sequencing_rounds(
        data_path,
        ops["ref_tile"],
        suffix=ops["projection"],
        prefix=prefix,
        nrounds=nrounds,
    )
    nch, nrounds = stack.shape[2:]
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
        for iround in range(nrounds):
            for ich in range(nch):
                stack[stack < 0] = 0
                pixel_dist[:, ich, iround] += np.bincount(
                    stack[:, :, ich, iround].flatten().astype(np.uint16),
                    minlength=max_val + 1,
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

    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)
    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
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
    script_path = str(Path(__file__).parent.parent.parent / f"{script}.sh")
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()
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
    ops_path = processed_path / data_path / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()

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

    stack = np.moveaxis(stack, 2, 3)

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
        omp_score_threshold=ops["spot_threshold"],
    )

    for df, gene in zip(gene_spots, omp_stat["gene_names"]):
        df["gene"] = gene
    save_dir = processed_path / data_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_spots).to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
