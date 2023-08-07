import subprocess, shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iss_preprocess as iss
from pathlib import Path
from skimage.morphology import binary_dilation
from ..image import filter_stack, apply_illumination_correction, compute_distribution
from ..reg import apply_corrections
from ..io import (
    load_tile_by_coors,
    load_metadata,
    load_hyb_probes_metadata,
    load_ops,
    get_roi_dimensions,
)
from ..segment import detect_spots
from ..call import extract_spots
from ..coppafish import scaled_k_means


def load_and_register_hyb_tile(
    data_path,
    tile_coors=(1, 0, 0),
    prefix="hybridisation_1_1",
    suffix="fstack",
    filter_r=(2, 4),
    correct_illumination=False,
    correct_channels=False,
):
    """Load hybridisation tile and align channels. Optionally, filter, correct
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
        correct_illumination (bool, optional): Whether to correct vignetting.
            Defaults to False.
        correct_channels (bool, optional): Whether to normalize channel brightness.
            Defaults to False.

    Returns:
        numpy.ndarray: X x Y x Nch image stack.
        numpy.ndarray: X x Y boolean mask, identifying bad pixels that we were not imaged
            for all channels (due to registration offsets) and should be discarded
            during analysis.

    """
    processed_path = iss.io.get_processed_path(data_path)
    tforms_fname = (
        f"tforms_corrected_{prefix}_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
    )
    tforms = np.load(processed_path / "reg" / tforms_fname, allow_pickle=True)
    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
    )
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)
    stack = apply_corrections(
        stack, tforms["scales"], tforms["angles"], tforms["shifts"], cval=np.nan
    )
    bad_pixels = np.any(np.isnan(stack), axis=(2))
    stack[np.isnan(stack)] = 0
    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
        mask = np.ones((filter_r[1] * 2 + 1, filter_r[1] * 2 + 1))
        bad_pixels = binary_dilation(bad_pixels, mask)
    if correct_channels:
        correction_path = processed_path / f"correction_{prefix}.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        stack = stack / norm_factors[np.newaxis, np.newaxis, :]
    return stack, bad_pixels


def estimate_channel_correction_hybridisation(data_path):
    """Compute grayscale value distribution and normalisation factors for
    all hybridisation rounds.

    Each `correction_tiles` of `ops` is filtered before being used to compute the
    distribution of pixel values.
    Normalisation factor to equalise these distribution across channels and rounds are
    defined as `ops["correction_quantile"]` of the distribution.

    Args:
        data_path (str or Path): Relative path to the data folder

    Returns:
        pixel_dist (np.array): A 65536 x Nch x Nrounds distribution of grayscale values
            for filtered stacks
        norm_factors (np.array) A Nch x Nround array of normalisation factors

    """
    metadata = load_metadata(data_path)
    if "hybridisation" not in metadata.keys():
        return
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    nch = len(ops["black_level"])
    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch))
    for hyb_round in metadata["hybridisation"].keys():
        for tile in ops["correction_tiles"]:
            print(
                f"counting pixel values for {hyb_round}, roi {tile[0]}, tile {tile[1]}, {tile[2]}"
            )
            stack = filter_stack(
                load_tile_by_coors(
                    data_path,
                    tile_coors=tile,
                    suffix=ops["hybridisation_projection"],
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

        save_path = processed_path / f"correction_{hyb_round}.npz"
        np.savez(save_path, pixel_dist=pixel_dist, norm_factors=norm_factors)


def setup_hyb_spot_calling(data_path, vis=True):
    """Prepare and save bleedthrough matrices for hybridisation rounds.

    Args:
        data_path (str): Relative path to data
        vis (bool, optional): Whether to generate diagnostic plots. Defaults to True.

    """
    processed_path = iss.io.get_processed_path(data_path)
    metadata = load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        cluster_means, spot_colors, cluster_inds, genes = hyb_spot_cluster_means(
            data_path, hyb_round
        )
        if vis:
            plt.figure()
            plt.imshow(cluster_means)
            plt.title(hyb_round)
            plt.yticks(ticks=range(cluster_means.shape[0]), labels=genes)
        save_path = processed_path / f"{hyb_round}_cluster_means.npz"
        np.savez(
            save_path,
            cluster_means=cluster_means,
            genes=genes,
            spot_colors=spot_colors,
            cluster_inds=cluster_inds,
        )
    iss.pipeline.check_hybridisation_setup(data_path)


def hyb_spot_cluster_means(data_path, prefix):
    """Estimate bleedthrough matrices for hybridisation spots. Spot
    colors for each dye are initialized based on the metadata in the
    hybridisation probe list.

    Uses tiles specified in `ops["barcode_ref_tiles"]`.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Prefix of hybridisation round, e.g. "hybridisation_1_1".

    Returns:
        numpy.ndarray: Nprobes x Nch bleedthrough matrix.
        pandas.DataFrame: DataFrame of all detected spots across all tiles.
        list: list of gene names based on probe metadata.

    """
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
            suffix=ops["hybridisation_projection"],
            correct_channels=ops["hybridisation_correct_channels"],
        )
        spots = detect_spots(
            np.max(stack, axis=2), threshold=ops["hybridisation_detection_threshold"]
        )
        stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
        spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
        extract_spots(spots, stack, ops["spot_extraction_radius"])
        all_spots.append(spots)

    all_spots = pd.concat(all_spots, ignore_index=True)
    spot_colors = np.stack(all_spots["trace"], axis=2)

    cluster_means, _, cluster_inds, _, _, _ = scaled_k_means(
        spot_colors[0, :, :].T,
        init_spot_colors,
        score_thresh=ops["hybridisation_cluster_score_thresh"],
    )

    return cluster_means, spot_colors, cluster_inds, genes


def extract_hyb_spots_all(data_path):
    """Start `sbatch` jobs to detect hybridisation spots for each hybridisation
    round and ROI.

    Args:
        data_path (str): Relative path to data.

    """
    roi_dims = get_roi_dimensions(data_path)

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
    """Detect hybridisation spots for a given hybridisation round and ROI.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Prefix of the hybridisation round, e.g. "hybridisation_1_1".
        roi (int): ID of the ROI to process, as specified in MicroManager
            (i.e. 1-based)

    """
    roi_dims = get_roi_dimensions(data_path)
    ntiles = roi_dims[roi_dims[:, 0] == roi, 1:][0] + 1
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            extract_hyb_spots_tile(data_path, (roi, ix, iy), prefix)


def extract_hyb_spots_tile(data_path, tile_coors, prefix):
    """Detect hybridisation spots for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of tile to load: ROI, Xpos, Ypos.
        prefix (str): Prefix of the hybridisation round, e.g. "hybridisation_1_1".

    """
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    clusters = np.load(
        processed_path / f"{prefix}_cluster_means.npz", allow_pickle=True
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
        np.max(stack, axis=2), threshold=ops["hybridisation_detection_threshold"]
    )
    if spots.shape[0]:
        stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
        spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
        iss.pipeline.extract_spots(spots, stack, ops["spot_extraction_radius"])
        x = np.stack(spots["trace"], axis=2)
        x_norm = x[0, :, :].T / np.linalg.norm(x[0, :, :].T, axis=1)[:, np.newaxis]
        score = x_norm @ clusters["cluster_means"].T
        cluster_ind = np.argmax(score, axis=1)
        spots["cluster"] = cluster_ind
        spots["gene"] = clusters["genes"][cluster_ind]
        spots["score"] = np.squeeze(score[np.arange(x_norm.shape[0]), cluster_ind])
        spots["mean_intensity"] = [np.max(trace) for trace in spots["trace"]]
    else:
        print(f"No spots detected in tile {tile_coors} for round {prefix}")
        spots = pd.DataFrame(
            columns=[
                "y",
                "x",
                "size",
                "trace",
                "cluster",
                "gene",
                "score",
                "mean_intensity",
            ]
        )
    save_dir = processed_path / "spots"
    save_dir.mkdir(parents=True, exist_ok=True)
    spots.to_pickle(
        save_dir / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl"
    )
