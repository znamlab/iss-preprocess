import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation
from znamutils import slurm_it

import iss_preprocess as iss

from ..call import extract_spots
from ..coppafish import scaled_k_means
from ..image import apply_illumination_correction, compute_distribution, filter_stack
from ..io import (
    get_roi_dimensions,
    load_hyb_probes_metadata,
    load_metadata,
    load_ops,
    load_tile_by_coors,
)
from ..reg import apply_corrections
from ..segment import detect_spots
from .core import batch_process_tiles


def load_and_register_hyb_tile(
    data_path,
    tile_coors=(1, 0, 0),
    prefix="hybridisation_1_1",
    suffix="max",
    filter_r=(2, 4),
    correct_illumination=False,
    correct_channels=False,
    corrected_shifts="best",
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
        correct_shifts (str, optional): Which shift to use. One of `reference`,
            `single_tile`, `ransac`, or `best`. Defaults to 'best'.

    Returns:
        numpy.ndarray: X x Y x Nch image stack.
        numpy.ndarray: X x Y boolean mask, identifying bad pixels that we were not
            imaged for all channels (due to registration offsets) and should be
            discarded during analysis.

    """
    processed_path = iss.io.get_processed_path(data_path)
    valid_shifts = ["reference", "single_tile", "ransac", "best"]
    assert corrected_shifts in valid_shifts, (
        f"unknown shift correction method, must be one of {valid_shifts}",
    )
    tforms = get_channel_shifts(data_path, prefix, tile_coors, corrected_shifts)
    stack = load_tile_by_coors(
        data_path, tile_coors=tile_coors, suffix=suffix, prefix=prefix
    )
    if correct_illumination:
        stack = apply_illumination_correction(data_path, stack, prefix)
    if "matrix_between_channels" not in tforms.keys():
        stack = apply_corrections(
            stack,
            matrix=None,
            scales=tforms["scales"],
            angles=tforms["angles"],
            shifts=tforms["shifts"],
            cval=np.nan,
        )
    else:
        stack = apply_corrections(
            stack,
            matrix=tforms["matrix_between_channels"],
            cval=np.nan,
        )

    bad_pixels = np.any(np.isnan(stack), axis=(2))
    stack = np.nan_to_num(stack)

    if filter_r:
        stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
        mask = np.ones((filter_r[1] * 2 + 1, filter_r[1] * 2 + 1))
        bad_pixels = binary_dilation(bad_pixels, mask)
    if correct_channels:
        correction_path = processed_path / f"correction_{prefix}.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        stack = stack / norm_factors[np.newaxis, np.newaxis, :]
    return stack, bad_pixels


def get_channel_shifts(data_path, prefix, tile_coors, corrected_shifts):
    """Load the channel shifts for a given tile and sequencing acquisition.

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
    if corrected_shifts == "reference":
        tforms_fname = f"tforms_{prefix}.npz"
        tforms_path = processed_path
    elif corrected_shifts == "single_tile":
        tforms_fname = f"tforms_{prefix}_{tile_name}.npz"
        tforms_path = processed_path / "reg"
    elif corrected_shifts == "ransac":
        tforms_fname = f"tforms_corrected_{prefix}_{tile_name}.npz"
        tforms_path = processed_path / "reg"
    elif corrected_shifts == "best":
        tforms_fname = f"tforms_best_{prefix}_{tile_name}.npz"
        tforms_path = processed_path / "reg"
    else:
        raise ValueError(f"unknown shift correction method: {corrected_shifts}")
    tforms = np.load(tforms_path / tforms_fname, allow_pickle=True)
    return tforms


@slurm_it(conda_env="iss-preprocess")
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
            roi_name = f"roi {tile[0]}, tile {tile[1]}, {tile[2]}"
            print(f"counting pixel values for {hyb_round}, roi {roi_name}")
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


@slurm_it(conda_env="iss-preprocess")
def setup_hyb_spot_calling(data_path, prefix=None, vis=True):
    """Prepare and save bleedthrough matrices for hybridisation rounds.

    Args:
        data_path (str): Relative path to data
        prefix (list, optional): List of prefix of hybridisation rounds to process.
            If None, all hybridisation rounds are processed. Defaults to None.
        vis (bool, optional): Whether to generate diagnostic plots. Defaults to True.

    """
    processed_path = iss.io.get_processed_path(data_path)
    metadata = load_metadata(data_path)
    if prefix is None:
        prefix = list(metadata["hybridisation"].keys())
    elif isinstance(prefix, str):
        prefix = [prefix]

    print("setting up hybridisation spot calling. Will process the following rounds:")
    print(prefix)
    for hyb_round in prefix:
        print(f"processing {hyb_round}")
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
    iss.pipeline.check_hybridisation_setup(data_path, prefixes=prefix)


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
    ref_tiles = ops["genes_ref_tiles"]
    ref_tiles = ops.get("hybridisation_ref_tiles", ref_tiles)
    for ref_tile in ref_tiles:
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
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    metadata = load_metadata(data_path)
    for hyb_round in metadata["hybridisation"].keys():
        for roi in roi_dims[use_rois, :]:
            extract_hyb_spots_roi(
                data_path=data_path,
                prefix=hyb_round,
                roi=roi[0],
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
    if roi is not None:
        roi_dims = roi_dims[roi_dims[:, 0] == int(roi), :]
        assert len(roi_dims), f"no ROI {roi} found in roi_dims for {data_path}"
    batch_process_tiles(
        data_path,
        "extract_hyb_spots",
        roi_dims=roi_dims,
        additional_args=f",PREFIX={prefix}",
    )


@slurm_it(conda_env="iss-preprocess", slurm_options={"mem": "16G", "time": "1:00:00"})
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
    print("loading and registering tile")
    stack, _ = load_and_register_hyb_tile(
        data_path,
        tile_coors=tile_coors,
        prefix=prefix,
        suffix=ops["hybridisation_projection"],
        correct_illumination=True,
        correct_channels=ops["hybridisation_correct_channels"],
    )
    print(f"detecting spots in tile {tile_coors}")
    spots = detect_spots(
        np.max(stack, axis=2), threshold=ops["hybridisation_detection_threshold"]
    )
    if spots.shape[0]:
        print(f"Found {spots.shape[0]} spots. Extracting")
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
    print(f"saved spots for {prefix} tile {tile_coors} to {save_dir}")
