import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.morphology import binary_dilation
from znamutils import slurm_it

from ..call import extract_spots
from ..coppafish import scaled_k_means
from ..diagnostics.diag_hybridisation import check_hybridisation_setup
from ..image import compute_distribution, filter_stack
from ..io import (
    get_channel_round_transforms,
    get_processed_path,
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
    processed_path = get_processed_path(data_path)
    valid_shifts = ["reference", "single_tile", "ransac", "best"]
    assert corrected_shifts in valid_shifts, (
        f"unknown shift correction method, must be one of {valid_shifts}",
    )
    tforms = get_channel_round_transforms(
        data_path, prefix, tile_coors, corrected_shifts
    )
    stack = load_tile_by_coors(
        data_path,
        tile_coors=tile_coors,
        suffix=suffix,
        prefix=prefix,
        correct_illumination=correct_illumination,
    )
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


@slurm_it(conda_env="iss-preprocess")
def estimate_channel_correction_hybridisation(data_path, prefix=None):
    """Compute grayscale value distribution and normalisation factors for
    all hybridisation rounds.

    Each `correction_tiles` of `ops` is filtered before being used to compute the
    distribution of pixel values.
    Normalisation factor to equalise these distribution across channels and rounds are
    defined as `ops["correction_quantile"]` of the distribution.

    Args:
        data_path (str or Path): Relative path to the data folder
        prefix (list, optional): List of prefix of hybridisation rounds to process. If
            None, all hybridisation rounds are processed. Defaults to None.

    Returns:
        pixel_dist (np.array): A 65536 x Nch x Nrounds distribution of grayscale values
            for filtered stacks
        norm_factors (np.array) A Nch x Nround array of normalisation factors

    """
    if prefix is None:
        metadata = load_metadata(data_path)
        if "hybridisation" not in metadata.keys():
            return
        for prefix in metadata["hybridisation"].keys():
            estimate_channel_correction_hybridisation(data_path, prefix)
            return

    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    nch = len(ops["black_level"])
    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch))
    corr_tiles = ops.get("correction_tiles", None)
    if corr_tiles is None:
        # If it exists, use the genes_ref_tiles, otherwise use the ref_tile
        corr_tiles = ops.get("genes_ref_tiles", None)
        if corr_tiles is not None:
            txt = "genes_ref_tiles"
        else:
            txt = "ref_tile"
            corr_tiles = ops[txt]
        print(f"No correction tiles specified, using {txt}")

    for tile in corr_tiles:
        roi_name = f"roi {tile[0]}, tile {tile[1]}, {tile[2]}"
        print(f"counting pixel values for {prefix}, roi {roi_name}")
        stack = filter_stack(
            load_tile_by_coors(
                data_path,
                tile_coors=tile,
                suffix=ops["hybridisation_projection"],
                prefix=prefix,
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

    save_path = processed_path / f"correction_{prefix}.npz"
    np.savez(save_path, pixel_dist=pixel_dist, norm_factors=norm_factors)
    print(f"saved correction factors for {prefix} to {save_path}")


@slurm_it(conda_env="iss-preprocess")
def setup_hyb_spot_calling(data_path, prefix=None, vis=True):
    """Prepare and save bleedthrough matrices for hybridisation rounds.

    Args:
        data_path (str): Relative path to data
        prefix (list, optional): List of prefix of hybridisation rounds to process.
            If None, all hybridisation rounds are processed. Defaults to None.
        vis (bool, optional): Whether to generate diagnostic plots. Defaults to True.

    """
    processed_path = get_processed_path(data_path)
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
    check_hybridisation_setup(data_path, prefixes=prefix)


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
    use_rois = np.isin(roi_dims[:, 0], ops["use_rois"])
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
def extract_hyb_spots_tile(data_path, tile_coors, prefix, detect_only=None, return_spots=False, channels=None):
    """Detect hybridisation spots for a given tile.

    Args:
        data_path (str): Relative path to data.
        tile_coors (tuple): Coordinates of tile to load: ROI, Xpos, Ypos.
        prefix (str): Prefix of the hybridisation round, e.g. "hybridisation_1_1".
        detect_only (bool | None): If True, only detect spot coordinates (skip extraction /
            clustering). If None, value is read from `ops['hybridisation_detect_only']`
            (defaults to False if missing).
        return_spots (bool | False): If True and detect_only=True, return the spots
            DataFrame instead of (only) saving it. If None, read from
            `ops['hybridisation_return_spots']` (defaults False).
        channels (Iterable[int] | str | None): Subset of channels to use. If None, will
            use all channels unless `ops['hybridisation_channels']` is defined. A string
            like "0,2,3" or "0 2 3" is accepted. If a subset is specified while
            classification is requested, classification is automatically disabled to
            avoid mismatches with cluster means.

    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)

    # Pull defaults from ops if not explicitly provided, with per-round overrides
    # Per-round keys use the pattern: <prefix>_hyb_detect_only, <prefix>_hyb_return_spots, <prefix>_hyb_channels
    if detect_only is None:
        detect_only = ops.get(f"{prefix}_hyb_detect_only", ops.get("hybridisation_detect_only", False))
    if channels is None:
        channels = ops.get(f"{prefix}_hyb_channels", ops.get("hybridisation_channels", None))

    # Normalise channels specification from ops if provided as string
    if isinstance(channels, str):
        if channels.strip().lower() in ("all", "*"):
            channels = None
        else:
            # allow both comma and space separated lists
            parts = [p for seg in channels.replace(",", " ").split() for p in [seg] if p != ""]
            try:
                channels = tuple(int(p) for p in parts)
            except ValueError:
                raise ValueError(
                    f"Could not parse hybridisation_channels string '{channels}'. Use comma/space separated integers."
                )

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
    if channels is None:
        channels = tuple(range(stack.shape[2]))  # all channels
    else:
        try:
            channels = tuple(channels)
        except TypeError:
            channels = (int(channels),)

    # Basic validation
    if len(channels) == 0:
        raise ValueError("channels cannot be empty")
    if max(channels) >= stack.shape[2] or min(channels) < 0:
        raise ValueError(
            f"Channel indices {channels} out of bounds for stack with {stack.shape[2]} channels"
        )
    if len(set(channels)) != len(channels):
        print("WARNING: duplicate channel indices detected; using unique ordering")
        channels = tuple(dict.fromkeys(channels))

    subset = len(channels) != stack.shape[2]
    if subset:
        if not detect_only:
            print(
                "INFO: Channel subset specified; forcing detect_only=True to avoid mismatch with cluster means."
            )
            detect_only = True
        stack = stack[:, :, list(channels)]
        if detect_only:
            print(
                f"Detecting using channel subset {channels} (original nch={ops['black_level'].__len__()})."
            )
    
    spots = detect_spots(
        np.max(stack, axis=2), threshold=ops["hybridisation_detection_threshold"]
    )
    if detect_only:
        print(f"Found {spots.shape[0]} spots. Stopping here as requested")
        if return_spots:
            return spots
        else:
            # save spots data frame
            save_dir = processed_path / "spots"
            save_dir.mkdir(parents=True, exist_ok=True)
            channels_str = "_".join(map(str, list(channels)))
            spots.to_pickle(
                save_dir
                / f"{prefix}_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}_chs_{channels_str}.pkl"
            )
            print(f"saved spots for {prefix} tile {tile_coors} to {save_dir}")
            return
    if spots.shape[0]:
        print(f"Found {spots.shape[0]} spots. Extracting")
        stack = stack[:, :, np.argsort(ops["camera_order"]), np.newaxis]
        spots["size"] = np.ones(len(spots)) * ops["spot_extraction_radius"]
        extract_spots(spots, stack, ops["spot_extraction_radius"])
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