import numpy as np
import pandas as pd
import glob
import re
from flexiznam.config import PARAMETERS
from pathlib import Path
from os.path import isfile
from ..image import filter_stack
from ..reg import align_channels_and_rounds, generate_channel_round_transforms
from ..io import load_stack, write_stack
from ..segment import detect_isolated_spots
from ..call import extract_spots, make_gene_templates, run_omp, find_gene_spots


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
        stack,
        detection_threshold=detection_threshold,
        isolation_threshold=isolation_threshold,
    )

    rois = extract_spots(spots, stack)
    codebook = pd.read_csv(
        Path(__file__).parent.parent / "call" / codebook_name,
        header=None,
        names=["gii", "seq", "gene"],
    )
    gene_dict, unique_genes = make_gene_templates(rois, codebook, vis=True)

    norm_shift = np.sqrt(
        np.median(
            np.sum(
                np.reshape(stack, (stack.shape[0], stack.shape[1], -1)) ** 2,
                axis=2,
            )
        )
    )
    return gene_dict, unique_genes, norm_shift


def check_files(data_path, nrounds=7):
    """Check that TIFFs are present for all imaging rounds and return their list

    Args:
        data_path (str): relative path to the raw data
        nrounds (int, optional): number of sequencing rounds to look for

    Returns:
        bool: whether matching TIFFs for found for all rounds
        list: list of tiff paths for round 1

    """
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    data_path = raw_path / data_path

    tiffs = sorted(glob.glob(str(data_path / "round_01_1/*.tif")))
    success = True
    # check that all files exist
    for iround in range(nrounds):
        for tiff in tiffs:
            fname = tiff.replace("round_01", f"round_{str(iround+1).zfill(2)}")
            if not isfile(fname):
                print(f"{fname} does not exist")
                success = False
    return success, tiffs


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


def load_processed_tile(
    data_path, tile_coors=(1, 0, 0), nrounds=7, suffix="fstack", prefix="round"
):
    """Load processed tile images across rounds

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).
        nrounds (int, optional): Number of rounds to load. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to '_fstack'.

    Returns:
        numpy.ndarray: X x Y x channels x rounds stack.

    """
    tile_roi, tile_x, tile_y = tile_coors
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ims = []
    for iround in range(nrounds):
        # dirname = f"{prefix}_{str(iround+1).zfill(2)}_1"
        dirname = f"{prefix}_{iround+1}_1"
        fname = (
            f"{prefix}_{iround+1}_1_MMStack_{tile_roi}-"
            + f"Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_{suffix}.tif"
        )
        ims.append(load_stack(processed_path / data_path / dirname / fname))
    return np.stack(ims, axis=3)


def estimate_channel_correction(data_path, ops, prefix="round"):
    stack = load_processed_tile(
        data_path, ops["ref_tile"], suffix=ops["projection"], prefix=prefix
    )
    nch, nrounds = stack.shape[2:]
    max_val = 65535
    pixel_dist = np.zeros((max_val + 1, nch, nrounds))

    for ix in ops["correction_tiles_x"]:
        for iy in ops["correction_tiles_y"]:
            print(f"counting pixel values for tile {ix}, {iy}")
            stack = filter_stack(
                load_processed_tile(
                    data_path,
                    [ops["correction_roi"], ix, iy],
                    suffix=ops["projection"],
                    prefix=prefix,
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


def load_and_register_tile(
    data_path,
    tile_coors=(0, 0, 0),
    prefix="round",
    suffix="proj",
    filter_r=(2, 4),
    correct_channels=False,
):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    tforms_fname = (
        f"tforms_corrected_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.npz"
    )
    tforms_path = processed_path / data_path / "reg" / tforms_fname
    tforms = np.load(tforms_path, allow_pickle=True)

    stack = load_processed_tile(data_path, tile_coors, suffix=suffix, prefix=prefix)
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

    stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
    if correct_channels:
        correction_path = processed_path / data_path / "correction.npz"
        norm_factors = np.load(correction_path, allow_pickle=True)["norm_factors"]
        stack = stack / norm_factors[np.newaxis, np.newaxis, :, :]

    return stack, bad_pixels


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
        alpha=200.0,
        norm_shift=omp_stat["norm_shift"],
        max_comp=12,
        min_intensity=ops["omp_min_intensity"],
    )

    for igene in range(g.shape[2]):
        g[bad_pixels, igene] = 0

    spot_image_path = processed_path / data_path / "spot_sign_image.npy"
    if spot_image_path.exists():
        spot_sign_image = np.load(spot_image_path)
    else:
        print("No spot sign image for this dataset - using default.")
        spot_sign_image = np.load(
            Path(__file__).parent.parent / "call/spot_signimage.npy"
        )

    spot_sign_threshold = 0.15
    spot_sign_image[np.abs(spot_sign_image) < spot_sign_threshold] = 0

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
