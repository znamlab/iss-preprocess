from pathlib import Path

import numpy as np
import pandas as pd
from image_tools.similarity_transforms import make_transform
from znamutils import slurm_it

import iss_preprocess as iss
from iss_preprocess.io import get_roi_dimensions, load_ops
from iss_preprocess.pipeline.reg2ref import get_shifts_to_ref
from iss_preprocess.pipeline.stitch import get_tile_corners


def _align_dataframe(df, data_path, tile_coors, prefix, ref_prefix=None):
    """Align a dataframe of spots to reference coordinates

    Split in internal function to reuse for cells and spots

    Args:
        df (pd.DataFrame): The dataframe with x and y to align
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinates
        prefix (str): Prefix of spots to load
        ref_prefix (str, optional): Prefix of the reference spots. If None, reads from
            ops. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with x and y registered to reference tile.

    """
    processed_path = iss.io.get_processed_path(data_path)
    if ref_prefix is None:
        ops = load_ops(data_path)
        ref_prefix = ops["reference_prefix"]

    if ref_prefix.startswith(prefix):
        # it is the ref, no need to register
        return df

    tform = get_shifts_to_ref(data_path, prefix, *tile_coors)
    if ops["align_method"] == "similarity":
        tile_shape = np.load(processed_path / "reg" / f"{ref_prefix}_shifts.npz")[
            "tile_shape"
        ]
        df_tform = make_transform(
            tform["scales"][0][0], tform["angles"][0][0], tform["shifts"][0], tile_shape
        )
    else:
        df_tform = tform["matrix_between_channels"][0]

    transformed_coors = df_tform @ np.stack([df["x"], df["y"], np.ones(len(df))])
    df["x_raw"] = df["x"].copy()
    df["y_raw"] = df["y"].copy()
    df["x"] = [x for x in transformed_coors[0, :]]
    df["y"] = [y for y in transformed_coors[1, :]]
    return df


def align_spots(data_path, tile_coors, prefix, ref_prefix=None):
    """Use previously computed transformation matrices to align spots to reference
    coordinates.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (roi, tilex, tiley) tuple of tile coordinates
        prefix (str): Prefix of spots to load
        ref_prefix (str, optional): Prefix of the reference spots. If None, reads from
            ops. Defaults to None.

    Returns:
        pd.DataFrame: The spot dataframe with x and y registered to reference tile.

    """
    roi, tilex, tiley = tile_coors
    processed_path = iss.io.get_processed_path(data_path)
    spots = pd.read_pickle(
        processed_path / "spots" / f"{prefix}_spots_{roi}_{tilex}_{tiley}.pkl"
    )
    spots = _align_dataframe(spots, data_path, tile_coors, prefix, ref_prefix)
    return spots


def merge_roi_spots(
    data_path, prefix, tile_origins, tile_centers, iroi=1, keep_all_spots=False
):
    """Load and combine spot locations across all tiles for an ROI.

    To avoid duplicate spots from tile overlap, we determine which tile center
    each spot is closest to. We then only keep the spots that are closest to
    the center of the tile they were detected on. The tile_centers do not need to be the
    center of the reference tile. For acquisition with a significant shift, it might be
    better to use the center of the acquisition tile registered to the reference.
    See merge_and_align_spots for an example.

    Args:
        data_path (str): path to pickle files containing spot locations for each tile.
        prefix (str): prefix of the spots to load and register (e.g. barcode_round)
        tile_origins (numpy.array): origin of each tile
        tile_centers (numpy array): center of each tile for ROI duplication detection.
        iroi (int, optional): ID of ROI to load. Defaults to 1.
        keep_all_spots (bool, optional): If True, keep all spots. Otherwise, keep only
            spots which are closer to the tile_centers. Defaults to False.



    Returns:
        pandas.DataFrame: table containing spot locations across all tiles.

    """
    roi_dims = get_roi_dimensions(data_path)
    all_spots = []
    ntiles = roi_dims[roi_dims[:, 0] == iroi, 1:][0] + 1

    for tx in range(ntiles[0]):
        for ty in range(ntiles[1]):
            try:
                spots = align_spots(data_path, tile_coors=(iroi, tx, ty), prefix=prefix)
                spots["x_in_tile"] = spots["x"].copy()
                spots["y_in_tile"] = spots["y"].copy()
                spots["tile"] = f"{iroi}_{tx}_{ty}"
                spots["x"] = spots["x"] + tile_origins[tx, ty, 1]
                spots["y"] = spots["y"] + tile_origins[tx, ty, 0]

                if not keep_all_spots:
                    # calculate distance to tile centers
                    spot_dist = (
                        spots["x"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 1]
                    ) ** 2 + (
                        spots["y"].to_numpy()[:, np.newaxis, np.newaxis]
                        - tile_centers[np.newaxis, :, :, 0]
                    ) ** 2
                    home_tile_dist = (spot_dist[:, tx, ty]).copy()
                    spot_dist[:, tx, ty] = np.inf
                    min_spot_dist = np.min(spot_dist, axis=(1, 2))
                    keep_spots = home_tile_dist < min_spot_dist
                else:
                    keep_spots = np.ones(spots.shape[0], dtype=bool)
                all_spots.append(spots[keep_spots])
            except FileNotFoundError:
                print(f"could not load roi {iroi}, tile {tx}, {ty}")

    spots = pd.concat(all_spots, ignore_index=True)
    return spots


@slurm_it(conda_env="iss-preprocess")
def merge_and_align_spots(
    data_path,
    roi,
    spots_prefix="barcode_round",
    ref_prefix=None,
    keep_all_spots=False,
):
    """Combine spots across tiles and align to reference coordinates for a single ROI.

    For each tile, spots will be registered to the reference coordinates using
    `iss.pipeline.register.align_spots`. The spots will then be merged together using
    `merge_roi_spots`. To avoid duplicate spots, we define a set of tile centers and
    keep only the spots that are closest to the center of the tile they were detected on


    Args:
        data_path (str): Relative path to data.
        roi (int): ROI ID to process (as specified in MicroManager).
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        ref_prefix (str, optional): Acquisition prefix of the reference acquisition
            to transform spot coordinates to. Defaults to "genes_round_1_1".
        keep_all_spots (bool, optional): If True, keep all spots. Otherwise, keep only
            spots which are closer to the tile_centers. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing all spots in reference coordinates.

    """
    print(f"Aligning spots for ROI {roi}")
    ops = load_ops(data_path)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]
    processed_path = iss.io.get_processed_path(data_path)

    # find tile origin, final shape, and shifts in reference coordinates
    ref_corners = get_tile_corners(data_path, prefix=ref_prefix, roi=roi)
    ref_centers = np.mean(ref_corners, axis=3)
    ref_origins = ref_corners[..., 0]

    # always use the center of the reference tile for spot merging
    # we might have to change that
    trans_centers = ref_centers
    spots = merge_roi_spots(
        data_path,
        prefix=spots_prefix,
        tile_centers=trans_centers,
        tile_origins=ref_origins,
        iroi=roi,
        keep_all_spots=keep_all_spots,
    )
    fname = processed_path / f"{spots_prefix}_spots_{roi}.pkl"
    spots.to_pickle(fname)
    print(f"Saved spots for ROI in {fname}")
    return spots


def merge_and_align_spots_all_rois(
    data_path,
    spots_prefix="barcode_round",
    ref_prefix="genes_round_1_1",
    keep_all_spots=False,
    dependency=None,
):
    """Start batch jobs to combine spots across tiles and align to reference coordinates
    for all ROIs.

     Args:
        data_path (str): Relative path to data.
        spots_prefix (str, optional): Filename prefix of the spot files to combine.
            Defaults to "barcode_round".
        ref_prefix (str, optional): Acquisition prefix to use as a reference for
            registration. Defaults to "genes_round_1_1".

    """
    ops = load_ops(data_path)
    roi_dims = get_roi_dimensions(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, 0]:
        slurm_folder = Path.home() / "slurm_logs" / data_path / "align_spots"
        slurm_folder.mkdir(exist_ok=True, parents=True)
        merge_and_align_spots(
            data_path,
            roi,
            spots_prefix=spots_prefix,
            ref_prefix=ref_prefix,
            keep_all_spots=keep_all_spots,
            use_slurm=True,
            slurm_folder=slurm_folder,
            scripts_name=f"iss_align_spots_{spots_prefix}_{roi}",
            job_dependency=dependency,
        )


def align_cell_dataframe(data_path, prefix, ref_prefix=None):
    """Align a cell dataframe to reference coordinates

    Designed for mCherry cells. Reads the f"{prefix}_df_corrected.pkl" file generated
    by remove_all_duplicate_masks and aligns the x and y coordinates to the reference
    tile by tile.

    Args:
        data_path (str): Relative path to data
        prefix (str): Prefix of cells to load
        ref_prefix (str, optional): Prefix of the reference cells. If None, reads from
            ops. Defaults to None.

    Returns:
        pd.DataFrame: The cell dataframe with x and y registered to reference tile.
    """
    mask_folder = iss.io.get_processed_path(data_path) / "cells" / f"{prefix}_cells"
    cells_df = mask_folder / f"{prefix}_df_corrected.pkl"
    assert cells_df.exists(), (
        f"Cells dataframe {cells_df} does not exist. "
        + "Run remove_all_duplicate_masks first"
    )
    cells_df = pd.read_pickle(cells_df)
    if "x" not in cells_df.columns:
        cells_df.rename(columns={"centroid-1": "x", "centroid-0": "y"}, inplace=True)

    aligned_df = []
    for (roi, tilex, tiley), df in cells_df.groupby(["roi", "tilex", "tiley"]):
        aligned_df.append(
            _align_dataframe(df, data_path, (roi, tilex, tiley), prefix, ref_prefix)
        )
    aligned_df = pd.concat(aligned_df)

    return aligned_df


@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "1:00:00", "mem": "8G"})
def stitch_cell_dataframes(data_path, prefix, ref_prefix=None):
    """Stitch cell dataframes across all tiles and ROI.

    Args:
        data_path (str): path to data
        prefix (str): prefix of the cell dataframe to load
        ref_prefix (str, optional): prefix of the reference tiles to use for stitching.
            Defaults to None.

    Returns:
        pandas.DataFrame: stitched cell dataframe
    """

    ops = load_ops(data_path)
    if ref_prefix is None:
        ref_prefix = ops["reference_prefix"]

    stitched_df = align_cell_dataframe(data_path, prefix, ref_prefix=None).copy()
    stitched_df["x_in_tile"] = stitched_df["x"].copy()
    stitched_df["y_in_tile"] = stitched_df["y"].copy()
    stitched_df["tile"] = "Not Processed"
    stitched_df["x"] = np.nan
    stitched_df["y"] = np.nan

    for roi, df in stitched_df.groupby("roi"):
        # find tile origin, final shape, and shifts in reference coordinates
        ref_corners = get_tile_corners(data_path, prefix=ref_prefix, roi=roi)
        ref_origins = ref_corners[..., 0]
        for (tx, ty), tdf in df.groupby(["tilex", "tiley"]):
            stitched_df.loc[tdf.index, "tile"] = f"{roi}_{tx}_{ty}"
            stitched_df.loc[tdf.index, "x"] = tdf["x_in_tile"] + ref_origins[tx, ty, 1]
            stitched_df.loc[tdf.index, "y"] = tdf["y_in_tile"] + ref_origins[tx, ty, 0]

    # save stitched dataframe
    mask_folder = iss.io.get_processed_path(data_path) / "cells" / f"{prefix}_cells"
    cells_df = mask_folder / f"{prefix}_df_corrected.pkl"
    stitched_df.to_pickle(cells_df)

    return stitched_df
