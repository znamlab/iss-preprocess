from pathlib import Path

import numpy as np
import pandas as pd
from image_tools.similarity_transforms import make_transform
from znamutils import slurm_it

from ..io import get_processed_path, get_roi_dimensions, load_ops
from ..io.load import get_shifts_to_ref
from .stitch import get_tile_corners


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
    processed_path = get_processed_path(data_path)
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
    processed_path = get_processed_path(data_path)
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
    `align_spots`. The spots will then be merged together using `merge_roi_spots`. To
    avoid duplicate spots, we define a set of tile centers and keep only the spots that
    are closest to the center of the tile they were detected on


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
    processed_path = get_processed_path(data_path)

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
    use_rois = np.isin(roi_dims[:, 0], ops["use_rois"])
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


def align_cell_dataframe(data_path, prefix, ref_prefix=None, sindbis=False):
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
    ### TODO: adapt to find dataframes from sindbis soma barcode calling ie. already in reference frame
    mask_folder = get_processed_path(data_path) / "cells"
    if not sindbis:
        cells_df = mask_folder / f"{prefix}_df_corrected.pkl"
        assert cells_df.exists(), (
            f"Cells dataframe {cells_df} does not exist. "
            + "Run remove_all_duplicate_masks first"
        )
        cells_df = pd.read_pickle(cells_df)
    else: 
        # get roidims
        roi_dims = get_roi_dimensions(data_path)
        # loop over rois and load all tiles, concatenate and drop duplicates again (just in case)
        cells_df = []       
        for roi in roi_dims[:, 0]:
            for tx in range(roi_dims[roi_dims[:, 0] == roi, 1:][0, 0] + 1):
                for ty in range(roi_dims[roi_dims[:, 0] == roi, 1:][0, 1] + 1):
                    cell_file = mask_folder / f"{prefix}_somata_{roi}_{tx}_{ty}.pkl"
                    if cell_file.exists():
                        cells_df.append(pd.read_pickle(cell_file))
                    else:
                        print(f"Cell dataframe {cell_file} does not exist. Skipping.")
    cells_df = pd.concat(cells_df, ignore_index=True)

    if "x" not in cells_df.columns:
        cells_df.rename(columns={"centroid-1": "x", "centroid-0": "y"}, inplace=True)

    aligned_df = []
    for (roi, tilex, tiley), df in cells_df.groupby(["roi", "tilex", "tiley"]):
        aligned_df.append(
            _align_dataframe(df, data_path, (roi, tilex, tiley), prefix, ref_prefix)
        )
    aligned_df = pd.concat(aligned_df)

    return aligned_df

def drop_duplicated_masks_center_dist(df_roi, corners):
    """
    df_roi: rows for one roi with columns: ['tilex','tiley','x','y'] where x,y are GLOBAL coords
    corners: output of get_tile_corners(..., roi=roi) with shape [ntx, nty, 2, 4], coords in (y,x)
    """
    df_roi = df_roi.reset_index(drop=True)
    # Precompute bbox + center per tile from corners
    ys = corners[:, :, 0, :]  # (tx,ty,4)
    xs = corners[:, :, 1, :]  # (tx,ty,4)

    xmin = xs.min(axis=2); xmax = xs.max(axis=2)
    ymin = ys.min(axis=2); ymax = ys.max(axis=2)
    cx   = xs.mean(axis=2); cy   = ys.mean(axis=2)

    ntx, nty = xmin.shape

    # 9-neighborhood offsets
    offsets = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    keep = np.ones(len(df_roi), dtype=bool)

    # Work tile-by-tile so we only ever check 9 candidates per row
    for (tx, ty), idx in df_roi.groupby(["tilex","tiley"], sort=False).groups.items():
        tx = int(tx); ty = int(ty)
        x = df_roi.loc[idx, "x"].to_numpy(float)
        y = df_roi.loc[idx, "y"].to_numpy(float)

        best_d2 = np.full(len(idx), np.inf)
        best_tx = np.full(len(idx), tx)
        best_ty = np.full(len(idx), ty)
        found   = np.zeros(len(idx), dtype=bool)

        for dtx, dty in offsets:
            tx2, ty2 = tx + dtx, ty + dty
            if tx2 < 0 or ty2 < 0 or tx2 >= ntx or ty2 >= nty:
                continue

            inside = (x >= xmin[tx2,ty2]) & (x < xmax[tx2,ty2]) & (y >= ymin[tx2,ty2]) & (y < ymax[tx2,ty2])
            if not inside.any():
                continue

            dx = x - cx[tx2,ty2]
            dy = y - cy[tx2,ty2]
            d2 = dx*dx + dy*dy

            better = inside & (d2 < best_d2)
            if better.any():
                best_d2[better] = d2[better]
                best_tx[better] = tx2
                best_ty[better] = ty2
                found[better] = True

        # if no candidate tile contained the point, keep it (don’t delete blindly)
        keep_idx = (~found) | ((best_tx == tx) & (best_ty == ty))
        keep[np.array(list(idx))] = keep_idx

    return df_roi.loc[keep].copy()

@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "1:00:00", "mem": "8G"})
def stitch_cell_dataframes(data_path, prefix, ref_prefix=None, sindbis=False):
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

    stitched_df = align_cell_dataframe(data_path, prefix, ref_prefix=None, sindbis=sindbis).copy() # if sindbis, we assume the cell dataframe is already in reference frame, so we skip alignment
    stitched_df["x_in_tile"] = stitched_df["x"].copy()
    stitched_df["y_in_tile"] = stitched_df["y"].copy()
    stitched_df["tile"] = "Not Processed"
    stitched_df["x"] = np.nan
    stitched_df["y"] = np.nan

    kept = []

    for roi, df_roi in stitched_df.groupby("roi", sort=False):
        ref_corners = get_tile_corners(data_path, prefix=ref_prefix, roi=roi)  # [ntx,nty,2,4]
        ref_origins = ref_corners[..., 0]  # [ntx,nty,2] (y,x) of corner [0,0]

        df_roi = df_roi.copy()

        # compute global x/y for this ROI
        tx = df_roi["tilex"].to_numpy(dtype=int)
        ty = df_roi["tiley"].to_numpy(dtype=int)

        # vectorized lookup of origins
        x0 = ref_origins[tx, ty, 1]
        y0 = ref_origins[tx, ty, 0]

        df_roi["tile"] = [f"{roi}_{a}_{b}" for a, b in zip(tx, ty)]
        df_roi["x"] = df_roi["x_in_tile"].to_numpy(float) + x0
        df_roi["y"] = df_roi["y_in_tile"].to_numpy(float) + y0

        # drop duplicates inside this ROI
        df_roi = drop_duplicated_masks_center_dist(df_roi, ref_corners)

        kept.append(df_roi)

    stitched_df = pd.concat(kept, ignore_index=True)

    mask_folder = get_processed_path(data_path) / "cells" / f"{prefix}_cells"
    target = mask_folder / f"{prefix}_df_corrected.pkl"
    mask_folder.mkdir(exist_ok=True)
    stitched_df.to_pickle(target)
    print(f"Saved stitched cell dataframe to {target}")
    return stitched_df