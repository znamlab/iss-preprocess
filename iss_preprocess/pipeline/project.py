import multiprocessing as mp
import os
import shutil
from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
from znamutils import slurm_it

from ..decorators import updates_flexilims
from ..image import fstack_channels
from ..io import (
    get_processed_path,
    get_raw_filename,
    get_raw_path,
    get_roi_dimensions,
    get_tile_ome,
    load_metadata,
    load_ops,
    write_stack,
)
from .core import batch_process_tiles


@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "00:30:00", "mem": "8G"})
def check_projection(data_path, prefix, suffixes=("max", "median")):
    """Check if all tiles have been projected successfully.

    Args:
            data_path (str): Relative path to data.
            prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
            suffixes (tuple, optional): Projection suffixes to check for.
            Defaults to ("max", "median").

    """
    processed_path = get_processed_path(data_path)
    if prefix is None:
        metadata = load_metadata(data_path)
        prefixes = [f"genes_round_{i+1}_1" for i in range(metadata["genes_rounds"])]
        prefixes += [
            f"barcode_round_{i+1}_1" for i in range(metadata["barcode_rounds"])
        ]
        prefixes.extend(metadata["hybridisation"].keys())
        prefixes.extend(metadata["fluorescence"].keys())
        for prefix in prefixes:
            check_projection(data_path, prefix, suffixes)
        return
    roi_dims = get_roi_dimensions(data_path, prefix)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.isin(roi_dims[:, 0], ops["use_rois"])
    not_projected = []
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                tile_name = f"Pos{str(ix).zfill(3)}_{str(iy).zfill(3)}"
                fname = f"{prefix}_MMStack_{roi[0]}-{tile_name}"
                for suffix in suffixes:
                    proj_path = processed_path / prefix / f"{fname}_{suffix}.tif"
                    if not proj_path.exists():
                        print(f"{proj_path} missing!", flush=True)
                        not_projected.append(fname)

    np.savetxt(
        processed_path / prefix / "missing_tiles.txt",
        not_projected,
        fmt="%s",
        delimiter="\n",
    )

    if not not_projected:
        print(f"all tiles projected for {data_path} {prefix}!", flush=True)


@slurm_it(conda_env="iss-preprocess")
def check_roi_dims(data_path):
    """
    Check if all ROI dimensions are the same across rounds.
    Args:
        data_path (str): Relative path to data.
    Raises:
        ValueError: If ROI dimensions are not the same across rounds.
    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    rounds_info = []
    for root, dirs, _ in os.walk(processed_path):
        dirs.sort()
        for d in dirs:
            if d.endswith("_1"):
                if d in ops["overview_round"]:
                    continue
                roi_dims = get_roi_dimensions(data_path, d)
                rounds_info.append((d, roi_dims))

    all_same = all(
        np.array_equal(rounds_info[0][1], roi_dim) for _, roi_dim in rounds_info
    )
    if not all_same:
        differences = ""
        for i, (round_name, roi_dims) in enumerate(rounds_info):
            if not np.array_equal(rounds_info[0][1], roi_dims):
                differences += (
                    f"{round_name} \n{roi_dims} "
                    + "\n{rounds_info[0][0]} \n{rounds_info[0][1]}\n"
                )
        raise ValueError(f"Differences in roi_dims found across rounds:\n{differences}")
    else:
        print("All ROI dimensions are the same across rounds.", flush=True)


@slurm_it(conda_env="iss-preprocess")
def reproject_failed(
    data_path,
):
    """Re-project tiles that failed to project previously.

    Args:
        data_path (str): Relative path to data.

    """
    processed_path = get_processed_path(data_path)
    missing_tiles = []
    for d in processed_path.iterdir():
        if not d.is_dir() or not d.name.endswith("_1"):
            continue
        prefix = d.name
        for fname in (
            (processed_path / prefix / "missing_tiles.txt").read_text().split("\n")
        ):
            missing_tiles.append(fname)
    missing_tiles = list(set(missing_tiles))
    print(f"Reprojecting {len(missing_tiles)} tiles...", flush=True)
    print(missing_tiles, flush=True)
    for tile in missing_tiles:
        if len(tile) == 0:
            continue
        prefix = tile.split("_MMStack")[0]
        roi = int(tile.split("_MMStack_")[1].split("-")[0])
        ix = int(tile.split("_MMStack")[1].split("-Pos")[1].split("_")[0])
        iy = int(tile.split("_MMStack")[1].split("-Pos")[1].split("_")[1])
        print(f"Reprojecting {prefix} {roi}_{ix}_{iy}", flush=True)
        project_tile_by_coors((roi, ix, iy), data_path, prefix, overwrite=True)
    if len(missing_tiles) == 0:
        print("No failed tiles to re-project!", flush=True)


@updates_flexilims(name_source="prefix")  # type: ignore
def project_round(data_path, prefix, overwrite=False):
    """Start SLURM jobs to z-project all tiles from a single imaging round.
    Also, copy one of the MicroManager metadata files from raw to processed directory.

    Args:
        data_path (str): Relative path to dataset.
        prefix (str):  Full folder name prefix, including round number.
        overwrite (bool, optional): Whether to re-project if files already exist.
            Defaults to False.

    """
    processed_path = get_processed_path(data_path)
    target_path = processed_path / prefix
    target_path.mkdir(parents=True, exist_ok=True)
    roi_dims = get_roi_dimensions(data_path, prefix)
    ops = load_ops(data_path)
    # Change ref tile to a central position where tissue will be
    try:
        metadata = load_metadata(data_path)
        ref_roi = list(metadata["ROI"].keys())[0]
    except FileNotFoundError:
        ref_roi = roi_dims[0, 0]
        warn(f"Metadata file not found, using ROI {ref_roi} as reference tile.")
    ops.update(
        {
            "ref_tile": [
                ref_roi,
                round(roi_dims[0, 1] / 2),
                round(roi_dims[0, 2] / 2),
            ]
        }
    )
    additional_args = f",PREFIX={prefix}"
    if overwrite:
        additional_args += ",OVERWRITE=--overwrite"
    tileproj_job_ids, failed_job = batch_process_tiles(
        data_path, "project_tile", roi_dims=roi_dims, additional_args=additional_args
    )
    # copy one of the tiff metadata files
    raw_path = get_raw_path(data_path)
    metadata_fname = (
        get_raw_filename(data_path, prefix, tile_coors=(roi_dims[0][0], 0, 0))
        + "_metadata.txt"
    )
    if not (target_path / metadata_fname).exists():
        shutil.copy(
            raw_path / prefix / metadata_fname,
            target_path / metadata_fname,
        )

    return tileproj_job_ids


def project_tile_by_coors(tile_coors, data_path, prefix, overwrite=False):
    """Project a single tile by its coordinates.

    Args:
        tile_coors (tuple): (roi, x, y) coordinates of the tile.
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        overwrite (bool, optional): Whether to re-project if files already exist.
            Defaults to False.

    """
    fname = get_raw_filename(data_path, prefix, tile_coors)
    tile_path = str(Path(data_path) / prefix / fname)
    ops = load_ops(data_path)
    # we want to ensure that file all have the same name after projection, even if raw
    # might be different
    r, x, y = tile_coors
    target = str(Path(data_path) / prefix / f"{prefix}_MMStack_{r}-Pos{x:03d}_{y:03d}")
    project_tile(tile_path, ops, overwrite=overwrite, target_name=target)


def project_tile(fname, ops, overwrite=False, sth=13, target_name=None):
    """Calculates projections for a single tile.

    Args:
        fname (str): path to tile *without* `'.ome.tif'` extension.
        ops (dict): dictionary of values from the ops file.
        overwrite (bool): whether to repeat if already completed
        sth (int): size of the structuring element for the fstack projection.
        target_name (str): name of the target file. If None, it will be the same as the
            input file.

    """
    if target_name is None:
        target_name = fname
    print(f"Target name: {target_name}")
    save_path_fstack = get_processed_path(target_name + "_fstack.tif")
    save_path_max = get_processed_path(target_name + "_max.tif")
    save_path_median = get_processed_path(target_name + "_median.tif")
    if not overwrite and (
        save_path_fstack.exists() or save_path_max.exists() or save_path_median.exists()
    ):
        print(f"{fname} already projected...\n")
        return
    print(f"loading {fname}\n")
    im = get_tile_ome(
        get_raw_path(fname + ".ome.tif"),
        get_raw_path(fname + "_metadata.txt"),  # note that this won't be used
        # if use_indexmap is True
        use_indexmap=True,
    )
    print("computing projection\n")
    get_processed_path(fname).parent.mkdir(parents=True, exist_ok=True)
    if ops["make_fstack"]:
        print("making fstack projection\n")
        im_fstack = fstack_channels(
            im.astype(float), sth=sth
        )  # TODO check if float is useful here
        write_stack(im_fstack, save_path_fstack, bigtiff=True)
    if ops["make_median"]:
        print("making median projection\n")
        im_median = np.median(im, axis=3)
        write_stack(im_median, save_path_median, bigtiff=True)
    if ops["make_max"]:
        print("making max projection\n")
        im_max = np.max(im, axis=3)
        write_stack(im_max, save_path_max, bigtiff=True)
    # To check if the focus was correct, we also save a small projectiong along Z
    std_z = np.std(im, axis=(0, 1))
    perc_z = np.percentile(im, 99.9, axis=(0, 1))
    np_z_profile = get_processed_path(target_name + "_zprofile.npz")
    np.savez(np_z_profile, std=std_z, top_1permille=perc_z)


def project_tile_row(data_path, prefix, tile_roi, tile_row, max_col, overwrite=False):
    """Calculate max intensity and extended DOF projections for a row of tiles in an ROI

    Args:
        data_path (str): relative path to dataset
        prefix (str): directory / file name prefix, e.g. 'gene_round'
        tile_roi (int): index of the ROI
        tile_row (int): index of the row to process
        max_col (int): Maximum columns index. Column 0 to max_col will be projected.
        overwrite (bool, optional): whether to redo projection if files already exist.
            Defaults to False.

    """
    n_workers = np.min((mp.cpu_count(), max_col + 1))
    print(f"Starting a pool with {n_workers} workers on {mp.cpu_count()} CPUs.")
    pool = mp.Pool(n_workers)
    cols = range(max_col + 1)
    tile_coors = [[tile_roi, tile_row, tile_col] for tile_col in cols]

    pool.map(
        partial(
            project_tile_by_coors,
            data_path=data_path,
            prefix=prefix,
            overwrite=overwrite,
        ),
        tile_coors,
    )
    pool.close()
