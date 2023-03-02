import numpy as np
import multiprocessing as mp
import shutil
from functools import partial
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..image import fstack_channels
from ..io import get_tile_ome, write_stack, get_roi_dimensions, load_ops
from .pipeline import batch_process_tiles


def check_projection(data_path, prefix, suffixes=("max", "fstack")):
    """Check if all tiles have been projected successfully.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        suffixes (tuple, optional): Projection suffixes to check for.
            Defaults to ("max", "fstack").
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = get_roi_dimensions(data_path)
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    all_projected = True
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                fname = f"{prefix}_MMStack_{roi[0]}-Pos{str(ix).zfill(3)}_{str(iy).zfill(3)}"
                for suffix in suffixes:
                    proj_path = (
                        processed_path / data_path / prefix / f"{fname}_{suffix}.tif"
                    )
                    if not proj_path.exists():
                        print(f"{proj_path} missing!")
                        all_projected = False
    if all_projected:
        print("all tiles projected!")


def project_round(data_path, prefix, overwrite=False):
    """Start SLURM jobs to z-project all tiles from a single imaging round.
    Also, copy one of the MicroManager metadata files from raw to processed directory.

    Args:
        data_path (str): Relative path to dataset.
        prefix (str):  Full folder name prefix, including round number.
        overwrite (bool, optional): Whether to re-project if files already exist.
            Defaults to False.
    """
    additional_args = f",PREFIX={prefix}"
    if overwrite:
        additional_args += ",OVERWRITE=--overwrite"

    roi_dims = get_roi_dimensions(data_path, prefix)
    batch_process_tiles(
        data_path, "project_tile", roi_dims=roi_dims, additional_args=additional_args
    )
    # copy one of the tiff metadata files
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    metadata_fname = f"{prefix}_MMStack_{roi_dims[0][0]}-Pos000_000_metadata.txt"
    target_path = processed_path / data_path / prefix
    target_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        raw_path / data_path / prefix / metadata_fname,
        target_path / metadata_fname,
    )


def project_tile_by_coors(tile_coors, data_path, prefix, overwrite=False):
    fname = f"{prefix}_MMStack_{tile_coors[0]}-Pos{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}"
    tile_path = str(Path(data_path) / prefix / fname)
    project_tile(tile_path, overwrite=overwrite)


def project_tile(fname, overwrite=False, sth=13):
    """Calculates extended depth of field and max intensity projections for a single tile.

    Args:
        fname (str): path to tile *without* `'.ome.tif'` extension.
        overwrite (bool): whether to repeat if already completed

    """
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    save_path_fstack = processed_path / (fname + "_fstack.tif")
    save_path_max = processed_path / (fname + "_max.tif")
    if not overwrite and (save_path_fstack.exists() or save_path_max.exists()):
        print(f"{fname} already projected...\n")
        return
    print(f"loading {fname}\n")
    im = get_tile_ome(
        raw_path / (fname + ".ome.tif"), raw_path / (fname + "_metadata.txt")
    )
    print("computing projection\n")
    im_fstack = fstack_channels(im, sth=sth)
    im_max = np.max(im, axis=3)
    (processed_path / fname).parent.mkdir(parents=True, exist_ok=True)
    write_stack(im_fstack, save_path_fstack, bigtiff=True)
    write_stack(im_max, save_path_max, bigtiff=True)


def project_tile_row(data_path, prefix, tile_roi, tile_row, max_col, overwrite=False):
    """Calculate max intensity and extended DOF projections for a row of tiles in an ROI.

    Args:
        data_path (str): relative path to dataset
        prefix (str): directory / file name prefix, e.g. 'gene_round'
        tile_roi (int): index of the ROI
        tile_row (int): index of the row to process
        max_col (int): maximum columns index. Will project tiles from column 0 to max_col
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
