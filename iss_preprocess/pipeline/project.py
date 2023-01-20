import numpy as np
import multiprocessing as mp
import shutil
from functools import partial
from flexiznam.config import PARAMETERS
from pathlib import Path
from os import system
from ..image import fstack_channels
from ..io import get_tile_ome, write_stack
from . import get_roi_dimensions


def project_round(data_path, prefix, overwrite=False):
    """TODO: update to use `batch_process_tiles`

    Args:
        data_path (str): Relative path to dataset.
        prefix (str):  Full folder name prefix, including round number.
        overwrite (bool, optional): _description_. Defaults to False.
    """
    rois_list = get_roi_dimensions(data_path, prefix)
    script_path = str(Path(__file__).parent.parent.parent / "scripts" / "project_tile.sh")
    for roi in rois_list:
        for tilex in range(roi[1] + 1):
            for tiley in range(roi[2] + 1):
                args = f"--export=DATAPATH={data_path},ROI={roi[0]},TILEX={tilex},TILEY={tiley},PREFIX={prefix}"
                if overwrite:
                    args = args + ",OVERWRITE=--overwrite"
                args = (
                    args + f" --output={Path.home()}/slurm_logs/iss_project_tile_%j.out"
                )
                command = f"sbatch {args} {script_path}"
                print(command)
                system(command)

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    raw_path = Path(PARAMETERS["data_root"]["raw"])
    metadata_fname = f"{prefix}_MMStack_{rois_list[0][0]}-Pos000_000_metadata.txt"
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
