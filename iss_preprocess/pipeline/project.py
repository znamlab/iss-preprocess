import numpy as np
import multiprocessing as mp
import shutil
import shlex
import iss_preprocess as iss
import subprocess
from warnings import warn
from functools import partial
from pathlib import Path
from ..image import fstack_channels
from ..io import get_tile_ome, write_stack, get_roi_dimensions, load_ops
from .pipeline import batch_process_tiles
from ..decorators import updates_flexilims
from znamutils import slurm_it

@slurm_it(conda_env="iss-preprocess")
def check_projection(data_path, prefix, suffixes=("max", "fstack")):
    """Check if all tiles have been projected successfully.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        suffixes (tuple, optional): Projection suffixes to check for.
            Defaults to ("max", "fstack").

    """
    processed_path = iss.io.get_processed_path(data_path)
    if prefix is None:
        metadata = iss.io.load_metadata(data_path)
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
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    all_projected = True
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                fname = f"{prefix}_MMStack_{roi[0]}-Pos{str(ix).zfill(3)}_{str(iy).zfill(3)}"
                for suffix in suffixes:
                    proj_path = processed_path / prefix / f"{fname}_{suffix}.tif"
                    not_projected = []
                    if not proj_path.exists():
                        print(f"{proj_path} missing!")
                        all_projected = False
                        not_projected.append(fname)

    if all_projected:
        print(f"all tiles projected for {prefix}!")
    else:
       np.savetxt(
           processed_path / prefix / "missing_tiles.txt",
           not_projected,
           fmt="%s",
           )

@slurm_it(conda_env="iss-preprocess")
def reproject_failed(
    data_path,
):
    """Re-project tiles that failed to project previously."""
    processed_path = iss.io.get_processed_path(data_path)
    missing_tiles = []
    for prefix in processed_path.iterdir():
        if not prefix.is_dir():
            continue
        for fname in (prefix / "missing_tiles.txt").read_text().split("\n"):
            if len(fname) == 0:
                continue
            missing_tiles.append(fname)
    job_ids = []
    for fname in missing_tiles:
        for tile in missing_tiles:
            prefix = tile.split("_MMStack")[0]
            roi = int(tile.split("_MMStack")[1].split("-")[0])
            ix = int(tile.split("_MMStack")[1].split("-Pos")[1].split("_")[0])
            iy = int(tile.split("_MMStack")[1].split("-Pos")[1].split("_")[1])
            args = (
            f"--export=DATAPATH={data_path},PREFIX={prefix},"
            f"ROI={roi},TILEX={ix},TILEY={iy},OVERWRITE=--overwrite"
            )
            log_fname = f"iss_reproject_{roi[0]}_{ix}_{iy}_%j"
            args = args + f" --output={Path.home()}/slurm_logs/{log_fname}.out"
            script_path = Path(__file__).parent / "project_tile.sh"
            command = f"sbatch --parsable {args} {script_path}"
            print(command)
            process = subprocess.Popen(
                shlex.split(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = process.communicate()
            job_id = stdout.decode().strip().split(";")[0]  # Extract the job ID
            job_ids.append(job_id)
    if len(missing_tiles) == 0:
        print("No failed tiles to re-project!")
    else:
        return job_ids

@updates_flexilims(name_source="prefix")
def project_round(data_path, prefix, overwrite=False):
    """Start SLURM jobs to z-project all tiles from a single imaging round.
    Also, copy one of the MicroManager metadata files from raw to processed directory.

    Args:
        data_path (str): Relative path to dataset.
        prefix (str):  Full folder name prefix, including round number.
        overwrite (bool, optional): Whether to re-project if files already exist.
            Defaults to False.

    """
    processed_path = iss.io.get_processed_path(data_path)
    target_path = processed_path / prefix
    target_path.mkdir(parents=True, exist_ok=True)
    roi_dims = get_roi_dimensions(data_path, prefix)
    ops = load_ops(data_path)
    # Change ref tile to a central position where tissue will be
    try:
        metadata = iss.io.load_metadata(data_path)
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
    tileproj_job_ids = batch_process_tiles(
        data_path, "project_tile", roi_dims=roi_dims, additional_args=additional_args
    )
    # copy one of the tiff metadata files
    raw_path = iss.io.get_raw_path(data_path)
    metadata_fname = f"{prefix}_MMStack_{roi_dims[0][0]}-Pos000_000_metadata.txt"
    if not (target_path / metadata_fname).exists():
        shutil.copy(
            raw_path / prefix / metadata_fname, target_path / metadata_fname,
        )
 
    overview_job_ids = iss.vis.plot_overview_images(data_path, prefix, dependency=','.join(tileproj_job_ids))
    
    return tileproj_job_ids, overview_job_ids

def project_tile_by_coors(tile_coors, data_path, prefix, overwrite=False):
    """Project a single tile by its coordinates.

    Args:
        tile_coors (tuple): (roi, x, y) coordinates of the tile.
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix, e.g. "genes_round_1_1".
        overwrite (bool, optional): Whether to re-project if files already exist.
            Defaults to False.

    """
    fname = f"{prefix}_MMStack_{tile_coors[0]}-Pos{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}"
    tile_path = str(Path(data_path) / prefix / fname)
    ops = load_ops(data_path)
    project_tile(tile_path, ops, overwrite=overwrite)


def project_tile(fname, ops, overwrite=False, sth=13):
    """Calculates extended depth of field and max intensity projections for a single tile.

    Args:
        fname (str): path to tile *without* `'.ome.tif'` extension.
        overwrite (bool): whether to repeat if already completed

    """
    save_path_fstack = iss.io.get_processed_path(fname + "_fstack.tif")
    save_path_max = iss.io.get_processed_path(fname + "_max.tif")
    save_path_median = iss.io.get_processed_path(fname + "_median.tif")
    if not overwrite and (
        save_path_fstack.exists() or save_path_max.exists() or save_path_median.exists()
    ):
        print(f"{fname} already projected...\n")
        return
    print(f"loading {fname}\n")
    im = get_tile_ome(
        iss.io.get_raw_path(fname + ".ome.tif"),
        iss.io.get_raw_path(fname + "_metadata.txt"),
    )
    print("computing projection\n")
    iss.io.get_processed_path(fname).parent.mkdir(parents=True, exist_ok=True)
    if ops["make_fstack"]:
        print("making fstack projection\n")
        im_fstack = fstack_channels(im, sth=sth)
        write_stack(im_fstack, save_path_fstack, bigtiff=True)
    if ops["make_median"]:
        print("making median projection\n")
        im_median = np.median(im, axis=3)
        write_stack(im_median, save_path_median, bigtiff=True)
    if ops["make_max"]:
        print("making max projection\n")
        im_max = np.max(im, axis=3)
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
