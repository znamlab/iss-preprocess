import subprocess, shlex
import warnings
import numpy as np
from flexiznam.config import PARAMETERS
from pathlib import Path
from . import ara_registration as ara_reg
from ..io import (
    write_stack,
    load_metadata,
    get_roi_dimensions,
    load_ops,
)
from ..image import tilestats_and_mean_image


def batch_process_tiles(data_path, script, additional_args=""):
    """Start sbatch scripts for all tiles across all rois.

    Args:
        data_path (str): Relative path to data.
        script (str): Filename stem of the sbatch script, e.g. `extract_tile`.
        additional_args (str, optional): Additional environment variable to export
            to pass to the sbatch job. Should start with a leading comma.
            Defaults to "".
    """
    roi_dims = get_roi_dimensions(data_path)
    script_path = str(Path(__file__).parent.parent.parent / "scripts" / f"{script}.sh")
    ops = load_ops(data_path)
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                args = (
                    f"--export=DATAPATH={data_path},ROI={roi[0]},TILEX={ix},TILEY={iy}"
                )
                args = args + additional_args
                args = args + f" --output={Path.home()}/slurm_logs/iss_{script}_%j.out"
                command = f"sbatch {args} {script_path}"
                print(command)
                subprocess.Popen(
                    shlex.split(command),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )


def create_single_average(
    data_path,
    subfolder,
    subtract_black,
    prefix_filter=None,
    suffix=None,
    combine_tilestats=False,
):
    """Create normalised average of all tifs in a single folder.

    If prefix_filter is not None, the output will be "{prefix_filter}_average.tif",
    otherwise it will be "{folder_path.name}_average.tif"

    Other arguments are read from `ops`:
        average_clip_value: Value to clip images before averaging.
        normalise: Normalise output maximum to one.

    Args:
        data_path (str): Path to the acquisition folder, relative to `projects` folder
        subfolder (str): subfolder in folder_path containing the tifs to average.
        subtract_black (bool): Subtract black level (read from `ops`)
        prefix_filter (str, optional): prefix name to filter tifs. Only file starting
            with `prefix` will be averaged. Defaults to None.
        suffix (str, optional): suffix to filter tifs. Defaults to None
        combine_tilestats (bool, optional): Compute new tilestats distribution of
            averaged images if True, combine pre-existing tilestats into one otherwise.
            Defaults to False

    Returns:
        np.array: Average image
        np.array: Distribution of pixel values
    """

    print("Creating single average")
    print("  Args:")
    print(f"    data_path={data_path}")
    print(f"    subfolder={subfolder}")
    print(f"    subtract_black={subtract_black}")
    print(f"    prefix_filter={prefix_filter}")
    print(f"    suffix={suffix}")
    print(f"    combine_tilestats={combine_tilestats}", flush=True)

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    if prefix_filter is None:
        target_file = f"{subfolder}_average.tif"
    else:
        target_file = f"{prefix_filter}_average.tif"
    target_stats = target_file.replace("_average.tif", "_tilestats.npy")
    target_file = processed_path / data_path / "averages" / target_file
    target_stats = processed_path / data_path / "averages" / target_stats
    # ensure the directory exists for first average.
    target_file.parent.mkdir(exist_ok=True)

    black_level = ops["black_level"] if subtract_black else 0

    av_image, tilestats = tilestats_and_mean_image(
        processed_path / data_path / subfolder,
        prefix=prefix_filter,
        black_level=black_level,
        max_value=ops["average_clip_value"],
        verbose=True,
        median_filter=ops["average_median_filter"],
        normalise=True,
        suffix=suffix,
        combine_tilestats=combine_tilestats,
    )
    write_stack(av_image, target_file, bigtiff=False, dtype="float", clip=False)
    np.save(target_stats, tilestats)
    print(f"Average saved to {target_file}, tilestats to {target_stats}", flush=True)
    return av_image, tilestats


def create_all_single_averages(
    data_path,
    todo=("genes_rounds", "barcode_rounds", "fluorescence", "hybridisation"),
):
    """Average all tiffs in each folder and then all folders by acquisition type

    Args:
        data_path (str): Path to data, relative to project.
        todo (tuple): type of acquisition to process. Default to `("genes_rounds",
            "barcode_rounds", "fluorescence", "hybridisation")`
    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    metadata = load_metadata(data_path)

    # Collect all folder names
    to_average = []
    for kind in todo:
        if kind.endswith("rounds"):
            folders = [f"{kind[:-1]}_{acq + 1}_1" for acq in range(metadata[kind])]
            to_average.extend(folders)
        elif kind in ("fluorescence", "hybridisation"):
            to_average.extend(list(metadata[kind].keys()))
        else:
            raise IOError(
                f"Unknown type of acquisition: {kind}.\n"
                + "Valid types are 'XXXXX_rounds', 'fluorescence', 'hybridisation'"
            )

    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "create_single_average.sh"
    )
    for folder in to_average:
        data_folder = processed_path / data_path
        if not data_folder.is_dir():
            warnings.warn("{0} does not exists. Skipping".format(data_folder / folder))
            continue
        export_args = dict(
            DATAPATH=data_path,
            SUBFOLDER=folder,
            SUFFIX=ops["projection"],
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_create_single_average_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_create_single_average_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def create_grand_averages(
    data_path,
    prefix_todo=("genes_round", "barcode_round"),
):
    """Average single acquisition averages into grand average

    Args:
        data_path (str): Path to the folder, relative to `projects` folder
        prefix_todo (tuple, optional): List of str, names of the tifs to average.
            Defaults to ("genes_round", "barcode_round").
    """
    subfolder = "averages"
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "create_grand_average.sh"
    )
    for kind in prefix_todo:
        export_args = dict(
            DATAPATH=data_path,
            SUBFOLDER=subfolder,
            PREFIX=kind,
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_create_grand_average_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_create_grand_average_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def overview_for_ara_registration(
    data_path,
    rois_to_do=None,
    bulb_first=True,
    sigma_blur=10,
):
    """Generate a stitched overview for registering to the ARA

    ABBA requires pyramidal OME-TIFF with resolution information. We will generate such
    stitched files and save them with a log yaml file indicating info about downsampling

    Args:
        data_path (str): Relative path to the data folder
        rois_to_do (list, optional): ROIs to process. If None (default), process all
            ROIs
        max_pixel_size (float, optional): Pixel size in um for the highest level of the
            pyramid. None to keep original size. Defaults to 1
        bulb_first (bool, optional): Was the first slice closer to the olfactory
            bulb than the last? Defaults to True.
        sigma_blur (float, optional): sigma of the gaussian filter, in downsampled
            pixel size. Defaults to 10
    """

    processed_path = Path(PARAMETERS["data_root"]["processed"])
    registration_folder = processed_path / data_path / "register_to_ara"
    registration_folder.mkdir(exist_ok=True)
    # also make sure that the relevant subfolders are created
    (registration_folder / "qupath_project").mkdir(exist_ok=True)
    (registration_folder / "deepslice").mkdir(exist_ok=True)

    metadata = load_metadata(data_path)
    if rois_to_do is None:
        rois_to_do = metadata["ROI"].keys()
    roi_slice_pos_um, min_step = ara_reg.find_roi_position_on_cryostat(
        data_path=data_path, bulb_first=bulb_first
    )
    roi2section_order = {
        roi: int(pos / min_step) for roi, pos in roi_slice_pos_um.items()
    }
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "overview_single_roi.sh"
    )

    for roi in rois_to_do:

        export_args = dict(
            DATAPATH=data_path,
            ROI=roi,
            SIGMA=sigma_blur,
            SLICE_ID=roi2section_order[roi],
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_overview_roi_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_overview_roi_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
