import shlex
import subprocess
import warnings
from pathlib import Path
import numpy as np
import iss_preprocess as iss
from ..image import apply_illumination_correction, tilestats_and_mean_image
from ..io import (
    get_roi_dimensions,
    load_metadata,
    load_ops,
    load_tile_by_coors,
    write_stack,
)
from ..decorators import updates_flexilims
from . import ara_registration as ara_reg
from .hybridisation import load_and_register_hyb_tile
from .sequencing import load_and_register_sequencing_tile


def load_and_register_tile(data_path, tile_coors, prefix, filter_r=True):
    """Load one single tile

    Load a tile of `prefix` with channels/rounds registered, apply illumination correction
    and filtering.

    Args:
        data_path (str): Relative path to data
        tile_coors (tuple): (Roi, tileX, tileY) tuple
        prefix (str): Acquisition to load. If `genes_round` or `barcode_round` will load
            all the rounds.
        filter_r (bool, optional): Apply filter on rounds data? Parameters will be read
            from `ops`. Default to True

    Returns:
        numpy.ndarray: A (X x Y x Nchannels x Nrounds) registered stack
        numpy.ndarray: X x Y boolean mask of bad pixels where data is missing after registration

    """
    ops = load_ops(data_path)
    metadata = load_metadata(data_path)
    projection = ops[f"{prefix.split('_')[0].lower()}_projection"]
    if filter_r and isinstance(filter_r, bool):
        filter_r = ops["filter_r"]
    if prefix.startswith("genes_round") or prefix.startswith("barcode_round"):
        parts = prefix.split("_")
        if len(parts) > 2:
            acq_type = "_".join(parts[:2])
            rounds = np.array([int(parts[2])])
        else:
            acq_type = prefix
            rounds = np.arange(ops[f"{acq_type}s"]) + 1

        stack, bad_pixels = load_and_register_sequencing_tile(
            data_path,
            tile_coors=tile_coors,
            suffix=projection,
            prefix=acq_type,
            filter_r=filter_r,
            correct_channels=True,
            correct_illumination=True,
            corrected_shifts=ops["corrected_shifts"],
            specific_rounds=rounds,
        )
        # the transforms for all rounds are the same and saved with round 1
        prefix = acq_type + "_1_1"

    elif prefix in metadata["hybridisation"]:
        stack, bad_pixels = load_and_register_hyb_tile(
            data_path,
            tile_coors=tile_coors,
            prefix=prefix,
            suffix=projection,
            filter_r=filter_r,
            correct_illumination=True,
            correct_channels=True,
        )
    else:
        stack = load_tile_by_coors(
            data_path, tile_coors=tile_coors, suffix=projection, prefix=prefix
        )
        bad_pixels = np.zeros(stack.shape, dtype=bool)
        stack = apply_illumination_correction(data_path, stack, prefix)

    stack[bad_pixels] = 0
    # ensure we have 4d to match acquisitions with rounds
    if stack.ndim == 3:
        stack = stack[..., np.newaxis]

    return stack, bad_pixels


def batch_process_tiles(data_path, script, roi_dims=None, additional_args=""):
    """Start sbatch scripts for all tiles across all rois.

    Args:
        data_path (str): Relative path to data.
        script (str): Filename stem of the sbatch script, e.g. `extract_tile`.
        roi_dims (numpy.array, optional): Nx3 array of roi dimensions. If None, will
            load `genes_round_1_1` dimensions
        additional_args (str, optional): Additional environment variable to export
            to pass to the sbatch job. Should start with a leading comma.
            Defaults to "".

    """
    if roi_dims is None:
        roi_dims = get_roi_dimensions(data_path)
    script_path = str(Path(__file__).parent.parent.parent / "scripts" / f"{script}.sh")
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])

    job_ids = []  # Store job IDs

    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                args = (
                    f"--export=DATAPATH={data_path},ROI={roi[0]},TILEX={ix},TILEY={iy}"
                )
                args = args + additional_args
                log_fname = f"iss_{script}_{roi[0]}_{ix}_{iy}_%j"
                args = args + f" --output={Path.home()}/slurm_logs/{log_fname}.out"
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

    return job_ids


def create_single_average(
    data_path,
    subfolder,
    subtract_black,
    n_batch,
    prefix_filter=None,
    suffix=None,
    combine_tilestats=False,
    exclude_tiffs=None,
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
        n_batch (int): Number of batch to average before taking their median.
        prefix_filter (str, optional): prefix name to filter tifs. Only file starting
            with `prefix` will be averaged. Defaults to None.
        suffix (str, optional): suffix to filter tifs. Defaults to None
        combine_tilestats (bool, optional): Compute new tilestats distribution of
            averaged images if True, combine pre-existing tilestats into one otherwise.
            Defaults to False
        exclude_tiffs (list, optional): List of str filter to exclude tiffs from average

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
    print(f"    n_batch={n_batch}")
    print(f"    combine_tilestats={combine_tilestats}")
    print("\nArgs read from ops file")
    ops = load_ops(data_path)
    for ops_values in ["average_clip_value", "average_median_filter", "black_level"]:
        print(f"    {ops_values}={ops[ops_values]}")
    print("", flush=True)

    processed_path = iss.io.get_processed_path(data_path)
    if prefix_filter is None:
        target_file = f"{subfolder}_average.tif"
    else:
        target_file = f"{prefix_filter}_average.tif"
    target_stats = target_file.replace("_average.tif", "_tilestats.npy")
    target_file = processed_path / "averages" / target_file
    target_stats = processed_path / "averages" / target_stats
    # ensure the directory exists for first average.
    target_file.parent.mkdir(exist_ok=True)

    black_level = ops["black_level"] if subtract_black else 0

    av_image, tilestats = tilestats_and_mean_image(
        processed_path / subfolder,
        prefix=prefix_filter,
        black_level=black_level,
        n_batch=n_batch,
        max_value=ops["average_clip_value"],
        verbose=True,
        median_filter=ops["average_median_filter"],
        normalise=True,
        suffix=suffix,
        combine_tilestats=combine_tilestats,
        exclude_tiffs=exclude_tiffs,
    )
    write_stack(av_image, target_file, bigtiff=False, dtype="float", clip=False)
    np.save(target_stats, tilestats)
    print(f"Average saved to {target_file}, tilestats to {target_stats}", flush=True)
    return av_image, tilestats


@updates_flexilims
def create_all_single_averages(
    data_path,
    n_batch,
    todo=("genes_rounds", "barcode_rounds", "fluorescence", "hybridisation"),
):
    """Average all tiffs in each folder and then all folders by acquisition type

    Args:
        data_path (str): Path to data, relative to project.
        todo (tuple): type of acquisition to process. Default to `("genes_rounds",
            "barcode_rounds", "fluorescence", "hybridisation")`

    """
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
    metadata = load_metadata(data_path)
    # Collect all folder names
    to_average = []
    for kind in todo:
        if kind.endswith("rounds"):
            folders = [f"{kind[:-1]}_{acq + 1}_1" for acq in range(metadata[kind])]
            to_average.extend(folders)
        elif kind in ("fluorescence", "hybridisation"):
            if kind in metadata.keys():
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
        data_folder = processed_path / folder
        if not data_folder.is_dir():
            warnings.warn(f"{data_folder} does not exists. Skipping")
            continue
        projection = ops[f"{folder.split('_')[0].lower()}_projection"]
        export_args = dict(
            DATAPATH=data_path, SUBFOLDER=folder, SUFFIX=projection, N_BATCH=n_batch
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
            shlex.split(command), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )


@updates_flexilims
def create_grand_averages(data_path, prefix_todo=("genes_round", "barcode_round")):
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
        export_args = dict(DATAPATH=data_path, SUBFOLDER=subfolder, PREFIX=kind)
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        args = (
            args
            + f" --output={Path.home()}/slurm_logs/iss_create_grand_average_%j.out"
            + f" --error={Path.home()}/slurm_logs/iss_create_grand_average_%j.err"
        )
        command = f"sbatch {args} {script_path}"
        print(command)
        subprocess.Popen(
            shlex.split(command), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )


def overview_for_ara_registration(data_path, rois_to_do=None, sigma_blur=10):
    """Generate a stitched overview for registering to the ARA

    ABBA requires pyramidal OME-TIFF with resolution information. We will generate such
    stitched files and save them with a log yaml file indicating info about downsampling

    Args:
        data_path (str): Relative path to the data folder
        rois_to_do (list, optional): ROIs to process. If None (default), process all
            ROIs
        max_pixel_size (float, optional): Pixel size in um for the highest level of the
            pyramid. None to keep original size. Defaults to 1
        sigma_blur (float, optional): sigma of the gaussian filter, in downsampled
            pixel size. Defaults to 10

    """
    processed_path = iss.io.get_processed_path(data_path)
    registration_folder = processed_path / "register_to_ara"
    registration_folder.mkdir(exist_ok=True)
    # also make sure that the relevant subfolders are created
    (registration_folder / "qupath_project").mkdir(exist_ok=True)
    (registration_folder / "deepslice").mkdir(exist_ok=True)

    metadata = load_metadata(data_path)
    if rois_to_do is None:
        rois_to_do = metadata["ROI"].keys()
    roi_slice_pos_um, min_step = ara_reg.find_roi_position_on_cryostat(
        data_path=data_path
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
            shlex.split(command), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )


def setup_channel_correction(data_path, use_slurm=True):
    """Setup channel correction for barcode, genes and hybridisation rounds

    Args:
        data_path (str): Relative path to the data folder
        use_slurm (bool, optional): Whether to use SLURM to run the jobs. Defaults to
            True.

    """
    ops = load_ops(data_path)
    slurm_folder = Path.home() / "slurm_logs"
    if ops["barcode_rounds"] > 0:
        iss.pipeline.estimate_channel_correction(
            data_path,
            prefix="barcode_round",
            nrounds=ops["barcode_rounds"],
            fit_norm_factors=ops["fit_channel_correction"],
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="barcode_channel_correction",
        )
    if ops["genes_rounds"] > 0:
        iss.pipeline.estimate_channel_correction(
            data_path,
            prefix="genes_round",
            nrounds=ops["genes_rounds"],
            fit_norm_factors=ops["fit_channel_correction"],
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="genes_channel_correction",
        )

    iss.pipeline.estimate_channel_correction_hybridisation(
        data_path, use_slurm=use_slurm, slurm_folder=slurm_folder
    )


def call_spots(data_path, genes=True, barcodes=True, hybridisation=True):
    """Master method to run spot calling. Must be run after `iss estimate-shifts`,
    `iss estimate-hyb-shifts`, `iss setup-channel-correction`, and `iss create-grand-averages`.

    Args:
        data_path (str): Relative path to the data folder
        genes (bool, optional): Run genes spot calling. Defaults to True.
        barcodes (bool, optional): Run barcode calling. Defaults to True.
        hybridisation (bool, optional): Run hybridisation spot calling. Defaults to True.

    """
    if genes:
        iss.pipeline.correct_shifts(data_path, prefix="genes_round")
        iss.pipeline.setup_omp(data_path)
        batch_process_tiles(data_path, "extract_tile")

    if barcodes:
        iss.pipeline.correct_shifts(data_path, prefix="barcode_round")
        iss.pipeline.setup_barcode_calling(data_path)
        batch_process_tiles(data_path, "basecall_tile")

    if hybridisation:
        iss.pipeline.correct_hyb_shifts(data_path)
        iss.pipeline.setup_hyb_spot_calling(data_path)
        iss.pipeline.extract_hyb_spots_all(data_path)
