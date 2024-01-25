import shlex
import subprocess
import warnings
from pathlib import Path
import numpy as np
import iss_preprocess as iss
import flexiznam as flz
import os
from ..image import apply_illumination_correction
from ..io import (
    get_roi_dimensions,
    load_metadata,
    load_ops,
    load_tile_by_coors,
)
from znamutils import slurm_it
from ..decorators import updates_flexilims
from . import ara_registration as ara_reg
from .hybridisation import load_and_register_hyb_tile
from .sequencing import load_and_register_sequencing_tile


def project_and_average(data_path, force_redo=False):
    """Project and average all available data then create plots.

    Creates a list of expected acquisition folders from metadata
    Checks for the existence of expected folders in the raw data
    and determines the completion status of each acquisition type.
    Runs projection on unprojected data and reprojects failed tiles.
    Creates averages of projections and then plots overview images.

    Args:
        data_path (str): Relative path to data.
        force_redo (bool, optional): Redo all processing steps? Defaults to False.

    Returns:
        po_job_ids (list): A list of job IDs for the slurm jobs created.
    """

    processed_path = iss.io.get_processed_path(data_path)
    metadata = iss.io.load_metadata(data_path)
    slurm_folder = processed_path / "slurm_scripts"
    slurm_folder.mkdir(parents=True, exist_ok=True)

    # First, set up flexilims, adding chamber
    iss.pipeline.setup_flexilims(data_path)

    # Make a list of expected acquisition folders using metadata.yml
    todo = ("genes_rounds", "barcode_rounds", "fluorescence", "hybridisation")
    data_by_kind = {kind: [] for kind in todo}
    acquisition_complete = {kind: True for kind in todo}
    for kind in todo:
        if kind.endswith("rounds"):
            data_by_kind[kind] = [
                f"{kind[:-1]}_{acq + 1}_1" for acq in range(metadata[kind])
            ]
        elif kind in ("fluorescence", "hybridisation"):
            data_by_kind[kind] = list(metadata[kind].keys())
    print("Files expected:")
    print(data_by_kind, flush=True)

    # Check for expected folders in raw_data and check acquisition types for completion
    raw_path = iss.io.get_raw_path(data_path)
    to_process = []
    print(f"Checking for expected folders in {raw_path}", flush=True)
    for kind in todo:
        for folder in data_by_kind[kind]:
            if not (raw_path / folder).exists():
                print(f"{folder} not found in raw, skipping", flush=True)
                acquisition_complete[kind] = False
                continue
        to_process.append(folder)

    # Run projection on unprojected data
    pr_job_ids = []
    proj, mouse, chamber = data_path.split(os.sep)[:-1]
    flm_sess = flz.get_flexilims_session(project_id=proj)
    print(f"to_process: {to_process}")
    for prefix in to_process:
        if not force_redo:
            # Skip if already projected
            flm_dataset = flz.get_entity(
                name="_".join([mouse, chamber, f"project_round_{prefix}"]),
                flexilims_session=flm_sess,
            )
            if flm_dataset is not None:
                print(f"{prefix} is already projected, continuing", flush=True)
                continue
        tileproj_job_ids, _ = iss.pipeline.project_round(
            data_path, prefix, overview=False
        )
        pr_job_ids.extend(tileproj_job_ids)

    # TODO: Before proceeding, check all tiles really are projected (slurm randomly fails sometimes)
    #       This would need to take pr_job_ids and output more job_ids for csa to wait for

    # Something like this
    for prefix in to_process:
        check_proj_job_ids = iss.pipeline.check_projection(
            data_path,
            prefix,
            use_slurm=True,
            slurm_folder=f"{Path.home()}/slurm_logs",
            scripts_name=f"check_projection_{folder}",
            job_dependency=",".join(pr_job_ids),
        )

    # Then run iss.pipeline.reproject_failed() which opens txt files from check projection
    # and reprojects failed tiles, collecting job_ids for each tile
    reproj_job_ids = iss.pipeline.reproject_failed(
        data_path, dependency=check_proj_job_ids
    )
    reproj_job_ids = reproj_job_ids if reproj_job_ids else []

    # Then create averages of projections
    csa_job_ids = iss.pipeline.create_all_single_averages(
        data_path, n_batch=1, to_average=to_process, dependency=reproj_job_ids
    )

    # Create grand averages if all rounds are projected
    if acquisition_complete["genes_rounds"] or acquisition_complete["barcode_rounds"]:
        cga_job_ids = iss.pipeline.create_grand_averages(
            data_path, dependency=csa_job_ids
        )
    else:
        print(
            "All rounds not yet projected, skipping grand average creation", flush=True
        )
        cga_job_ids = None

    plot_job_ids = csa_job_ids if csa_job_ids else []
    if cga_job_ids:
        plot_job_ids.extend(cga_job_ids)

    # TODO: When plotting overview, check whether grand average has occured if it is a
    # 'round' type, use it if so, otherwise use single average.

    po_job_ids = []
    for prefix in to_process:
        flm_dataset = flz.get_entity(
            name="_".join([mouse, chamber, f"plot_single_overview_{prefix}"]),
            flexilims_session=flm_sess,
        )
        if not force_redo:
            if flm_dataset is not None:
                print(f"{prefix} is already plotted, continuing", flush=True)
                continue
        job_id = iss.vis.plot_overview_images(
            data_path=data_path,
            prefix=prefix,
            plot_grid=True,
            downsample_factor=25,
            save_raw=False,
            dependency=plot_job_ids,
        )
        po_job_ids.extend(job_id)

    return po_job_ids


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
        bad_pixels = np.zeros(stack.shape[:2], dtype=bool)
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


@slurm_it(conda_env="iss-preprocess")
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

    av_image, tilestats = iss.image.tilestats_and_mean_image(
        processed_path / subfolder,
        prefix=prefix_filter,
        black_level=black_level,
        n_batch=n_batch,
        max_value=ops["average_clip_value"],
        verbose=True,
        median_filter_size=ops["average_median_filter"],
        normalise=True,
        suffix=suffix,
        combine_tilestats=combine_tilestats,
        exclude_tiffs=exclude_tiffs,
    )
    iss.io.write_stack(av_image, target_file, bigtiff=False, dtype="float", clip=False)
    np.save(target_stats, tilestats)
    print(f"Average saved to {target_file}, tilestats to {target_stats}", flush=True)
    return av_image, tilestats


@updates_flexilims
def create_all_single_averages(
    data_path,
    n_batch,
    todo=("genes_rounds", "barcode_rounds", "fluorescence", "hybridisation"),
    to_average=None,
    dependency=None,
):
    """Average all tiffs in each folder and then all folders by acquisition type

    Args:
        data_path (str): Path to data, relative to project.
        todo (tuple): type of acquisition to process. Default to `("genes_rounds",
            "barcode_rounds", "fluorescence", "hybridisation")`

    """
    processed_path = iss.io.get_processed_path(data_path)
    ops = iss.io.load_ops(data_path)
    metadata = iss.io.load_metadata(data_path)
    # Collect all folder names
    if to_average is None:
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

    job_ids = []
    for folder in to_average:
        data_folder = processed_path / folder
        if not data_folder.is_dir():
            warnings.warn(f"{data_folder} projected data does not exist. Skipping")
            continue
        average_image = processed_path / "averages" / f"{folder}_average.tif"
        if average_image.exists():
            print(f"{folder} average already exists. Skipping")
            continue
        projection = ops[f"{folder.split('_')[0].lower()}_projection"]
        job_ids.append(
            create_single_average(
                data_path,
                folder,
                n_batch=n_batch,
                subtract_black=True,
                prefix_filter=None,
                suffix=projection,
                use_slurm=True,
                slurm_folder=f"{Path.home()}/slurm_logs",
                scripts_name=f"create_single_average_{folder}",
                job_dependency=",".join(dependency) if dependency else None,
            )
        )
    return job_ids


@updates_flexilims
def create_grand_averages(
    data_path,
    prefix_todo=("genes_round", "barcode_round"),
    n_batch=1,
    dependency=None,
):
    """Average single acquisition averages into grand average

    Args:
        data_path (str): Path to the folder, relative to `projects` folder
        prefix_todo (tuple, optional): List of str, names of the tifs to average.
            Defaults to ("genes_round", "barcode_round").

    """
    subfolder = "averages"
    job_ids = []
    for kind in prefix_todo:
        job_ids.append(
            create_single_average(
                data_path,
                subfolder,
                n_batch=n_batch,
                subtract_black=False,
                prefix_filter=kind,
                combine_tilestats=True,
                suffix="_1_average",
                use_slurm=True,
                slurm_folder=f"{Path.home()}/slurm_logs",
                scripts_name=f"create_grand_average_{kind}",
                job_dependency=",".join(dependency) if dependency else None,
            )
        )
    return job_ids


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


def setup_flexilims(path):
    data_path = Path(path)
    flm_session = flz.get_flexilims_session(project_id=data_path.parts[0])
    # first level, which is the mouse, must exist
    mouse = flz.get_entity(
        name=data_path.parts[1], datatype="mouse", flexilims_session=flm_session
    )
    if mouse is None:
        raise ValueError(f"Mouse {data_path.parts[1]} does not exist in flexilims")
    else:
        if "genealogy" or "path" not in mouse:
            flz.update_entity(
                datatype="mouse",
                flexilims_session=flm_session,
                id=mouse["id"],
                mode="update",
                attributes=dict(
                    genealogy=[mouse["name"]], path="/".join(data_path.parts[:2])
                ),
            )
    parent_id = mouse["id"]
    for sample_name in data_path.parts[2:]:
        sample = flz.add_sample(
            parent_id,
            attributes=None,
            sample_name=sample_name,
            conflicts="skip",
            other_relations=None,
            flexilims_session=flm_session,
        )
        parent_id = sample["id"]
