import shlex
import subprocess
import warnings
from pathlib import Path

import numpy as np
from znamutils import slurm_it

from ..decorators import updates_flexilims
from ..diagnostics import check_illumination_correction
from ..diagnostics.diag_register import (
    check_ref_tile_registration,
    check_shift_correction,
    check_tile_registration,
    check_tile_shifts,
)
from ..diagnostics.diag_stitching import plot_overview_images
from ..image import tilestats_and_mean_image
from ..io import (
    find_roi_position_on_cryostat,
    get_processed_path,
    get_raw_path,
    load_metadata,
    load_ops,
    write_stack,
)
from .align_spots_and_cells import stitch_cell_dataframes
from .core import batch_process_tiles, setup_flexilims
from .hybridisation import (
    estimate_channel_correction_hybridisation,
    extract_hyb_spots_all,
    setup_hyb_spot_calling,
)
from .project import check_projection, check_roi_dims, project_round, reproject_failed
from .register import (
    correct_hyb_shifts,
    run_correct_shifts,
    run_register_reference_tile,
)
from .segment import (
    filter_mcherry_cells,
    remove_all_duplicate_masks,
    save_unmixing_coefficients,
)
from .sequencing import estimate_channel_correction, setup_barcode_calling, setup_omp
from .stitch import register_all_rois_within

__all__ = [
    "project_and_average",
    "register_acquisition",
    "register_reference_tile",
    "correct_shifts",
    "create_single_average",
    "create_all_single_averages",
    "create_grand_averages",
    "overview_for_ara_registration",
    "setup_channel_correction",
    "call_spots",
    "segment_and_stitch_mcherry_cells",
]


@slurm_it(conda_env="iss-preprocess")
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

    processed_path = get_processed_path(data_path)
    metadata = load_metadata(data_path)
    slurm_folder = processed_path / "slurm_scripts"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    ops = load_ops(data_path)
    # First, set up flexilims, adding chamber
    if ops["use_flexilims"]:
        setup_flexilims(data_path)

    todo = ["genes_rounds", "barcode_rounds", "fluorescence", "hybridisation"]
    # Make a list of expected acquisition folders using metadata.yml
    if metadata["genes_rounds"] == 0:
        todo.remove("genes_rounds")
    if metadata["barcode_rounds"] == 0:
        todo.remove("barcode_rounds")
    if ("fluorescence" not in metadata.keys()) or (len(metadata["fluorescence"]) == 0):
        todo.remove("fluorescence")
    if ("hybridisation" not in metadata.keys()) or (
        len(metadata["hybridisation"]) == 0
    ):
        todo.remove("hybridisation")

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
    raw_path = get_raw_path(data_path)
    to_process = []
    print(f"\nChecking for expected folders in {raw_path}", flush=True)
    for kind in todo:
        for folder in data_by_kind[kind]:
            if not (raw_path / folder).exists():
                print(f"{folder} not found in raw, skipping", flush=True)
                acquisition_complete[kind] = False
                continue
            print(f"{folder} found in raw", flush=True)
            to_process.append(folder)

    # Run projection on unprojected data
    pr_job_ids = []
    proj, mouse, chamber = Path(data_path).parts
    print(f"\nto_process: {to_process}")
    for prefix in to_process:
        if not force_redo:
            if (processed_path / prefix / "missing_tiles.txt").exists():
                print(f"{prefix} is already projected, continuing", flush=True)
                continue
        tileproj_job_ids, _ = project_round(data_path, prefix)
        pr_job_ids.extend(tileproj_job_ids)
    pr_job_ids = pr_job_ids if pr_job_ids else None

    # Now check that all roi_dims are the same, can sometimes be truncated
    # if project_round occurs during data transfer
    roi_dim_job_ids = check_roi_dims(
        data_path,
        use_slurm=True,
        slurm_folder=slurm_folder,
        scripts_name=f"check_roi_dims_{prefix}",
        dependency_type="afterany",
        job_dependency=pr_job_ids,
    )
    print(f"check_roi_dims job ids: {roi_dim_job_ids}", flush=True)

    # Before proceeding, check all tiles really are projected (slurm randomly fails
    # sometimes)
    all_check_proj_job_ids = []
    for prefix in to_process:
        slurm_folder = Path.home() / "slurm_logs" / data_path
        slurm_folder.mkdir(parents=True, exist_ok=True)
        check_proj_job_ids = check_projection(
            data_path,
            prefix,
            use_slurm=True,
            slurm_folder=slurm_folder,
            scripts_name=f"check_projection_{prefix}",
            dependency_type="afterany",
            job_dependency=roi_dim_job_ids,
        )
        all_check_proj_job_ids.append(check_proj_job_ids)
    all_check_proj_job_ids = all_check_proj_job_ids if all_check_proj_job_ids else None
    print(f"check_projection job ids: {all_check_proj_job_ids}", flush=True)

    # Then run reproject_failed() which opens txt files from check
    # projection and reprojects failed tiles, collecting job_ids for each tile
    slurm_folder = Path.home() / "slurm_logs" / data_path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    reproj_job_ids = reproject_failed(
        data_path,
        use_slurm=True,
        slurm_folder=slurm_folder,
        scripts_name="reproject_failed",
        dependency_type="afterany",
        job_dependency=all_check_proj_job_ids,
    )
    reproj_job_ids = reproj_job_ids if reproj_job_ids else None
    print(f"reproject_failed job ids: {reproj_job_ids}", flush=True)

    # Check tiles again to update missing tiles
    all_check_proj_job_ids = []
    for prefix in to_process:
        slurm_folder = Path.home() / "slurm_logs" / data_path
        slurm_folder.mkdir(parents=True, exist_ok=True)
        check_proj_job_ids = check_projection(
            data_path,
            prefix,
            use_slurm=True,
            slurm_folder=slurm_folder,
            scripts_name=f"check_projection_{prefix}",
            dependency_type="afterany",
            job_dependency=reproj_job_ids,
        )
        all_check_proj_job_ids.append(check_proj_job_ids)
    all_check_proj_job_ids = all_check_proj_job_ids if all_check_proj_job_ids else None
    print(f"check_projection job ids: {all_check_proj_job_ids}", flush=True)

    # Then create averages of projections
    csa_job_ids = create_all_single_averages(
        data_path,
        n_batch=1,
        to_average=to_process,
        dependency=all_check_proj_job_ids,
        force_redo=force_redo,
    )

    # Create grand averages if all rounds of a certain type are projected
    prefix_todo = []
    if acquisition_complete.get("genes_rounds", False):
        prefix_todo.append("genes_round")
    if acquisition_complete.get("barcode_rounds", False):
        prefix_todo.append("barcode_round")
    if all(acquisition_complete.values()):
        prefix_todo.append("")

    if prefix_todo:
        cga_job_ids = create_grand_averages(
            data_path,
            dependency=csa_job_ids if csa_job_ids else None,
            force_redo=force_redo,
            prefix_todo=prefix_todo,
        )
    else:
        print(
            "All rounds not yet projected, skipping grand average creation", flush=True
        )
        cga_job_ids = None
    print(f"create_single_average job ids: {csa_job_ids}", flush=True)
    print(f"create_grand_average job ids: {cga_job_ids}", flush=True)

    plot_job_ids = csa_job_ids if csa_job_ids else None
    if cga_job_ids and plot_job_ids:
        plot_job_ids.extend(cga_job_ids)

    # TODO: When plotting overview, check whether grand average has occurred if it is a
    # 'round' type, use it if so, otherwise use single average.
    po_job_ids = []
    for prefix in to_process:
        if not force_redo:
            if (
                processed_path
                / "figures"
                / "round_overviews"
                / f"{Path(data_path).parts[2]}_roi_01_{prefix}_channels_0_1_2_3.png"
            ).exists():
                print(f"{prefix} is already plotted, continuing", flush=True)
                continue
        job_id = plot_overview_images(
            data_path,
            prefix,
            plot_grid=True,
            downsample_factor=25,
            save_raw=True,
            dependency=plot_job_ids,
        )
        po_job_ids.extend(job_id)
    print(f"create_grand_average job ids: {cga_job_ids}", flush=True)
    print("All jobs submitted", flush=True)


@slurm_it(conda_env="iss-preprocess")
def register_acquisition(data_path, prefix, force_redo=False):
    """Register an acquisition across all rounds and channels

    Args:
        path (str): Path to the data folder
        prefix (str): Prefix of the acquisition to register
        force_redo (bool, optional): Redo if files exist. Defaults to False.

    """
    ops = load_ops(data_path)
    sprefix = prefix.split("_")[0]  # short prefix, e.g. 'genes'
    if prefix.startswith("genes_round") or prefix.startswith("barcode_round"):
        # Register a sequencing acquisition
        job_id, diag_job = register_reference_tile(
            data_path, prefix, diag=True, use_slurm=True, force_redo=force_redo
        )
        suffix = ops[f"{sprefix}_projection"]
        additional_args = f",PREFIX={prefix},SUFFIX={suffix}"
        batch_id, rerun_id = batch_process_tiles(
            data_path,
            script="register_tile",
            additional_args=additional_args,
            job_dependency=job_id,
        )
        print(f"Final job id: {rerun_id}")
        jid = correct_shifts(data_path, prefix, use_slurm=True, job_dependency=rerun_id)
        print(f"Correct shifts job id: {jid}")

    else:
        # Register a single round fluorescence acquisition
        raise NotImplementedError("Single round registration not yet implemented")


def register_reference_tile(
    data_path, prefix="genes_round", diag=False, use_slurm=True, force_redo=False
):
    """Register the reference tile across channels and rounds

    This function estimates the shifts and rotations between rounds and
    channels using the reference tile and generates diagnostic plots if
    requested.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): Directory prefix to use, e.g. 'genes_round'.
            Defaults to 'genes_round'.
        diag (bool, optional): Save diagnostic plots. Defaults to False.
        use_slurm (bool, optional): Submit job to slurm. Defaults to True.
        redo (bool, optional): Redo if files exist. Defaults to False.
    """
    if use_slurm:
        slurm_folder = Path.home() / "slurm_logs" / data_path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    target_file = get_processed_path(data_path) / "reg" / prefix
    target_file /= f"ref_tile_tforms_{prefix}.npz"
    if target_file.exists() and not force_redo:
        print(f"{prefix} reference tile already registered, skipping")
        return None, None

    scripts_name = f"register_ref_tile_{prefix}"
    slurm_options = {"mem": "128G"} if diag else None
    job_id = run_register_reference_tile(
        data_path,
        prefix=prefix,
        diag=diag,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        slurm_options=slurm_options,
        scripts_name=scripts_name,
    )
    scripts_name = f"check_ref_tile_registration_{prefix}"
    job2 = check_ref_tile_registration(
        data_path,
        prefix,
        use_slurm=True,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=scripts_name,
    )
    if use_slurm:
        print(f"Started 2 jobs: {job_id}, {job2}")
    return job_id, job2


def correct_shifts(path, prefix, use_slurm=True, job_dependency=None):
    """Correct X-Y shifts using robust regression across tiles."""
    # import with different name to not get confused with the cli function name

    if use_slurm:
        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    job_id = run_correct_shifts(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"correct_shifts_{prefix}",
        job_dependency=job_dependency,
    )
    check_tile_shifts(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"correct_shifts_{prefix}",
        job_dependency=job_id if use_slurm else None,
    )
    check_corr_id = check_shift_correction(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_shift_correction_{prefix}",
    )
    check_reg_id = check_tile_registration(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_tile_registration_{prefix}",
    )
    return job_id, check_corr_id, check_reg_id


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
            If None, will do as many batches as images.
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
    print("\nArgs read from ops file", flush=True)
    ops = load_ops(data_path)
    for ops_values in [
        "average_clip_value",
        "average_median_filter",
        "average_gaussian_filter",
        "black_level",
    ]:
        print(f"    {ops_values}={ops[ops_values]}")
    print("", flush=True)

    processed_path = get_processed_path(data_path)

    if prefix_filter:
        target_file = f"{prefix_filter}_average.tif"
    else:
        target_file = f"{subfolder}_average.tif"
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
        gaussian_filter_size=ops["average_gaussian_filter"],
        median_filter_size=ops["average_median_filter"],
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
    to_average=None,
    dependency=None,
    force_redo=False,
):
    """Average all tiffs in each folder and then all folders by acquisition type

    Args:
        data_path (str): Path to data, relative to project.
        n_batch (int): Number of batch to average before taking their median.
            If None, will do as many batches as images.
        todo (tuple): type of acquisition to process. Default to `("genes_rounds",
            "barcode_rounds", "fluorescence", "hybridisation")`. Ignored if `to_average`
            is not None.
        to_average (list, optional): List of folders to average. If None, will average
            all folders listed in metadata. Defaults to None.
        dependency (list, optional): List of job IDs to wait for before starting the
            current job. Defaults to None.
        force_redo (bool, optional): Redo if the average already exists. Defaults to
            False.
    """
    processed_path = get_processed_path(data_path)
    ops = load_ops(data_path)
    metadata = load_metadata(data_path)
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
        if (not force_redo) and average_image.exists():
            print(f"{folder} average already exists. Skipping")
            continue
        print(f"Creating single average {folder}", flush=True)
        projection = ops["averaging_projection"]
        slurm_folder = Path.home() / "slurm_logs" / data_path / "averages"
        slurm_folder.mkdir(parents=True, exist_ok=True)
        job_ids.append(
            create_single_average(
                data_path,
                folder,
                n_batch=n_batch,
                subtract_black=True,
                prefix_filter=None,
                suffix=projection,
                use_slurm=True,
                slurm_folder=slurm_folder,
                scripts_name=f"create_single_average_{folder}",
                dependency_type="afterany",
                job_dependency=dependency if dependency else None,
            )
        )
    return job_ids


@updates_flexilims
def create_grand_averages(
    data_path,
    prefix_todo=("genes_round", "barcode_round", ""),
    n_batch=None,
    dependency=None,
    force_redo=False,
):
    """Average single acquisition averages into grand average

    Args:
        data_path (str): Path to the folder, relative to `projects` folder
        prefix_todo (tuple, optional): List of str, names of the tifs to average. An
            empty string will average all tifs. Defaults to ("genes_round",
                "barcode_round", "").
        n_batch (int, optional): Number of batch to average before taking their median.
            If None, will do as many batches as images. Defaults to None.
        force_redo (bool, optional): Redo if the average already exists. Defaults to
            False.

    """
    if prefix_todo is None:
        prefix_todo = []
        metadata = load_metadata(data_path)
        if metadata["genes_rounds"] > 0:
            prefix_todo.append("genes_round")
        if metadata["barcode_rounds"] > 0:
            prefix_todo.append("barcode_round")
        prefix_todo.append("")

    subfolder = "averages"
    job_ids = []
    slurm_folder = Path.home() / "slurm_logs" / data_path / "averages"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    target_folder = get_processed_path(data_path) / subfolder
    for kind in prefix_todo:
        if kind:
            target_file = target_folder / "averages_average.tif"
        else:
            target_file = target_folder / "grand_average.tif"
        if (not force_redo) and target_file.exists():
            print(f"{kind} grand average already exists. Skipping")
            continue

        print(f"Creating grand average {kind}", flush=True)
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
                slurm_folder=slurm_folder,
                scripts_name=f"create_grand_average_{kind}",
                dependency_type="afterany",
                job_dependency=dependency if dependency else None,
            )
        )

    check_illumination_correction(
        data_path,
        grand_averages=prefix_todo[:-1],
        plot_tilestats=True,
        verbose=True,
        slurm_folder=slurm_folder,
        use_slurm=True,
        job_dependency=job_ids,
    )
    return job_ids


def overview_for_ara_registration(
    data_path,
    prefix,
    rois_to_do=None,
    sigma_blur=10,
    ref_prefix="genes_round",
    non_similar_overview=False,
):
    """Generate a stitched overview for registering to the ARA

    ABBA requires pyramidal OME-TIFF with resolution information. We will generate such
    stitched files and save them with a log yaml file indicating info about downsampling

    Args:
        data_path (str): Relative path to the data folder
        prefix (str): Acquisition to use for the overview e.g. `genes_round_1_1`
        rois_to_do (list, optional): ROIs to process. If None (default), process all
            ROIs
        sigma_blur (float, optional): sigma of the gaussian filter, in downsampled
            pixel size. Defaults to 10
        ref_prefix (str, optional): Prefix of the reference coordinates. Defaults to
            `genes_round`

    """
    processed_path = get_processed_path(data_path)
    registration_folder = processed_path / "register_to_ara"
    registration_folder.mkdir(exist_ok=True)
    # also make sure that the relevant subfolders are created
    (registration_folder / "qupath_project").mkdir(exist_ok=True)
    (registration_folder / "deepslice").mkdir(exist_ok=True)

    metadata = load_metadata(data_path)
    if rois_to_do is None:
        rois_to_do = metadata["ROI"].keys()
    roi_slice_pos_um, min_step = find_roi_position_on_cryostat(data_path=data_path)
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
            PREFIX=prefix,
            REF_PREFIX=ref_prefix,
            NON_SIMILAR_OVERVIEW=non_similar_overview,
        )
        args = "--export=" + ",".join([f"{k}={v}" for k, v in export_args.items()])
        slurm_folder = Path.home() / "slurm_logs" / data_path / "ara"
        slurm_folder.mkdir(parents=True, exist_ok=True)
        args = (
            args
            + f" --output={slurm_folder}iss_overview_roi_{roi}_%j.out"
            + f" --error={slurm_folder}iss_overview_roi_{roi}_%j.err"
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
    slurm_folder = Path.home() / "slurm_logs" / data_path
    if ops["barcode_rounds"] > 0:
        estimate_channel_correction(
            data_path,
            prefix="barcode_round",
            nrounds=ops["barcode_rounds"],
            fit_norm_factors=ops["fit_channel_correction"],
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="barcode_channel_correction",
        )
    if ops["genes_rounds"] > 0:
        estimate_channel_correction(
            data_path,
            prefix="genes_round",
            nrounds=ops["genes_rounds"],
            fit_norm_factors=ops["fit_channel_correction"],
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="genes_channel_correction",
        )

    estimate_channel_correction_hybridisation(
        data_path, use_slurm=use_slurm, slurm_folder=slurm_folder
    )


def call_spots(data_path, genes=True, barcodes=True, hybridisation=True):
    """Master method to run spot calling. Must be run after `iss estimate-shifts`,
    `iss estimate-hyb-shifts`, `iss setup-channel-correction`, and `iss
    create-grand-averages`.

    Args:
        data_path (str): Relative path to the data folder
        genes (bool, optional): Run genes spot calling. Defaults to True.
        barcodes (bool, optional): Run barcode calling. Defaults to True.
        hybridisation (bool, optional): Run hybridisation spot calling. Defaults to True

    """
    if genes:
        correct_shifts(data_path, prefix="genes_round")
        setup_omp(data_path)
        batch_process_tiles(data_path, "extract_tile")

    if barcodes:
        correct_shifts(data_path, prefix="barcode_round")
        setup_barcode_calling(data_path)
        batch_process_tiles(data_path, "basecall_tile")

    if hybridisation:
        correct_hyb_shifts(data_path)
        setup_hyb_spot_calling(data_path)
        extract_hyb_spots_all(data_path)


def segment_and_stitch_mcherry_cells(
    data_path, prefix, use_slurm=True, slurm_folder=None, job_dependency=None
):
    """Master function for mCherry cell segmentation and stitching

    Will call in turn the following functions:
    - `segment_mcherry_cells`
    - `filter_mcherry_cells` if ops['filter_mask'] is True
    - `register_within` to find overlapping region (with reload=True)
    - `remove_duplicate`
    - `stitch_mcherry_cells`

    Args:
        data_path (str): Relative path to the data folder
        prefix (str): Prefix of the mCherry acquisition
        use_slurm (bool, optional): Whether to use SLURM to run the jobs. Defaults to
            True.
        slurm_folder (str, optional): Folder to save SLURM logs. Defaults to None.
        job_dependency (list, optional): List of job IDs to wait for before starting the
    """

    ops = load_ops(data_path)
    if slurm_folder is None:
        slurm_folder = Path.home() / "slurm_logs" / data_path / f"segment_{prefix}"
    slurm_folder.mkdir(parents=True, exist_ok=True)

    job_coef = save_unmixing_coefficients(
        data_path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"unmix_{prefix}",
        job_dependency=job_dependency,
    )

    additional_args = f",PREFIX={prefix}"
    job_ids, failed_job = batch_process_tiles(
        data_path,
        script="segment_mcherry_tile",
        additional_args=additional_args,
        job_dependency=job_coef,
    )
    print(f"Started {len(job_ids)} jobs for segmenting mCherry cells")

    spref = prefix.split("_")[0].lower()
    # ensure the "within" registration ran
    reg_jobs = register_all_rois_within(
        data_path,
        prefix=prefix,
        ref_ch=None,
        suffix=ops[f"{spref}_projection"],
        correct_illumination=True,
        reload=True,
        save_plot=True,
        dimension_prefix="genes_round_1_1",
        verbose=True,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"register_within_{prefix}",
        job_dependency=failed_job,
    )
    # remove duplicate masks
    dupl_job = remove_all_duplicate_masks(
        data_path,
        prefix,
        use_slurm=True,
        slurm_folder=slurm_folder,
        job_dependency=reg_jobs,
        scripts_name=f"remove_duplicate_masks_{prefix}",
    )

    # stitch the masks dataframes
    stitch_job = stitch_cell_dataframes(
        data_path,
        prefix,
        use_slurm=True,
        slurm_folder=slurm_folder,
        job_dependency=dupl_job,
        scripts_name=f"stitch_{prefix}",
    )
    out = (
        reg_jobs,
        dupl_job,
        stitch_job,
    )
    if ops.get(f"{spref}_gmm_filter_masks", False):
        filt_job = filter_mcherry_cells(
            data_path,
            prefix,
            use_slurm=True,
            slurm_folder=slurm_folder,
            job_dependency=stitch_job,
        )
        out += (filt_job,)
    return out
