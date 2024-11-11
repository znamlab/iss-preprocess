import click


@click.group()
def iss_cli():
    pass


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-f",
    "--force-redo",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force redoing all steps.",
)
@click.option(
    "--use-slurm/--no-use-slurm",
    default=True,
    help="Whether to use slurm for the main pipeline job "
    + "(subsequent steps always use slurm).",
)
def project_and_average(path, force_redo=False, use_slurm=True):
    """Project and average all available data then create plots."""
    from datetime import datetime
    from pathlib import Path

    from iss_preprocess.pipeline import project_and_average

    time = str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    click.echo(f"Processing {path}")
    project_and_average(
        path,
        force_redo=force_redo,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"project_and_average_{time}",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "--use-slurm/--no-use-slurm",
    default=True,
    help="Whether to use slurm for the main pipeline job "
    + "(subsequent steps always use slurm).",
)
def register_acquisition(path, prefix, use_slurm=True):
    """Register an acquisition across round and channels."""
    from datetime import datetime
    from pathlib import Path
    from iss_preprocess.pipeline import register_acquisition

    time = str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    slurm_folder = Path.home() / "slurm_logs" / path
    if use_slurm:
        slurm_folder.mkdir(parents=True, exist_ok=True)
    click.echo(f"Processing {path}")

    register_acquisition(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"register_acquisition_{time}",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
@click.option(
    "--save",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to save registered tile images.",
)
def extract_tile(path, roi=1, x=0, y=0, save=False):
    """Run OMP and a single tile and detect gene spots."""
    from iss_preprocess.pipeline import detect_genes_on_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    detect_genes_on_tile(path, (roi, x, y), save_stack=save)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
def basecall_tile(path, roi=1, x=0, y=0):
    """Run basecalling for barcodes on a single tile."""
    from iss_preprocess.pipeline import basecall_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    basecall_tile(path, (roi, x, y))


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "--diag/--no-diag",
    show_default=True,
    default=False,
    help="Save diagnostic cross correlogram plots",
)
def register_ref_tile(path, prefix, diag):
    """Run registration across channels and rounds for the reference tile."""
    from pathlib import Path

    from iss_preprocess.pipeline import register_reference_tile
    from iss_preprocess.pipeline.diagnostics import check_ref_tile_registration

    slurm_folder = Path.home() / "slurm_logs" / path
    scripts_name = f"register_ref_tile_{prefix}"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    slurm_options = {"mem": "128G"} if diag else None
    job_id = register_reference_tile(
        path,
        prefix=prefix,
        diag=diag,
        use_slurm=True,
        slurm_folder=str(slurm_folder),
        slurm_options=slurm_options,
        scripts_name=scripts_name,
    )
    scripts_name = f"check_ref_tile_registration_{prefix}"
    job2 = check_ref_tile_registration(
        path,
        prefix,
        use_slurm=True,
        slurm_folder=str(slurm_folder),
        job_dependency=job_id,
        scripts_name=scripts_name,
    )
    print(f"Started 2 jobs: {job_id}, {job2}")


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
def setup_omp(path, use_slurm=True):
    """Estimate bleedthrough matrices and construct gene dictionary for OMP."""
    from pathlib import Path

    from iss_preprocess.pipeline import setup_omp

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_omp(
        path, use_slurm=use_slurm, slurm_folder=slurm_folder, scripts_name="setup_omp"
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
def setup_barcodes(path, use_slurm=True):
    """Estimate bleedthrough matrices for barcode calling."""
    from pathlib import Path

    from iss_preprocess.pipeline import setup_barcode_calling

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_barcode_calling(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_barcodes",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--prefix",
    default=None,
    help="Path prefix, e.g. 'hybridisation_round'. If None,"
    + " all hybridisation rounds are processed.",
)
@click.option(
    "--use-slurm/--local", is_flag=True, default=True, help="Whether to use slurm"
)
def setup_hybridisation(path, prefix=None, use_slurm=True):
    """Estimate bleedthrough matrices for hybridisation spots."""
    from iss_preprocess.pipeline import setup_hyb_spot_calling

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    setup_hyb_spot_calling(path, prefix, use_slurm=use_slurm, slurm_folder=slurm_folder)


@iss_cli.command()
@click.option(
    "-p", "--path", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def estimate_shifts(path, prefix, suffix="max"):
    """Estimate X-Y shifts across rounds and channels for all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles

    additional_args = f",PREFIX={prefix},SUFFIX={suffix}"
    batch_process_tiles(path, script="register_tile", additional_args=additional_args)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Path prefix, e.g. 'genes_round'")
def estimate_hyb_shifts(path, prefix=None):
    """Estimate X-Y shifts across channels for a hybridisation round for all tiles."""
    from iss_preprocess.io import load_metadata
    from iss_preprocess.io import get_roi_dimensions
    from iss_preprocess.pipeline import batch_process_tiles

    if prefix:
        roi_dims = get_roi_dimensions(path, prefix)
        additional_args = f",PREFIX={prefix}"
        batch_process_tiles(
            path,
            script="register_hyb_tile",
            roi_dims=roi_dims,
            additional_args=additional_args,
        )
    else:
        metadata = load_metadata(path)
        for hyb_round in metadata["hybridisation"].keys():
            additional_args = f",PREFIX={hyb_round}"
            batch_process_tiles(
                path, script="register_hyb_tile", additional_args=additional_args
            )


@iss_cli.command()
@click.option(
    "-p", "--path", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
def correct_shifts(path, prefix, use_slurm=False):
    """Correct X-Y shifts using robust regression across tiles."""
    # import with different name to not get confused with the cli function name
    from iss_preprocess.pipeline import correct_shifts as corr_shifts
    from iss_preprocess.pipeline import diagnostics as diag

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    job_id = corr_shifts(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"correct_shifts_{prefix}",
    )
    diag.check_shift_correction(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_shift_correction_{prefix}",
    )
    diag.check_tile_registration(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_tile_registration_{prefix}",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Directory prefix to process.")
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
def correct_hyb_shifts(path, prefix=None, use_slurm=False):
    """
    Correct X-Y shifts for hybridisation rounds using robust regression
    across tiles.
    """
    from iss_preprocess.pipeline import correct_hyb_shifts
    from iss_preprocess.pipeline import diagnostics as diag

    if prefix is None:
        from iss_preprocess.io import load_metadata

        metadata = load_metadata(path)
        prefixes = metadata["hybridisation"].keys()
    else:
        prefixes = [prefix]
    for prefix in prefixes:
        print(f"Correcting shifts for {prefix}")
        if use_slurm:
            from pathlib import Path

            slurm_folder = Path.home() / "slurm_logs" / path
            slurm_folder.mkdir(parents=True, exist_ok=True)
        else:
            slurm_folder = None

        job_id = correct_hyb_shifts(
            path,
            prefix,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"correct_hyb_shifts_{prefix}",
        )
        job2 = diag.check_shift_correction(
            path,
            prefix,
            roi_dimension_prefix=prefix,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=job_id if use_slurm else None,
            scripts_name=f"check_shift_correction_{prefix}",
            within=False,
        )
        if use_slurm:
            print(f"Started 2 jobs: {job_id}, {job2}")


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Directory prefix to process.")
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
def correct_ref_shifts(path, prefix=None, use_slurm=False):
    """
    Correct X-Y shifts for registration to reference using robust regression
    across tiles.
    """
    from iss_preprocess.pipeline import correct_shifts_to_ref
    from iss_preprocess.pipeline import diagnostics as diag

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None

    job_id = correct_shifts_to_ref(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"correct_shifts_to_ref_{prefix}",
    )
    diag.check_reg_to_ref_correction(
        path,
        prefix,
        rois=None,
        roi_dimension_prefix="genes_round_1_1",
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_reg_to_ref_correction_{prefix}",
    )
    diag.check_registration_to_reference(
        path,
        prefix=prefix,
        ref_prefix=None,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_tile_reg_to_ref_{prefix}",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="genes_round", help="Path prefix, e.g. 'genes_round'"
)
def spot_sign_image(path, prefix="genes_round"):
    """Compute average spot image."""
    from iss_preprocess.pipeline import compute_spot_sign_image

    compute_spot_sign_image(path, prefix)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def check_omp(path, roi, tilex, tiley, use_slurm=True):
    """Compute average spot image."""
    from iss_preprocess.pipeline import check_omp_thresholds

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path / "check_omp"
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    if roi is not None and tilex is not None and tiley is not None:
        check_omp_thresholds(
            path,
            tile_coors=(roi, tilex, tiley),
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"check_omp",
        )
    else:
        check_omp_thresholds(
            path,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"check_omp",
        )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def check_omp_alpha(path, roi, tilex, tiley, use_slurm=True):
    """Compute average spot image."""
    from iss_preprocess.pipeline import check_omp_alpha_thresholds

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path / "check_omp"
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    if roi is not None and tilex is not None and tiley is not None:
        check_omp_alpha_thresholds(
            path,
            tile_coors=(roi, tilex, tiley),
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"check_omp",
        )
    else:
        check_omp_alpha_thresholds(
            path,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"check_omp",
        )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def basecall(path):
    """Start batch jobs to run basecalling for barcodes on all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles

    job_ids, failed_job = batch_process_tiles(path, "basecall_tile")
    click.echo(f"Basecalling started for {len(job_ids)} tiles.")
    click.echo(f"Last job id: {job_ids[-1]}")

    from pathlib import Path

    from iss_preprocess.pipeline.diagnostics import check_barcode_basecall

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    check_barcode_basecall(
        path,
        use_slurm=True,
        job_dependency=job_ids,
        slurm_folder=slurm_folder,
        scripts_name=f"check_basecall",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
@click.option("--ref-tile-index", default=0, help="Reference tile index")
def check_basecall(path, use_slurm=False, ref_tile_index=0):
    """Check if basecalling has completed for all tiles."""
    from iss_preprocess.pipeline.diagnostics import check_barcode_basecall

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    check_barcode_basecall(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        ref_tile_index=ref_tile_index,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def extract(path):
    """Start batch jobs to run OMP on all tiles in a dataset."""
    from iss_preprocess.pipeline import batch_process_tiles

    batch_process_tiles(path, "extract_tile")


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--prefix",
    default="DAPI_1",
    help="Path prefix to use for segmentation, e.g. 'DAPI_1",
)
@click.option("-r", "--roi", default=1, help="Number of the ROI to segment.")
@click.option(
    "--use-gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to use the GPU",
)
def segment(path, prefix, roi=1, use_gpu=False):
    from iss_preprocess.pipeline import segment_roi

    segment_roi(path, roi, prefix, use_gpu=use_gpu)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--prefix",
    default="DAPI_1",
    help="Path prefix to use for segmentation, e.g. 'DAPI_1",
)
@click.option(
    "--use-gpu",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to use the GPU",
)
def segment_all(path, prefix, use_gpu=False):
    from iss_preprocess.pipeline import segment_all_rois

    segment_all_rois(path, prefix, use_gpu=use_gpu)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--reg-prefix",
    default="barcode_round",
    help="Directory prefix to registration target.",
)
@click.option("-r", "--roi", default=None, help="ROI number. None for all.")
@click.option("-x", "--tilex", default=None, help="Tile X position. None for all.")
@click.option("-y", "--tiley", default=None, help="Tile Y position. None for all.")
@click.option(
    "-m/",
    "--use-masked-correlation/--no-use-masked-correlation",
    default=False,
    help="Whether to use masked correlation.",
)
@click.option(
    "--use-stitched",
    is_flag=True,
    help="Use stitched images for registration instead of running it tile by tile.",
)
def register_to_reference(
    path,
    reg_prefix,
    roi,
    tilex,
    tiley,
    use_masked_correlation,
    use_stitched,
):
    """Register an acquisition to reference tile by tile."""
    from iss_preprocess.pipeline import register

    if use_stitched:
        from pathlib import Path

        if not use_masked_correlation:
            print("!!! Stitched registration might fail without masked correlation !!!")
        if roi is None:
            from iss_preprocess.io import get_roi_dimensions

            roi_dims = get_roi_dimensions(path)
            rois = roi_dims[:, 0]
        elif isinstance(roi, int) or isinstance(roi, str):
            rois = [int(roi)]

        slurm_folder = Path.home() / "slurm_logs" / path
        for roi in rois:
            script_name = f"register_to_ref_{reg_prefix}_roi_{roi}"
            register.register_to_ref_using_stitched_registration(
                data_path=path,
                roi=roi,
                reg_prefix=reg_prefix,
                use_masked_correlation=use_masked_correlation,
                use_slurm=True,
                slurm_folder=slurm_folder,
                scripts_name=script_name,
                save_plot=True,
            )
        return
    if any([x is None for x in [roi, tilex, tiley]]):
        register.register_all_tiles_to_ref(path, reg_prefix, use_masked_correlation)
    else:
        print(f"Registering ROI {roi}, Tile ({tilex}, {tiley})", flush=True)
        register.register_tile_to_ref(
            data_path=path,
            reg_prefix=reg_prefix,
            tile_coors=(int(roi), int(tilex), int(tiley)),
            use_masked_correlation=use_masked_correlation,
        )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-s",
    "--spots-prefix",
    default="barcode_round",
    help="File name prefix for spot files.",
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Whether to reload register_adjacent_tiles shifts.",
)
def align_spots(
    path,
    spots_prefix="barcode_round",
    reload=True,
):
    from pathlib import Path

    from iss_preprocess.pipeline import (
        merge_and_align_spots_all_rois,
        register_all_rois_within,
    )
    from iss_preprocess.io import load_ops

    slurm_folder = Path.home() / "slurm_logs" / path / "align_spots"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    ref_job_id = None
    ops = load_ops(path)
    ref_prefix = ops["reference_prefix"]

    ref_job_id = register_all_rois_within(
        path,
        prefix=ref_prefix,
        reload=reload,
        save_plot=True,
        use_slurm=True,
    )

    merge_and_align_spots_all_rois(
        path,
        spots_prefix=spots_prefix,
        ref_prefix=ref_prefix,
        dependency=ref_job_id,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-s",
    "--spots-prefix",
    default="barcode_round",
    help="File name prefix for spot files.",
)
@click.option(
    "-g",
    "--reg_prefix",
    default="barcode_round_1_1",
    help="Directory prefix to registration.",
)
@click.option("-r", "--roi", default=1, help="Number of the ROI to segment.")
def align_spots_roi(
    path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
    roi=1,
):
    from iss_preprocess.pipeline import merge_and_align_spots

    merge_and_align_spots(
        path, spots_prefix=spots_prefix, reg_prefix=reg_prefix, roi=roi
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def hyb_spots(path):
    """Detect hybridisation in all ROIs / hybridisation rounds"""
    from iss_preprocess.pipeline import extract_hyb_spots_all

    extract_hyb_spots_all(path)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix for spot detection")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
def extract_hyb_spots(path, prefix, roi, tilex, tiley):
    """Detect hybridisation spots in a single ROI / hybridisation round"""

    if tilex is not None and tiley is not None:
        from iss_preprocess.pipeline.hybridisation import extract_hyb_spots_tile

        tile_coors = (roi, tilex, tiley)
        extract_hyb_spots_tile(path, tile_coors, prefix)
    else:
        from iss_preprocess.pipeline.hybridisation import extract_hyb_spots_roi

        extract_hyb_spots_roi(path, prefix, roi)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", help="Roi id", type=int)
@click.option("-s", "--slice_id", help="ID for ordering ROIs", type=int)
@click.option(
    "-n",
    "--prefix",
    help="Path prefix, e.g. 'genes_round_1_1'",
    default="genes_round_1_1",
    show_default=True,
)
@click.option("--sigma", help="Sigma for gaussian blur")
@click.option(
    "--ref_prefix",
    help="Path prefix for reference, e.g. 'genes_round'",
    default="genes_round",
    show_default=True,
)
def overview_for_ara_registration(
    path,
    roi,
    slice_id,
    prefix,
    sigma=10.0,
    ref_prefix="genes_round",
    non_similar_overview=False,
):
    """Generate the overview of one ROI used for registration

    Args:
        data_path (str): Relative path to data
        roi (int): ROI ID
        slice_id (int): Ordered slice id. Must increase across all chambers
        sigma_blur (float, optional): Sigma for gaussian blur. Defaults to 10.
    """
    from iss_preprocess.pipeline.ara_registration import overview_single_roi

    print("Calling")
    overview_single_roi(
        data_path=path,
        roi=roi,
        slice_id=slice_id,
        sigma_blur=sigma,
        prefix=prefix,
        ref_prefix=ref_prefix,
        non_similar_overview=non_similar_overview,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--use-slurm/--local", is_flag=True, default=True, help="Whether to use slurm"
)
def setup_channel_correction(path, use_slurm=True):
    """Setup channel correction for barcode, genes and hybridisation rounds"""

    from iss_preprocess.pipeline import setup_channel_correction as scc

    scc(path, use_slurm=use_slurm)
    click.echo("Channel correction setup complete.")


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--genes", is_flag=True, help="Whether to call spots for genes.")
@click.option("--barcodes", is_flag=True, help="Whether to call spots for barcodes.")
@click.option(
    "--hybridisation", is_flag=True, help="Whether to call spots for hybridisation."
)
def call_spots(path, genes, barcodes, hybridisation):
    """Call spots for genes, barcodes and hybridisation rounds"""
    from iss_preprocess.pipeline import call_spots

    called = []
    for spot_type in ["genes", "barcodes", "hybridisation"]:
        if locals()[spot_type]:
            called.append(spot_type)
    if not called:
        print("No spots to call.")
        return
    print(f"Calling spots for {', '.join(called)}")

    call_spots(path, genes, barcodes, hybridisation)


@iss_cli.command()
@click.option(
    "--path", "-p", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
@click.option(
    "--plot-grid/--no-plot-gird",
    "-g",
    help="Whether to plot grid",
    default=True,
    show_default=True,
)
@click.option(
    "--downsample_factor",
    "-d",
    help="Amount to downsample output",
    type=int,
    default=25,
)
@click.option(
    "--save-raw/--no-save-raw",
    "-r",
    help="Whether to save full size tif",
    default=False,
)
@click.option(
    "--separate-channels/--no-separate-channels",
    "-s",
    help="Whether to save a figure per channel",
    default=False,
)
def plot_overview(
    path, prefix, plot_grid, downsample_factor, save_raw, separate_channels
):
    """Plot individual channel overview images."""
    from iss_preprocess.vis import plot_overview_images

    plot_overview_images(
        data_path=path,
        prefix=prefix,
        plot_grid=plot_grid,
        downsample_factor=downsample_factor,
        save_raw=save_raw,
        group_channels=not separate_channels,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
def segment_all_mcherry(path, prefix="mCherry_1"):
    """Segment mcherry cells for all tiles in a dataset."""
    from pathlib import Path
    from iss_preprocess.pipeline import batch_process_tiles
    from iss_preprocess.pipeline.segment import (
        save_unmixing_coefficients,
        remove_all_duplicate_masks,
    )

    save_unmixing_coefficients(path, prefix, projection=None)
    additional_args = f",PREFIX={prefix}"
    job_ids, failed_job = batch_process_tiles(
        path, script="segment_mcherry_tile", additional_args=additional_args
    )
    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    remove_all_duplicate_masks(
        path,
        prefix,
        use_slurm=True,
        slurm_folder=slurm_folder,
        job_dependency=failed_job,
        scripts_name=f"remove_duplicate_masks_{prefix}",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def segment_mcherry_tile(path, prefix, roi, tilex, tiley):
    """Segment mCherry channel for a single tile."""
    from iss_preprocess.pipeline import segment_mcherry_tile

    # TODO: move out of main CLI
    segment_mcherry_tile(
        path,
        prefix,
        roi,
        tilex,
        tiley,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def remove_non_cell_masks(path, prefix, roi, tilex, tiley):
    """Remove masks from mCherry tiles that don't correspond to cells."""
    from iss_preprocess.pipeline.segment import _remove_non_cell_masks

    # TODO: move out of main CLI
    _remove_non_cell_masks(
        path,
        prefix,
        roi,
        tilex,
        tiley,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
def filter_mcherry_masks(path, prefix):
    """Find mCherry cells using a GMM to cluster masks based on their
    morphological features. Then remove non-cell masks from each tile.
    """
    from iss_preprocess.pipeline import _gmm_cluster_mcherry_cells

    # TODO: move out of main CLI
    _gmm_cluster_mcherry_cells(path, prefix)
