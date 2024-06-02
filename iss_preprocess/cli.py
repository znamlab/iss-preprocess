import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-f",
    "--force-redo",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force redoing all steps.",
)
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
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


@cli.command()
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


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", prompt="Enter path prefix", help="Path prefile, e.g. round_01_1"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
@click.option(
    "--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to overwrite tiles if already projected.",
)
def project_tile(path, prefix, roi=1, x=0, y=0, overwrite=False):
    """Calculate Z-projection on a single tile."""
    from iss_preprocess.pipeline import project_tile_by_coors

    click.echo(f"Projecting ROI {roi}, {prefix}, tile {x}, {y} from {path}")
    project_tile_by_coors((roi, x, y), path, prefix, overwrite=overwrite)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", prompt="Enter path prefix", help="Path prefile, e.g. round_01_1"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-m", "--max-col", default=0, help="Maximum column index.")
@click.option(
    "--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to overwrite tiles if already projected.",
)
def project_row(path, prefix, roi=1, x=0, max_col=0, overwrite=False):
    """Calculate Z-projection for a single row of tiles"""
    from iss_preprocess.pipeline import project_tile_row

    click.echo(f"Projecting ROI {roi}, {prefix}, row {x}, from {path}")
    project_tile_row(path, prefix, roi, x, max_col, overwrite=overwrite)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", prompt="Enter path prefix", help="Path prefile, e.g. round_01_1"
)
@click.option(
    "--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to overwrite tiles if already projected.",
)
@click.option(
    "--overview",
    is_flag=True,
    show_default=True,
    default=True,
    help="Whether to create overview images after projection.",
)
def project_round(path, prefix, overwrite=False, overview=True):
    """Calculate Z-projection for all tiles in a single sequencing round."""
    from iss_preprocess.pipeline import project_round

    click.echo(f"Projecting ROI {prefix} from {path}")
    project_round(path, prefix, overwrite=overwrite, overview=overview)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Acq prefix, e.g. `genes_round_1_1`, None for all")
def check_projection(path, prefix):
    """Check if projection has completed for all tile."""
    import iss_preprocess.pipeline as pipeline

    pipeline.check_projection(path, prefix)


@cli.command()
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


@cli.command()
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


@cli.command()
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


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def register_tile(path, prefix, roi, tilex, tiley, suffix="max"):
    """Estimate X-Y shifts across rounds and channels for a single tile."""
    from iss_preprocess.pipeline import estimate_shifts_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}")
    estimate_shifts_by_coors(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, suffix=suffix
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def register_hyb_tile(path, prefix, roi, tilex, tiley):
    """Estimate X-Y shifts across rounds and channels for a single tile."""
    from iss_preprocess.pipeline import register_fluorescent_tile

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}/{prefix}")
    register_fluorescent_tile(
        path,
        tile_coors=(roi, tilex, tiley),
        prefix=prefix,
        reference_prefix=None,
    )


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Path prefix, e.g. 'genes_round'")
def estimate_hyb_shifts(path, prefix=None):
    """Estimate X-Y shifts across channels for a hybridisation round for all tiles."""
    from iss_preprocess.io import load_metadata
    from iss_preprocess.pipeline import batch_process_tiles

    if prefix:
        additional_args = f",PREFIX={prefix}"
        batch_process_tiles(
            path, script="register_hyb_tile", additional_args=additional_args
        )
    else:
        metadata = load_metadata(path)
        for hyb_round in metadata["hybridisation"].keys():
            additional_args = f",PREFIX={hyb_round}"
            batch_process_tiles(
                path, script="register_hyb_tile", additional_args=additional_args
            )


@cli.command()
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


@cli.command()
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
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            job_dependency=job_id if use_slurm else None,
            scripts_name=f"check_shift_correction_{prefix}",
            within=False,
        )
        if use_slurm:
            print(f"Started 2 jobs: {job_id}, {job2}")


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="genes_round", help="Path prefix, e.g. 'genes_round'"
)
def spot_sign_image(path, prefix="genes_round"):
    """Compute average spot image."""
    from iss_preprocess.pipeline import compute_spot_sign_image

    compute_spot_sign_image(path, prefix)


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def basecall(path):
    """Start batch jobs to run basecalling for barcodes on all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles

    job_ids = batch_process_tiles(path, "basecall_tile")
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


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def extract(path):
    """Start batch jobs to run OMP on all tiles in a dataset."""
    from iss_preprocess.pipeline import batch_process_tiles

    batch_process_tiles(path, "extract_tile")


@cli.command()
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


@cli.command()
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


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-s",
    "--spots-prefix",
    default="barcode_round",
    help="File name prefix for spot files.",
)
@click.option(
    "-l",
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
        register_within_acquisition,
    )
    from iss_preprocess.io import load_ops

    slurm_folder = Path.home() / "slurm_logs" / path / "align_spots"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    ref_job_id = None
    ops = load_ops(path)
    ref_prefix = ops["reference_prefix"]

    ref_job_id = register_within_acquisition(
        path,
        prefix=ref_prefix,
        reload=reload,
        save_plot=True,
        use_slurm=True,
        slurm_folder=slurm_folder,
        scripts_name=f"register_within_acquisition_{ref_prefix}",
    )

    merge_and_align_spots_all_rois(
        path,
        spots_prefix=spots_prefix,
        ref_prefix=ref_prefix,
        dependency=ref_job_id,
    )


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def hyb_spots(path):
    """Detect hybridisation in all ROIs / hybridisation rounds"""
    from iss_preprocess.pipeline import extract_hyb_spots_all

    extract_hyb_spots_all(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=1, help="Number of the ROI to segment.")
@click.option("-n", "--prefix", help="Path prefix for spot detection")
def hyb_spots_roi(path, prefix, roi=1):
    """Detect hybridisation spots in a single ROI / hybridisation round"""
    from iss_preprocess.pipeline import extract_hyb_spots_roi

    extract_hyb_spots_roi(path, prefix, roi)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
# TODO: expose prefix_to_do
def create_grand_averages(path):
    """Create grand average for illumination correction"""
    from iss_preprocess import pipeline

    pipeline.create_grand_averages(path, prefix_todo=("genes_round", "barcode_round"))


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--n-batch",
    help="Number of average batches to compute before taking their median.",
    default=1,
)
def create_all_single_averages(path, n_batch):
    """Average all tiffs in all acquisition folders"""
    from iss_preprocess import pipeline

    pipeline.create_all_single_averages(path, n_batch=n_batch)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-b", "--subtract-black/--no-subtract-black", help="Subtract black level")
@click.option(
    "-s", "--subfolder", help="Subfolder containing tifs to average", default=""
)
@click.option(
    "--prefix_filter",
    help="Filter to average only subset of tifs of the folder",
    type=str,
    default=None,
)
@click.option(
    "--suffix",
    help="Filter to average only subset of tifs of the folder",
    type=str,
    default=None,
)
@click.option(
    "--combine-stats/--no-combine-stats",
    help="Combine pre-existing statistics into one instead of computing from images",
    default=False,
)
@click.option(
    "--n-batch",
    help="Number of average batches to compute before taking their median.",
    default=1,
)
def create_single_average(
    path, subtract_black, subfolder, prefix_filter, suffix, combine_stats, n_batch
):
    """Average all tiffs in an acquisition folder"""
    from iss_preprocess import pipeline

    pipeline.create_single_average(
        path,
        subfolder=subfolder,
        subtract_black=subtract_black,
        prefix_filter=prefix_filter,
        suffix=suffix,
        combine_tilestats=combine_stats,
        n_batch=n_batch,
    )


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_flexilims(path):
    """Setup the flexilims database"""
    from iss_preprocess.pipeline import setup_flexilims

    setup_flexilims(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--use-slurm/--local", is_flag=True, default=True, help="Whether to use slurm"
)
def setup_channel_correction(path, use_slurm=True):
    """Setup channel correction for barcode, genes and hybridisation rounds"""

    from iss_preprocess.pipeline import setup_channel_correction as scc

    scc(path, use_slurm=use_slurm)
    click.echo("Channel correction setup complete.")


@cli.command()
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


@cli.command()
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


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
@click.option(
    "-b", "--background_ch", default=3, help="Channel containing background, e.g. 3"
)
@click.option("-g", "--signal_ch", default=2, help="Channel containing signal, e.g. 2")
def unmix_channels(
    path, prefix="mCherry_1", suffix="max", background_ch=3, signal_ch=2
):
    """Unmix autofluorescence from signal for all tiles in a dataset."""
    from iss_preprocess.pipeline import batch_process_tiles
    from iss_preprocess.image import unmix_ref_tile
    from iss_preprocess.io.load import load_ops
    from iss_preprocess.pipeline import load_and_register_tile

    ops = load_ops(path)
    (roi, tilex, tiley) = ops["mcherry_ref_tile"]
    stack, _ = load_and_register_tile(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, filter_r=False
    )
    print(f"Unmixing autofluorescence from reference tile {roi}, {tilex}, {tiley}")
    _, coef, intercept = unmix_ref_tile(
        path,
        prefix,
        roi,
        tilex,
        tiley,
        stack,
        suffix=suffix,
        background_ch=background_ch,
        signal_ch=signal_ch,
    )

    additional_args = f",PREFIX={prefix},SUFFIX={suffix},BACKGROUND_CH={background_ch},SIGNAL_CH={signal_ch},COEF={coef},INTERCEPT={intercept}"
    batch_process_tiles(path, script="unmix_channels", additional_args=additional_args)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
@click.option(
    "-b", "--background_ch", default=3, help="Channel containing background, e.g. 3"
)
@click.option("-g", "--signal_ch", default=2, help="Channel containing signal, e.g. 2")
@click.option("-c", "--coef", help="Coefficient for linear unmixing")
@click.option("-i", "--intercept", help="Intercept for linear unmixing")
def unmix_tile(
    path,
    prefix,
    roi,
    tilex,
    tiley,
    suffix="max",
    background_ch=3,
    signal_ch=2,
    coef=None,
    intercept=None,
):
    """Unmix autofluorescence from signal for all tiles in a dataset."""
    from iss_preprocess.image import unmix_tile
    from iss_preprocess.pipeline import load_and_register_tile

    stack, _ = load_and_register_tile(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, filter_r=False
    )
    unmix_tile(
        path,
        prefix,
        roi,
        tilex,
        tiley,
        stack,
        suffix=suffix,
        background_ch=background_ch,
        signal_ch=signal_ch,
        coef=coef,
        intercept=intercept,
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def segment_all_mcherry(path, prefix="mCherry_1", suffix="max"):
    """Segment mcherry cells for all tiles in a dataset."""
    from iss_preprocess.pipeline import batch_process_tiles

    additional_args = f",PREFIX={prefix},SUFFIX={suffix}"
    batch_process_tiles(
        path, script="segment_mcherry_tile", additional_args=additional_args
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def segment_mcherry_tile(path, prefix, roi, tilex, tiley, suffix="max"):
    """Segment mCherry channel for a single tile."""
    from iss_preprocess.pipeline import segment_mcherry_tile

    segment_mcherry_tile(
        path,
        prefix,
        roi,
        tilex,
        tiley,
        suffix,
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def remove_non_cell_masks(path, roi, tilex, tiley):
    """Remove masks from mCherry tiles that don't correspond to cells."""
    from iss_preprocess.pipeline import remove_non_cell_masks

    remove_non_cell_masks(
        path,
        roi,
        tilex,
        tiley,
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def find_mcherry_cells(path):
    """Find mCherry cells using a GMM to cluster masks based on their
    morphological features. Then remove non-cell masks from each tile.
    """
    from iss_preprocess.pipeline import find_mcherry_cells

    find_mcherry_cells(path)


@cli.command()
@click.option("-j", "--jobsinfo", help="Job ids and args file.")
def handle_failed(job_info_path):
    from iss_preprocess.pipeline import handle_failed_jobs

    handle_failed_jobs(job_info_path)
