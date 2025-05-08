import click


@click.group()
def iss_cli():
    pass


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--project/--no-project",
    default=False,
    help="Project the data. Slow.",
    show_default=True,
)
@click.option(
    "--nproc", default=4, help="Number of processes to use.", show_default=True
)
def live_crunch(path, project, nproc):
    """Project data during acquisition."""
    from iss_preprocess.pipeline.pipeline import sync_and_crunch

    sync_and_crunch(path, project=project, nproc=nproc)


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

    click.echo("Importing")
    from iss_preprocess.pipeline.pipeline import project_and_average

    time = str(datetime.now().strftime("%Y-%m-%d_%H-%M"))
    click.echo(f"Home: {Path.home()}")
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
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
@click.option(
    "--use-slurm/--no-use-slurm",
    default=True,
    help="Whether to use slurm for the main pipeline job "
    + "(subsequent steps always use slurm).",
)
@click.option(
    "-f",
    "--force-redo",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force redoing all steps.",
)
def register(path, prefix, use_slurm=True, force_redo=False):
    """Register an acquisition across round and channels."""
    from datetime import datetime
    from pathlib import Path

    from iss_preprocess.pipeline.pipeline import register_acquisition

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
        force_redo=force_redo,
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--genes/--no-genes",
    default=False,
    help="Process genes.",
    show_default=True,
)
@click.option(
    "--barcodes/--no-barcodes",
    default=False,
    help="Process barcode.",
    show_default=True,
)
@click.option(
    "--hyb/--no-hyb",
    default=False,
    show_default=True,
    help="Process hybridisation.",
)
@click.option(
    "-f",
    "--force-redo",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force redoing all steps.",
)
@click.option(
    "--setup-only/",
    is_flag=True,
    default=False,
    help="Only setup the channel correction, do NOT run on all tiles.",
    show_default=True,
)
def call(
    path,
    genes=False,
    barcodes=False,
    hybridisation=False,
    force_redo=False,
    setup_only=False,
):
    """Call spots for genes, barcodes and/or hybridisation rounds"""
    from ..pipeline.pipeline import call_spots

    txt = "Calling "
    if genes:
        txt += "genes "
    if barcodes:
        txt += "barcodes "
    if hybridisation:
        txt += "hybridisation "
    click.echo(txt)
    if setup_only:
        click.echo("Setup only")
    call_spots(
        data_path=path,
        genes=genes,
        barcodes=barcodes,
        hybridisation=hybridisation,
        force_redo=force_redo,
        setup_only=setup_only,
        use_slurm=True,
    )
    click.echo("Job submitted")


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
    from iss_preprocess.pipeline.pipeline import segment_all_rois

    segment_all_rois(path, prefix, use_gpu=use_gpu)


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
    from iss_preprocess.diagnostics.diag_stitching import plot_overview_images

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

    from iss_preprocess.pipeline.core import batch_process_tiles
    from iss_preprocess.pipeline.segment import (
        remove_all_duplicate_masks,
        save_unmixing_coefficients,
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
