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

    from iss_preprocess.pipeline.pipeline import project_and_average

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
@click.option("--genes/--no-genes", default=True, help="Process genes.")
@click.option("--barcodes/--no-barcodes", default=True, help="Process barcode.")
@click.option(
    "--hybridisation/--no-hybridisation",
    default=True,
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
def call(path, genes=True, barcodes=True, hybridisation=True, force_redo=False):
    from ..pipeline.pipeline import call_spots

    call_spots(
        data_path=path,
        genes=genes,
        barcodes=barcodes,
        hybridisation=hybridisation,
        force_redo=force_redo,
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
    from ..diagnostics.diag_register import check_shift_correction
    from ..pipeline import correct_hyb_shifts

    if prefix is None:
        from ..io import load_metadata

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
        job2 = check_shift_correction(
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
    from ..diagnostics.diag_reg2ref import (
        check_reg_to_ref_correction,
        check_registration_to_reference,
    )
    from ..pipeline import correct_shifts_to_ref

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
    check_reg_to_ref_correction(
        path,
        prefix,
        rois=None,
        roi_dimension_prefix="genes_round_1_1",
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_reg_to_ref_correction_{prefix}",
    )
    check_registration_to_reference(
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
    from iss_preprocess.pipeline.pipeline import segment_roi

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
    from iss_preprocess.pipeline.pipeline import segment_all_rois

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
    from iss_preprocess.pipeline.reg2ref import (
        register_all_tiles_to_ref,
        register_tile_to_ref,
    )

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
        register_all_tiles_to_ref(path, reg_prefix, use_masked_correlation)
    else:
        print(f"Registering ROI {roi}, Tile ({tilex}, {tiley})", flush=True)
        register_tile_to_ref(
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

    from iss_preprocess.io import load_ops
    from iss_preprocess.pipeline.align_spots_and_cells import (
        merge_and_align_spots_all_rois,
    )
    from iss_preprocess.pipeline.stitch import register_all_rois_within

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
    from iss_preprocess.pipeline.align_spots_and_cells import merge_and_align_spots

    merge_and_align_spots(
        path, spots_prefix=spots_prefix, reg_prefix=reg_prefix, roi=roi
    )


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

    from iss_preprocess.pipeline.pipeline import setup_channel_correction as scc

    scc(path, use_slurm=use_slurm)
    click.echo("Channel correction setup complete.")


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
    from iss_preprocess.pipeline.pipeline import segment_mcherry_tile

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
    from iss_preprocess.pipeline.pipeline import _gmm_cluster_mcherry_cells

    # TODO: move out of main CLI
    _gmm_cluster_mcherry_cells(path, prefix)
