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
    from iss_preprocess.pipeline import project_and_average
    from pathlib import Path
    slurm_folder = Path.home() / "slurm_logs"
    click.echo(f"Processing {path}")
    project_and_average(
        path,
        force_redo=force_redo,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="project_and_average",
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
    "--diag",
    is_flag=True,
    show_default=True,
    default=True,
    help="Save diagnostic cross correlogram plots",
)
def register_ref_tile(path, prefix, diag=True):
    """Run registration across channels and rounds for the reference tile."""
    from iss_preprocess.pipeline import register_reference_tile

    register_reference_tile(path, prefix=prefix, diag=diag)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def setup_omp(path, use_slurm=True):
    """Estimate bleedthrough matrices and construct gene dictionary for OMP."""
    from iss_preprocess.pipeline import setup_omp
    from pathlib import Path

    slurm_folder = Path.home() / "slurm_logs"
    setup_omp(
        path, use_slurm=use_slurm, slurm_folder=slurm_folder, scripts_name="setup_omp"
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_barcodes(path):
    """Estimate bleedthrough matrices for barcode calling."""
    from iss_preprocess.pipeline import setup_barcode_calling

    setup_barcode_calling(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_hybridisation(path):
    """Estimate bleedthrough matrices for hybridisation spots."""
    from iss_preprocess.pipeline import setup_hyb_spot_calling

    setup_hyb_spot_calling(path)


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
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def register_hyb_tile(path, prefix, roi, tilex, tiley, suffix="max"):
    """Estimate X-Y shifts across rounds and channels for a single tile."""
    from iss_preprocess.pipeline import estimate_shifts_and_angles_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}/{prefix}")
    estimate_shifts_and_angles_by_coors(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, suffix=suffix
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
@click.option("-s", "--suffix", default="max", help="Projection suffix, e.g. 'max'")
def estimate_hyb_shifts(path, prefix=None, suffix="max"):
    """Estimate X-Y shifts across channels for a hybridisation round for all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles
    from iss_preprocess.io import load_metadata

    if prefix:
        additional_args = f",PREFIX={prefix},SUFFIX={suffix}"
        batch_process_tiles(
            path, script="register_hyb_tile", additional_args=additional_args
        )
    else:
        metadata = load_metadata(path)
        for hyb_round in metadata["hybridisation"].keys():
            additional_args = f",PREFIX={hyb_round},SUFFIX={suffix}"
            batch_process_tiles(
                path, script="register_hyb_tile", additional_args=additional_args
            )


@cli.command()
@click.option(
    "-p", "--path", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
def correct_shifts(path, prefix):
    """Correct X-Y shifts using robust regression across tiles."""
    from iss_preprocess.pipeline import correct_shifts

    correct_shifts(path, prefix)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Directory prefix to process.")
def correct_hyb_shifts(path, prefix=None):
    """
    Correct X-Y shifts for hybridisation rounds using robust regression
    across tiles.
    """
    from iss_preprocess.pipeline import correct_hyb_shifts

    correct_hyb_shifts(path, prefix)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Directory prefix to process.")
def correct_ref_shifts(path, prefix=None):
    """
    Correct X-Y shifts for registration to reference using robust regression
    across tiles.
    """
    from iss_preprocess.pipeline import correct_shifts_to_ref

    correct_shifts_to_ref(path, prefix)


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
def basecall(path):
    """Start batch jobs to run OMP on all tiles in a dataset."""
    from iss_preprocess.pipeline import batch_process_tiles

    batch_process_tiles(path, "basecall_tile")


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
    "--reg_prefix",
    default="barcode_round",
    help="Directory prefix to registration target.",
)
@click.option(
    "-f",
    "--ref_prefix",
    default="genes_round",
    help="Directory prefix to registration reference.",
)
@click.option("-r", "--roi", default=None, help="ROI number. None for all.")
@click.option("-x", "--tilex", default=None, help="Tile X position. None for all.")
@click.option("-y", "--tiley", default=None, help="Tile Y position. None for all.")
@click.option(
    "-c",
    "--reg_channels",
    default=None,
    help="Channels to register (comma separated string of integer).",
    type=str,
)
def register_to_reference(
    path, reg_prefix, ref_prefix, roi, tilex, tiley, reg_channels
):
    """Register an acquisition to reference tile by tile."""
    if any([x is None for x in [roi, tilex, tiley]]):
        print("Batch processing all tiles", flush=True)
        from iss_preprocess.pipeline import batch_process_tiles

        additional_args = f",REG_PREFIX={reg_prefix},REF_PREFIX={ref_prefix},REG_CHANNELS={reg_channels}"
        batch_process_tiles(
            path, "register_tile_to_ref", additional_args=additional_args
        )
    else:
        print(f"Registering ROI {roi}, Tile ({tilex}, {tiley})", flush=True)
        from iss_preprocess.pipeline import register

        if reg_channels is not None:
            reg_channels = [int(x) for x in reg_channels.split(",")]

        register.register_tile_to_ref(
            data_path=path,
            tile_coors=(roi, tilex, tiley),
            reg_prefix=reg_prefix,
            ref_prefix=ref_prefix,
            reg_channels=reg_channels,
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
    help="Directory prefix to registration target.",
)
@click.option(
    "-r",
    "--ref_prefix",
    default="genes_round_1_1",
    help="Directory prefix to registration reference.",
)
@click.option(
    "-l",
    "--reload",
    default=True,
    help="Whether to reload register_adjacent_tiles shifts.",
)
def align_spots(
    path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
    ref_prefix="genes_round_1_1",
    reload=True,
):
    from iss_preprocess.pipeline import (
        merge_and_align_spots_all_rois,
        register_within_acquisition,
    )

    register_within_acquisition(path, prefix=reg_prefix, reload=reload, save_plot=True)
    if reg_prefix != ref_prefix:
        register_within_acquisition(
            path, prefix=ref_prefix, reload=reload, save_plot=True
        )
    merge_and_align_spots_all_rois(
        path, spots_prefix=spots_prefix, reg_prefix=reg_prefix, ref_prefix=ref_prefix
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
@click.option(
    "-f",
    "--ref_prefix",
    default="genes_round_1_1",
    help="Directory prefix to use as a reference for registration.",
)
def align_spots_roi(
    path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
    roi=1,
    ref_prefix="genes_round_1_1",
):
    from iss_preprocess.pipeline import (
        merge_and_align_spots,
        stitch_and_register,
    )

    if ref_prefix != reg_prefix:
        stitch_and_register(
            path,
            reference_prefix=ref_prefix,
            target_prefix=reg_prefix,
            roi=roi,
            downsample=5,
            ref_ch=0,
            target_ch=0,
            estimate_scale=False,
        )

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
@click.option("--sigma", help="Sigma for gaussian blur")
def overview_for_ara_registration(path, roi, slice_id, sigma=10.0):
    """Generate the overview of one ROI used for registration

    Args:
        data_path (str): Relative path to data
        roi (int): ROI ID
        slice_id (int): Ordered slice id. Must increase across all chambers
        sigma_blur (float, optional): Sigma for gaussian blur. Defaults to 10.
    """
    from iss_preprocess.pipeline.ara_registration import overview_single_roi

    print("Calling")
    overview_single_roi(data_path=path, roi=roi, slice_id=slice_id, sigma_blur=sigma)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_flexilims(path):
    """Setup the flexilims database"""
    from iss_preprocess.pipeline import setup_flexilims

    setup_flexilims(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_channel_correction(path):
    """Setup channel correction for barcode, genes and hybridisation rounds"""
    from iss_preprocess.pipeline import setup_channel_correction

    setup_channel_correction(path)
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
@click.option(
    "--path", "-p", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
def plot_registration_correlograms(
    path,
    prefix,
):
    """Plot registration correlograms."""
    from iss_preprocess.vis import plot_registration_correlograms

    plot_registration_correlograms(data_path=path, prefix=prefix)
