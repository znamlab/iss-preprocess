import click


@click.group()
def cli():
    pass


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
    from iss_preprocess.pipeline import run_omp_on_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    run_omp_on_tile(path, (roi, x, y), save_stack=save, correct_channels=True)


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
def project_round(path, prefix, overwrite=False):
    """Calculate Z-projection for all tiles in a single sequencing round."""
    from iss_preprocess.pipeline import project_round

    click.echo(f"Projecting ROI {prefix} from {path}")
    project_round(path, prefix, overwrite=overwrite)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
@click.option(
    "-s", "--suffix", default="fstack", help="Projection suffix, e.g. 'fstack'"
)
def register_tile(path, prefix, roi, tilex, tiley, suffix="fstack"):
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
    "-s", "--suffix", default="fstack", help="Projection suffix, e.g. 'fstack'"
)
def estimate_shifts(path, prefix, suffix="fstack"):
    """Estimate X-Y shifts across rounds and channels for all tiles."""
    from iss_preprocess.pipeline import estimate_shifts_all_tiles

    estimate_shifts_all_tiles(path, prefix, suffix)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def correct_shifts(path):
    """Correct X-Y shifts using robust regression across tiles."""
    from iss_preprocess.pipeline import correct_shifts

    correct_shifts(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def extract(path):
    """Correct X-Y shifts using robust regression across tiles."""
    from iss_preprocess.pipeline import run_omp_all_rois

    run_omp_all_rois(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
def register_ref_tile(path, prefix):
    """Correct X-Y shifts using robust regression across tiles."""
    from iss_preprocess.pipeline import register_reference_tile

    register_reference_tile(path, prefix=prefix)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", help="Path prefix to use for segmentation, e.g. 'DAPI_1"
)
@click.option("-r", "--roi", default=1, help="Number of the ROI to segment.")
def segment(path, prefix, roi=1):
    from iss_preprocess.pipeline import segment_roi

    segment_roi(path, roi, prefix)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", help="Path prefix to use for segmentation, e.g. 'DAPI_1"
)
def segment_all(path, prefix):
    from iss_preprocess.pipeline import segment_all_rois

    segment_all_rois(path, prefix)
