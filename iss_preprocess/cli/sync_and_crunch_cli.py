"""Sync and crunch CLI.

Initial functions that reduce datasize and can be called first once data is acquired.
"""

import click


@click.group()
def sync_and_crunch_cli():
    pass


@sync_and_crunch_cli.command()
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
    from iss_preprocess.pipeline.project import project_tile_by_coors

    click.echo(f"Projecting ROI {roi}, {prefix}, tile {x}, {y} from {path}")
    project_tile_by_coors((roi, x, y), path, prefix, overwrite=overwrite)


@sync_and_crunch_cli.command()
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
    from iss_preprocess.pipeline.project import project_tile_row

    click.echo(f"Projecting ROI {roi}, {prefix}, row {x}, from {path}")
    project_tile_row(path, prefix, roi, x, max_col, overwrite=overwrite)


@sync_and_crunch_cli.command()
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
    from iss_preprocess.pipeline.project import project_round as proj_round_func

    click.echo(f"Projecting ROI {prefix} from {path}")
    proj_round_func(path, prefix, overwrite=overwrite)


@sync_and_crunch_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Acq prefix, e.g. `genes_round_1_1`, None for all")
def check_projection(path, prefix):
    """Check if projection has completed for all tile."""
    from iss_preprocess.pipeline.project import check_projection

    check_projection(path, prefix)


@sync_and_crunch_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def create_grand_averages(path):
    """Create grand average for illumination correction"""
    from iss_preprocess.pipeline.pipeline import create_grand_averages

    create_grand_averages(path, prefix_todo=None)


@sync_and_crunch_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--n-batch",
    help="Number of average batches to compute before taking their median.",
    default=1,
)
def create_all_single_averages(path, n_batch):
    """Average all tiffs in all acquisition folders"""
    from iss_preprocess.pipeline.pipeline import create_all_single_averages

    create_all_single_averages(path, n_batch=n_batch)


@sync_and_crunch_cli.command()
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
    from iss_preprocess.pipeline.pipeline import create_single_average

    create_single_average(
        path,
        subfolder=subfolder,
        subtract_black=subtract_black,
        prefix_filter=prefix_filter,
        suffix=suffix,
        combine_tilestats=combine_stats,
        n_batch=n_batch,
    )
