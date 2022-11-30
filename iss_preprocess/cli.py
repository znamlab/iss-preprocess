import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('-p', '--path', prompt='Enter data path', help='Data path.')
@click.option('-r', '--roi', default=1, prompt='Enter ROI number',
              help='Number of the ROI..')
@click.option('-x', default=0, help='Tile X position')
@click.option('-y', default=0, help='Tile Y position.')
def extract_tile(path, roi=1, x=0, y=0):
    from iss_preprocess.pipeline import run_omp_on_tile

    """Run OMP and a single tile and detect gene spots."""
    click.echo(f'Processing ROI {roi}, tile {x}, {y} from {path}')
    run_omp_on_tile(path, (roi,x,y), save_stack=True)


@cli.command()
@click.option('-p', '--path', prompt='Enter data path', help='Data path.')
@click.option('-n', '--prefix', prompt='Enter path prefix', help='Path prefile, e.g. round_01_1')
@click.option('-r', '--roi', default=1, prompt='Enter ROI number',
              help='Number of the ROI..')
@click.option('-x', default=0, help='Tile X position')
@click.option('-y', default=0, help='Tile Y position.')
@click.option("--overwrite", is_flag=True, show_default=True, default=False, 
    help="Whether to overwrite tiles if already projected.")
def project_tile(path, prefix, roi=1, x=0, y=0, overwrite=False):
    from iss_preprocess.pipeline import project_tile_by_coors

    """Run OMP and a single tile and detect gene spots."""
    click.echo(f'Projecting ROI {roi}, {prefix}, tile {x}, {y} from {path}')
    project_tile_by_coors(path, prefix, (roi,x,y), overwrite=overwrite)


@cli.command()
@click.option('-p', '--path', prompt='Enter data path', help='Data path.')
@click.option('-n', '--prefix', prompt='Enter path prefix', help='Path prefile, e.g. round_01_1')
@click.option('-r', '--roi', default=1, prompt='Enter ROI number',
              help='Number of the ROI..')
@click.option('-x', default=0, help='Tile X position')
@click.option('-m', '--max-col', default=0, help='Maximum column index.')
@click.option("--overwrite", is_flag=True, show_default=True, default=False, 
    help="Whether to overwrite tiles if already projected.")
def project_row(path, prefix, roi=1, x=0, max_col=0, overwrite=False):
    from iss_preprocess.pipeline import project_tile_row

    """Run OMP and a single tile and detect gene spots."""
    click.echo(f'Projecting ROI {roi}, {prefix}, row {x}, {y} from {path}')
    project_tile_row(path, prefix, roi, x, max_col, overwrite=overwrite)