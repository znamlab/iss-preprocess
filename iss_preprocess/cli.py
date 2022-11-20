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