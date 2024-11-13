import click


@click.group()
def register_cli():
    pass


@register_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "--diag/--no-diag",
    show_default=True,
    default=False,
    help="Save diagnostic cross correlogram plots",
)
@click.option("-f/", "--force-redo/--no-force-redo", default=False, help="Force redo.")
def register_ref_tile(path, prefix, diag, force_redo):
    """Run registration across channels and rounds for the reference tile."""
    from iss_preprocess.pipeline import register_reference_tile

    register_reference_tile(path, prefix, diag, force_redo)


@register_cli.command()
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
    from iss_preprocess.pipeline.register import estimate_shifts_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}")
    estimate_shifts_by_coors(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, suffix=suffix
    )


@register_cli.command()
@click.option(
    "-p", "--path", prompt="Enter data path", help="Data path.", required=True
)
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'", required=True)
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
def correct_shifts(path, prefix, use_slurm=False):
    """Correct X-Y shifts using robust regression across tiles."""
    # import with different name to not get confused with the cli function name
    from iss_preprocess.pipeline import correct_shifts as corr_shifts

    corr_shifts(path, prefix, use_slurm=use_slurm, job_dependency=None)


@register_cli.command()
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


@register_cli.command()
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


@register_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Path prefix, e.g. 'genes_round'")
def estimate_hyb_shifts(path, prefix=None):
    """Estimate X-Y shifts across channels for a hybridisation round for all tiles."""
    from iss_preprocess.io import get_roi_dimensions, load_metadata
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
