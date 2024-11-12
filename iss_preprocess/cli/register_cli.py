import click


@click.group()
def register_cli():
    pass


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
    from iss_preprocess.pipeline import estimate_shifts_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}")
    estimate_shifts_by_coors(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, suffix=suffix
    )


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
