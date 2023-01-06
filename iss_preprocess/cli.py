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
@click.option("-m", "--nrounds", default=7, help="Number of sequencing rounds")
def register_tile(path, prefix, roi, tilex, tiley, suffix="fstack", nrounds=7):
    """Estimate X-Y shifts across rounds and channels for a single tile."""
    from iss_preprocess.pipeline import estimate_shifts_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}")
    estimate_shifts_by_coors(
        path,
        tile_coors=(roi, tilex, tiley),
        prefix=prefix,
        suffix=suffix,
        nrounds=nrounds,
    )


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
def register_hyb_tile(path, prefix, roi, tilex, tiley, suffix="fstack"):
    """Estimate X-Y shifts across rounds and channels for a single tile."""
    from iss_preprocess.pipeline import estimate_shifts_and_angles_by_coors

    click.echo(f"Registering ROI {roi}, tile {tilex}, {tiley} from {path}")
    estimate_shifts_and_angles_by_coors(
        path, tile_coors=(roi, tilex, tiley), prefix=prefix, suffix=suffix
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "-s", "--suffix", default="fstack", help="Projection suffix, e.g. 'fstack'"
)
@click.option("-m", "--nrounds", default=7, help="Number of sequencing rounds")
def estimate_shifts(path, prefix, suffix="fstack", nrounds=7):
    """Estimate X-Y shifts across rounds and channels for all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles

    additional_args = f",PREFIX={prefix},SUFFIX={suffix},NROUNDS={nrounds}"
    batch_process_tiles(path, script="register_tile", additional_args=additional_args)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option(
    "-s", "--suffix", default="fstack", help="Projection suffix, e.g. 'fstack'"
)
def estimate_hyb_shifts(path, prefix, suffix="fstack"):
    """Estimate X-Y shifts across channels for a hybridisation round for all tiles."""
    from iss_preprocess.pipeline import batch_process_tiles

    additional_args = f",PREFIX={prefix},SUFFIX={suffix}"
    batch_process_tiles(
        path, script="register_hyb_tile", additional_args=additional_args
    )


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
def correct_shifts(path, prefix):
    """Correct X-Y shifts using robust regression across tiles."""
    from iss_preprocess.pipeline import correct_shifts

    correct_shifts(path, prefix)


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
@click.option("-n", "--prefix", help="Path prefix, e.g. 'genes_round'")
@click.option("-r", "--rounds", default=7, help="Number of sequencing rounds")
def register_ref_tile(path, prefix, rounds=7):
    """Run registration across channels and rounds for the reference tile."""
    from iss_preprocess.pipeline import register_reference_tile

    register_reference_tile(path, prefix=prefix, nrounds=rounds)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", help="Path prefix to use for segmentation, e.g. 'DAPI_1"
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
    "-n", "--prefix", help="Path prefix to use for segmentation, e.g. 'DAPI_1"
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
def align_barcodes(path):
    from iss_preprocess.pipeline import merge_and_align_barcodes_all_rois

    merge_and_align_barcodes_all_rois(path)


@cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=1, help="Number of the ROI to segment.")
def align_barcodes_roi(path, roi=1):
    from iss_preprocess.pipeline import merge_and_align_barcodes

    merge_and_align_barcodes(path, roi)
