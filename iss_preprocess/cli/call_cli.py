import click

from iss_preprocess.cli.iss import iss_cli


@click.group()
def call_cli():
    pass


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def extract(path):
    """Start batch jobs to run OMP on all tiles in a dataset."""
    from iss_preprocess.pipeline.core import batch_process_tiles

    batch_process_tiles(path, "extract_tile")


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
@click.option("--ref-tile-index", default=0, help="Reference tile index")
def check_basecall(path, use_slurm=False, ref_tile_index=0):
    """Check if basecalling has completed for all tiles."""
    from iss_preprocess.diagnostics.diag_sequencing import check_barcode_basecall

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    check_barcode_basecall(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        ref_tile_index=ref_tile_index,
    )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def basecall(path):
    """Start batch jobs to run basecalling for barcodes on all tiles."""
    from iss_preprocess.pipeline.core import batch_process_tiles

    job_ids, failed_job = batch_process_tiles(path, "basecall_tile")
    click.echo(f"Basecalling started for {len(job_ids)} tiles.")
    click.echo(f"Last job id: {job_ids[-1]}")

    from pathlib import Path

    from iss_preprocess.diagnostics.diag_sequencing import check_barcode_basecall

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    check_barcode_basecall(
        path,
        use_slurm=True,
        job_dependency=job_ids,
        slurm_folder=slurm_folder,
        scripts_name="check_basecall",
    )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def check_omp(path, roi, tilex, tiley, use_slurm=True):
    """Compute average spot image."""
    from iss_preprocess.pipeline.pipeline import check_omp_thresholds

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path / "check_omp"
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    if roi is not None and tilex is not None and tiley is not None:
        check_omp_thresholds(
            path,
            tile_coors=(roi, tilex, tiley),
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="check_omp",
        )
    else:
        check_omp_thresholds(
            path,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="check_omp",
        )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def check_omp_alpha(path, roi, tilex, tiley, use_slurm=True):
    """Compute average spot image."""
    from iss_preprocess.pipeline.pipeline import check_omp_alpha_thresholds

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path / "check_omp"
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    if roi is not None and tilex is not None and tiley is not None:
        check_omp_alpha_thresholds(
            path,
            tile_coors=(roi, tilex, tiley),
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="check_omp",
        )
    else:
        check_omp_alpha_thresholds(
            path,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name="check_omp",
        )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="genes_round", help="Path prefix, e.g. 'genes_round'"
)
def spot_sign_image(path, prefix="genes_round"):
    """Compute average spot image."""
    from iss_preprocess.pipeline.pipeline import compute_spot_sign_image

    compute_spot_sign_image(path, prefix)


@iss_cli.command()
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
    from iss_preprocess.pipeline.pipeline import detect_genes_on_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    detect_genes_on_tile(path, (roi, x, y), save_stack=save)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
def basecall_tile(path, roi=1, x=0, y=0):
    """Run basecalling for barcodes on a single tile."""
    from iss_preprocess.pipeline.pipeline import basecall_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    basecall_tile(path, (roi, x, y))


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
def setup_omp(path, use_slurm=True):
    """Estimate bleedthrough matrices and construct gene dictionary for OMP."""
    from pathlib import Path

    from iss_preprocess.pipeline.pipeline import setup_omp

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_omp(
        path, use_slurm=use_slurm, slurm_folder=slurm_folder, scripts_name="setup_omp"
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
def setup_barcodes(path, use_slurm=True):
    """Estimate bleedthrough matrices for barcode calling."""
    from pathlib import Path

    from iss_preprocess.pipeline.pipeline import setup_barcode_calling

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_barcode_calling(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_barcodes",
    )


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--prefix",
    default=None,
    help="Path prefix, e.g. 'hybridisation_round'. If None,"
    + " all hybridisation rounds are processed.",
)
@click.option(
    "--use-slurm/--local", is_flag=True, default=True, help="Whether to use slurm"
)
def setup_hybridisation(path, prefix=None, use_slurm=True):
    """Estimate bleedthrough matrices for hybridisation spots."""
    from iss_preprocess.pipeline.pipeline import setup_hyb_spot_calling

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    setup_hyb_spot_calling(path, prefix, use_slurm=use_slurm, slurm_folder=slurm_folder)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def hyb_spots(path):
    """Detect hybridisation in all ROIs / hybridisation rounds"""
    from iss_preprocess.pipeline.pipeline import extract_hyb_spots_all

    extract_hyb_spots_all(path)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", help="Path prefix for spot detection")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
def extract_hyb_spots(path, prefix, roi, tilex, tiley):
    """Detect hybridisation spots in a single ROI / hybridisation round"""

    if tilex is not None and tiley is not None:
        from iss_preprocess.pipeline.hybridisation import extract_hyb_spots_tile

        tile_coors = (roi, tilex, tiley)
        extract_hyb_spots_tile(path, tile_coors, prefix)
    else:
        from iss_preprocess.pipeline.hybridisation import extract_hyb_spots_roi

        extract_hyb_spots_roi(path, prefix, roi)


@iss_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--genes", is_flag=True, help="Whether to call spots for genes.")
@click.option("--barcodes", is_flag=True, help="Whether to call spots for barcodes.")
@click.option(
    "--hybridisation", is_flag=True, help="Whether to call spots for hybridisation."
)
def call_spots(path, genes, barcodes, hybridisation):
    """Call spots for genes, barcodes and hybridisation rounds"""
    from iss_preprocess.pipeline.pipeline import call_spots

    called = []
    for spot_type in ["genes", "barcodes", "hybridisation"]:
        if locals()[spot_type]:
            called.append(spot_type)
    if not called:
        print("No spots to call.")
        return
    print(f"Calling spots for {', '.join(called)}")

    call_spots(path, genes, barcodes, hybridisation)
