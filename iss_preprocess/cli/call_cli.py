import click


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
def basecall_somata(path):
    """Start batch jobs to run basecalling for barcodes on all tiles."""
    from iss_preprocess.pipeline.core import batch_process_tiles

    job_ids, failed_job = batch_process_tiles(path, "basecall_somata_tile")
    click.echo(f"Basecalling started for {len(job_ids)} tiles.")
    click.echo(f"Last job id: {job_ids[-1]}")

    # TODO soma basecalling diagnostics
    # from pathlib import Path

    # from iss_preprocess.diagnostics.diag_sequencing import check_barcode_basecall

    # slurm_folder = Path.home() / "slurm_logs" / path
    # slurm_folder.mkdir(parents=True, exist_ok=True)
    # check_barcode_basecall(
    #     path,
    #     use_slurm=True,
    #     job_dependency=job_ids,
    #     slurm_folder=slurm_folder,
    #     scripts_name="check_basecall_soma",
    # )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--force", is_flag=True, help="Overwrite existing valid trace caches.")
def extract_soma_traces(path, force=False):
    """Start batch jobs to extract/cache soma traces on all tiles."""
    from iss_preprocess.pipeline.somata import extract_soma_trace_tiles

    job_ids, failed_job = extract_soma_trace_tiles(path, use_slurm=True, force=force)
    click.echo(f"Soma trace extraction started for {len(job_ids)} tiles.")
    click.echo(f"Last job id: {job_ids[-1]}")


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-r", "--roi", default=None, help="Number of the ROI..")
@click.option("-x", "--tilex", default=None, help="Tile X position")
@click.option("-y", "--tiley", default=None, help="Tile Y position.")
@click.option("--use-slurm", is_flag=True, default=True, help="Whether to use slurm")
def check_omp(path, roi, tilex, tiley, use_slurm=True):
    """Compute average spot image."""
    from iss_preprocess.diagnostics.diag_sequencing import check_omp_thresholds

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
    from iss_preprocess.diagnostics.diag_sequencing import check_omp_alpha_thresholds

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
    from iss_preprocess.pipeline.sequencing import compute_spot_sign_image

    compute_spot_sign_image(path, prefix)


@call_cli.command()
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
    from iss_preprocess.pipeline.sequencing import detect_genes_on_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    detect_genes_on_tile(path, (roi, x, y), save_stack=save)


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
def basecall_tile(path, roi=1, x=0, y=0):
    """Run basecalling for barcodes on a single tile."""
    from iss_preprocess.pipeline.sequencing import basecall_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    basecall_tile(path, (roi, x, y))


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
@click.option(
    "--use-trace-cache/--live-extract",
    default=True,
    show_default=True,
    help="Use cached soma traces, or live-load image data for diagnostics.",
)
def basecall_somata_tile(path, roi=1, x=0, y=0, use_trace_cache=True):
    """Run basecalling for barcodes on a single tile."""
    from iss_preprocess.pipeline.sequencing import basecall_somata_tile

    click.echo(f"Processing ROI {roi}, tile {x}, {y} from {path}")
    basecall_somata_tile(path, (roi, x, y), use_trace_cache=use_trace_cache)


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", default=0, help="Tile X position")
@click.option("-y", default=0, help="Tile Y position.")
@click.option("--force", is_flag=True, help="Overwrite an existing valid trace cache.")
def extract_soma_trace_tile(path, roi=1, x=0, y=0, force=False):
    """Extract/cache soma traces on a single tile."""
    from iss_preprocess.pipeline.somata import extract_soma_trace_tile

    click.echo(f"Extracting soma traces for ROI {roi}, tile {x}, {y} from {path}")
    extract_soma_trace_tile(path, (roi, x, y), force=force)



@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
@click.option("--force-redo", is_flag=True, help="Whether to force redo")
def setup_omp(path, use_slurm=True, force_redo=False):
    """Estimate bleedthrough matrices and construct gene dictionary for OMP."""
    from pathlib import Path

    from iss_preprocess.pipeline.sequencing import setup_omp

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_omp(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_omp",
        force_redo=force_redo,
    )


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
@click.option("--force-redo", is_flag=True, help="Whether to force redo")
def setup_barcodes(path, use_slurm=True, force_redo=False):
    """Estimate bleedthrough matrices for barcode calling."""
    from pathlib import Path

    from iss_preprocess.pipeline.sequencing import setup_barcode_calling

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_barcode_calling(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_barcodes",
        force_redo=force_redo,
    )

@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("--use-slurm", is_flag=True, help="Whether to use slurm")
@click.option("--force-redo", is_flag=True, help="Whether to force redo")
def setup_soma_barcodes(path, use_slurm=True, force_redo=False):
    """Estimate bleedthrough matrices for barcode calling of barcode filled cells."""
    from pathlib import Path

    from iss_preprocess.pipeline.somata import setup_soma_calling_reference

    slurm_folder = Path.home() / "slurm_logs" / path
    slurm_folder.mkdir(parents=True, exist_ok=True)
    setup_soma_calling_reference(
        path,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_soma_barcodes",
        force_redo=force_redo,
    )


@call_cli.command()
@click.option(
    "-p",
    "--path",
    "paths",
    multiple=True,
    required=True,
    help="Chamber data path. Repeat -p for each chamber to pool.",
)
@click.option(
    "--use-slurm/--local",
    is_flag=True,
    default=True,
    help="Whether to use slurm",
)
def setup_shared_soma_barcodes(paths, use_slurm=True):
    """Build one soma cluster-means matrix from reference tiles pooled across
    multiple chambers of one mouse, saved at the mouse level alongside
    diagnostic plots.

    Each consuming chamber must have ``use_shared_soma_cluster_means: true`` in
    its ops for soma basecalling to read the shared file.
    """
    from pathlib import Path

    from iss_preprocess.pipeline.somata import build_shared_soma_cluster_means

    mouse_rel = Path(paths[0]).parent
    slurm_folder = Path.home() / "slurm_logs" / mouse_rel / "shared_soma_reference"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    build_shared_soma_cluster_means(
        list(paths),
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name="setup_shared_soma_barcodes",
        save=True,
        save_diagnostics=True,
    )


@call_cli.command()
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
    from iss_preprocess.pipeline.hybridisation import setup_hyb_spot_calling

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None
    setup_hyb_spot_calling(path, prefix, use_slurm=use_slurm, slurm_folder=slurm_folder)


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def hyb_spots(path):
    """Detect hybridisation in all ROIs / hybridisation rounds"""
    from iss_preprocess.pipeline.hybridisation import extract_hyb_spots_all

    extract_hyb_spots_all(path)


@call_cli.command()
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


@call_cli.command()
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


@call_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--use-slurm/--local", is_flag=True, default=True, help="Whether to use slurm"
)
def setup_channel_correction(path, use_slurm=True):
    """Setup channel correction for barcode, genes and hybridisation rounds"""

    from iss_preprocess.pipeline.pipeline import setup_channel_correction as scc

    scc(path, use_slurm=use_slurm)
    click.echo("Channel correction setup complete.")
