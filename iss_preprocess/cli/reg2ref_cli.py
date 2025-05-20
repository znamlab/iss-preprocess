import click


@click.group()
def reg2ref_cli():
    pass


@reg2ref_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option("-n", "--prefix", default=None, help="Directory prefix to process.")
@click.option("--use-slurm", is_flag=True, default=False, help="Whether to use slurm")
def correct_ref_shifts(path, prefix=None, use_slurm=False):
    """
    Correct X-Y shifts for registration to reference using robust regression
    across tiles.
    """
    from ..diagnostics.diag_reg2ref import (
        check_reg_to_ref_correction,
        check_registration_to_reference,
    )
    from ..pipeline import correct_shifts_to_ref

    if use_slurm:
        from pathlib import Path

        slurm_folder = Path.home() / "slurm_logs" / path
        slurm_folder.mkdir(parents=True, exist_ok=True)
    else:
        slurm_folder = None

    job_id = correct_shifts_to_ref(
        path,
        prefix,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        scripts_name=f"correct_shifts_to_ref_{prefix}",
    )
    check_reg_to_ref_correction(
        path,
        prefix,
        rois=None,
        roi_dimension_prefix=None,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_reg_to_ref_correction_{prefix}",
    )
    check_registration_to_reference(
        path,
        prefix=prefix,
        ref_prefix=None,
        use_slurm=use_slurm,
        slurm_folder=slurm_folder,
        job_dependency=job_id if use_slurm else None,
        scripts_name=f"check_tile_reg_to_ref_{prefix}",
    )


@reg2ref_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--reg-prefix",
    default="barcode_round",
    help="Directory prefix to registration target.",
)
@click.option("-r", "--roi", default=None, help="ROI number. None for all.")
@click.option("-x", "--tilex", default=None, help="Tile X position. None for all.")
@click.option("-y", "--tiley", default=None, help="Tile Y position. None for all.")
@click.option(
    "-m/",
    "--use-masked-correlation/--no-use-masked-correlation",
    default=False,
    help="Whether to use masked correlation.",
)
@click.option(
    "--use-stitched",
    is_flag=True,
    help="Use stitched images for registration instead of running it tile by tile.",
)
def register_to_reference(
    path,
    reg_prefix,
    roi,
    tilex,
    tiley,
    use_masked_correlation,
    use_stitched,
):
    """Register an acquisition to reference tile by tile."""
    from iss_preprocess.pipeline.reg2ref import (
        register_all_tiles_to_ref,
        register_tile_to_ref,
        register_to_ref_using_stitched_registration,
    )

    if use_stitched:
        from pathlib import Path

        if not use_masked_correlation:
            print("!!! Stitched registration might fail without masked correlation !!!")
        if roi is None:
            from iss_preprocess.io import get_roi_dimensions

            roi_dims = get_roi_dimensions(path)
            rois = roi_dims[:, 0]
        elif isinstance(roi, int) or isinstance(roi, str):
            rois = [int(roi)]

        slurm_folder = Path.home() / "slurm_logs" / path
        for roi in rois:
            script_name = f"register_to_ref_{reg_prefix}_roi_{roi}"
            register_to_ref_using_stitched_registration(
                data_path=path,
                roi=roi,
                reg_prefix=reg_prefix,
                use_masked_correlation=use_masked_correlation,
                use_slurm=True,
                slurm_folder=slurm_folder,
                scripts_name=script_name,
                save_plot=True,
            )
        return
    if any([x is None for x in [roi, tilex, tiley]]):
        register_all_tiles_to_ref(path, reg_prefix, use_masked_correlation)
    else:
        print(f"Registering ROI {roi}, Tile ({tilex}, {tiley})", flush=True)
        register_tile_to_ref(
            data_path=path,
            reg_prefix=reg_prefix,
            tile_coors=(int(roi), int(tilex), int(tiley)),
            use_masked_correlation=use_masked_correlation,
        )


@reg2ref_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-s",
    "--spots-prefix",
    default="barcode_round",
    help="File name prefix for spot files (e.g. 'barcode_round').",
    show_default=True,
)
@click.option(
    "--reload/--no-reload",
    default=True,
    help="Whether to reload register_adjacent_tiles shifts.",
    show_default=True,
)
def align_spots(
    path,
    spots_prefix="barcode_round",
    reload=True,
):
    from pathlib import Path

    from iss_preprocess.io import load_ops
    from iss_preprocess.pipeline.align_spots_and_cells import (
        merge_and_align_spots_all_rois,
    )
    from iss_preprocess.pipeline.stitch import register_all_rois_within

    slurm_folder = Path.home() / "slurm_logs" / path / "align_spots"
    slurm_folder.mkdir(parents=True, exist_ok=True)
    ref_job_id = None
    ops = load_ops(path)
    ref_prefix = ops["reference_prefix"]

    ref_job_id = register_all_rois_within(
        path,
        prefix=ref_prefix,
        reload=reload,
        save_plot=True,
        use_slurm=True,
    )

    merge_and_align_spots_all_rois(
        path,
        spots_prefix=spots_prefix,
        ref_prefix=ref_prefix,
        dependency=ref_job_id,
    )


@reg2ref_cli.command()
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
def align_spots_roi(
    path,
    spots_prefix="barcode_round",
    reg_prefix="barcode_round_1_1",
    roi=1,
):
    from iss_preprocess.pipeline.align_spots_and_cells import merge_and_align_spots

    merge_and_align_spots(
        path, spots_prefix=spots_prefix, reg_prefix=reg_prefix, roi=roi
    )
