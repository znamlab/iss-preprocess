import click


@click.group()
def volume_registration_cli():
    pass


@volume_registration_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-o",
    "--output-name",
    default="unregistered_slices.npz",
    show_default=True,
    help="Output file name inside the mouse-level tangential_volume folder.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite an existing exported stack.",
)
def export_unregistered_volume(path, output_name, overwrite):
    """Export centered overview slices as a stack for manual masking on the VM."""
    from iss_preprocess.pipeline.volume_registration import (
        export_unregistered_volume_stack,
    )

    target = export_unregistered_volume_stack(
        path,
        output_name=output_name,
        overwrite=overwrite,
    )
    click.echo(f"Saved unregistered slice stack to {target}")
    click.echo(
        "Open that NPZ on the VM, draw or edit `user_masks`, save it back, "
        "then run `iss-volume-registration register-volume-pairs`."
    )


@volume_registration_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--stack-path",
    default=None,
    help=(
        "Path to the exported unregistered NPZ stack. Defaults to "
        "tangential_volume/unregistered_slices.npz."
    ),
)
@click.option(
    "--state-path",
    default=None,
    help=(
        "JSON file used to store pairwise registration results and "
        "review decisions."
    ),
)
@click.option(
    "--force/--no-force",
    default=False,
    show_default=True,
    help="Recompute already accepted pairwise registrations.",
)
@click.option(
    "--local-search-radius",
    default=None,
    type=float,
    help="Optional translation radius for local simulated annealing refinement.",
)
@click.option(
    "--angle-radius-deg",
    default=5.0,
    type=float,
    show_default=True,
    help="Angular search radius used during local refinement.",
)
@click.option(
    "--use-slurm/--no-slurm",
    default=False,
    show_default=True,
    help="Submit one SLURM job per pair plus a reducer instead of running inline.",
)
@click.option(
    "--cleanup-pair-records/--keep-pair-records",
    default=False,
    show_default=True,
    help="Delete per-pair JSON records after the reducer merges them.",
)
def register_volume_pairs(
    path,
    stack_path,
    state_path,
    force,
    local_search_radius,
    angle_radius_deg,
    use_slurm,
    cleanup_pair_records,
):
    """Estimate pairwise slice transforms from the exported unregistered stack."""
    from iss_preprocess.pipeline.volume_registration import (
        get_volume_root,
        register_adjacent_slices,
    )

    if stack_path is None:
        stack_path = get_volume_root(path) / "unregistered_slices.npz"
    result = register_adjacent_slices(
        stack_path=stack_path,
        state_path=state_path,
        force=force,
        local_search_radius=local_search_radius,
        angle_radius_deg=angle_radius_deg,
        use_slurm=use_slurm,
        cleanup_pair_records=cleanup_pair_records,
    )
    if use_slurm:
        click.echo(
            f"Submitted {len(result['pair_job_ids'])} pair jobs + reducer "
            f"{result['reducer_job_id']}"
        )
    else:
        click.echo(f"Saved pairwise registration state to {result}")


@volume_registration_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--stack-path",
    default=None,
    help=(
        "Path to the exported unregistered NPZ stack. Defaults to "
        "tangential_volume/unregistered_slices.npz."
    ),
)
@click.option(
    "--state-path",
    default=None,
    help="JSON file storing pairwise registration results and review decisions.",
)
@click.option(
    "--reference-slice",
    default=None,
    type=int,
    help=(
        "Optional anchor slice number. If omitted a slice is chosen "
        "automatically."
    ),
)
@click.option(
    "--reference-strategy",
    default="largest_area",
    type=click.Choice(["largest_area", "middle", "first"]),
    show_default=True,
    help="How to choose the anchor slice when --reference-slice is omitted.",
)
@click.option(
    "-o",
    "--output-name",
    default="global_slice_transforms.npz",
    show_default=True,
    help="Output transform table name inside tangential_volume.",
)
def compose_volume_transforms(
    path,
    stack_path,
    state_path,
    reference_slice,
    reference_strategy,
    output_name,
):
    """Concatenate adjacent transforms into slice-to-global transforms."""
    from iss_preprocess.pipeline.volume_registration import (
        compose_global_slice_transforms,
        get_volume_root,
    )

    if stack_path is None:
        stack_path = get_volume_root(path) / "unregistered_slices.npz"
    transforms_path = compose_global_slice_transforms(
        stack_path=stack_path,
        state_path=state_path,
        reference_slice=reference_slice,
        reference_strategy=reference_strategy,
        output_name=output_name,
    )
    click.echo(f"Saved global slice transforms to {transforms_path}")


@volume_registration_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--transforms-path",
    default=None,
    help=(
        "Path to the saved global slice transform table. Defaults to "
        "tangential_volume/global_slice_transforms.npz."
    ),
)
@click.option(
    "-n",
    "--prefix",
    required=True,
    help="Acquisition prefix to stitch into the volume.",
)
@click.option(
    "-o",
    "--output-name",
    default=None,
    help="Optional NPZ file name for the registered volume stack.",
)
@click.option(
    "--suffix",
    default="max",
    show_default=True,
    help="Projection suffix to stitch for each ROI.",
)
@click.option(
    "--z-step-um",
    default=None,
    type=float,
    help="Optional z-step in microns for the output volume indexing.",
)
def build_registered_volume(
    path,
    transforms_path,
    prefix,
    output_name,
    suffix,
    z_step_um,
):
    """Build a registered tangential volume stack from stitched ROIs."""
    from iss_preprocess.pipeline.volume_registration import (
        build_registered_volume_stack,
        get_volume_root,
    )

    if transforms_path is None:
        transforms_path = get_volume_root(path) / "global_slice_transforms.npz"
    output_path = build_registered_volume_stack(
        data_path=path,
        transforms_path=transforms_path,
        prefix=prefix,
        output_name=output_name,
        suffix=suffix,
        z_step_um=z_step_um,
    )
    click.echo(f"Saved registered volume stack to {output_path}")


@volume_registration_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "--transforms-path",
    default=None,
    help=(
        "Path to the saved global slice transform table. Defaults to "
        "tangential_volume/global_slice_transforms.npz."
    ),
)
@click.option(
    "-s",
    "--spots-prefix",
    default="barcode_round",
    show_default=True,
    help="ROI-level spots prefix to project into the global frame.",
)
@click.option(
    "-o",
    "--output-name",
    default=None,
    help="Optional pickle file name for the merged global spots table.",
)
@click.option(
    "--output-unit",
    default="pixel",
    type=click.Choice(["pixel", "um"]),
    show_default=True,
    help="Whether to save global x/y coordinates in overview pixels or microns.",
)
def register_spots_global(
    path,
    transforms_path,
    spots_prefix,
    output_name,
    output_unit,
):
    """Register already-detected spots into the global tangential volume frame."""
    from iss_preprocess.pipeline.volume_registration import (
        get_volume_root,
        register_spots_to_global_volume,
    )

    if transforms_path is None:
        transforms_path = get_volume_root(path) / "global_slice_transforms.npz"
    output_path = register_spots_to_global_volume(
        data_path=path,
        transforms_path=transforms_path,
        spots_prefix=spots_prefix,
        output_name=output_name,
        output_unit=output_unit,
    )
    click.echo(f"Saved global spots table to {output_path}")
