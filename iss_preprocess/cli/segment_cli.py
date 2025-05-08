"""Segmentation and cell detection cli"""

import click


@click.group()
def segment_cli():
    pass


@segment_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def segment_mcherry_tile(path, prefix, roi, tilex, tiley):
    """Segment mCherry channel for a single tile."""
    from iss_preprocess.pipeline.segment import segment_mcherry_tile as smt

    smt(
        path,
        prefix,
        roi,
        tilex,
        tiley,
    )


@segment_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
@click.option(
    "-r", "--roi", default=1, prompt="Enter ROI number", help="Number of the ROI.."
)
@click.option("-x", "--tilex", default=0, help="Tile X position")
@click.option("-y", "--tiley", default=0, help="Tile Y position.")
def remove_non_cell_masks(path, prefix, roi, tilex, tiley):
    """Remove masks from mCherry tiles that don't correspond to cells."""
    from iss_preprocess.pipeline.segment import _remove_non_cell_masks

    # TODO: move out of main CLI
    _remove_non_cell_masks(
        path,
        prefix,
        roi,
        tilex,
        tiley,
    )


@segment_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n", "--prefix", default="mCherry_1", help="Path prefix, e.g. 'mCherry_1'"
)
def filter_mcherry_masks(path, prefix):
    """Find mCherry cells using a GMM to cluster masks based on their
    morphological features. Then remove non-cell masks from each tile.
    """
    from iss_preprocess.pipeline.pipeline import _gmm_cluster_mcherry_cells

    # TODO: move out of main CLI
    _gmm_cluster_mcherry_cells(path, prefix)


@segment_cli.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
@click.option(
    "-n",
    "--prefix",
    default="DAPI_1",
    help="Path prefix to use for segmentation, e.g. 'DAPI_1",
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
    from iss_preprocess.pipeline.pipeline import segment_roi

    segment_roi(path, roi, prefix, use_gpu=use_gpu)
