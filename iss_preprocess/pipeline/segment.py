from os import system
import numpy as np
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..segment import cellpose_segmentation
from .stitch import stitch_registered
from ..io import get_roi_dimensions, load_ops


def segment_all_rois(data_path, prefix="DAPI_1", use_gpu=False):
    """Start batch jobs for segmentation for each ROI.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): acquisition prefix to use for segmentation.
            Defaults to "DAPI_1".
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.
    """
    roi_dims = get_roi_dimensions(data_path)
    script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "segment_roi.sh"
    )
    for roi in roi_dims:
        args = f"--export=DATAPATH={data_path},ROI={roi[0]},PREFIX={prefix}"
        if use_gpu:
            args = args + ",USE_GPU=--use-gpu --partition=gpu --gpus-per-node=1"
        else:
            args = args + " --partition=cpu"
        args = args + f" --output={Path.home()}/slurm_logs/iss_segment_%j.out"

        command = f"sbatch {args} {script_path}"
        print(command)
        system(command)


def segment_roi(
    data_path, iroi, prefix="DAPI_1", reference="genes_round_1_1", use_gpu=False
):
    """Detect cells in a single ROI using Cellpose.

    Much faster with GPU but requires very amount of VRAM for large ROIs.

    Args:
        data_path (str): Relative path to data.
        iroi (int): ROI ID to segment as specificied in MicroManager (i.e. 1-based).
        prefix (str, optional): Acquisition prefix to use for segmentation. Defaults to "DAPI_1".
        reference (str, optional): Acquisition prefix to align the stitched image to.
            Defaults to "genes_round_1_1".
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.
    """
    print(f"running segmentation on roi {iroi} from {data_path} using {prefix}")
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops = load_ops(data_path)
    print(f"stitching {prefix} and aligning to {reference}", flush=True)
    stitched_stack = stitch_registered(
        data_path, ref_prefix=reference, prefix=prefix, roi=iroi
    )
    print("starting segmentation", flush=True)
    masks = cellpose_segmentation(
        stitched_stack,
        channels=(0, 0),
        flow_threshold=ops["cellpose_flow_threshold"],
        min_pix=0,
        dilate_pix=0,
        rescale=ops["cellpose_rescale"],
        model_type=ops["cellpose_model"],
        use_gpu=use_gpu,
    )
    np.save(processed_path / data_path / f"masks_{iroi}.npy", masks)
