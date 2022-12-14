from os import system
import numpy as np
from flexiznam.config import PARAMETERS
from pathlib import Path
from ..segment import cellpose_segmentation
from .stitch import stitch_and_register


def segment_all_rois(data_path, prefix="DAPI_1"):
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    roi_dims = np.load(processed_path / data_path / "roi_dims.npy")
    script_path = str(Path(__file__).parent.parent.parent / "segment_roi.sh")
    for roi in roi_dims:
        args = f"--export=DATAPATH={data_path},ROI={roi[0]},PREFIX={prefix}"
        args = args + f" --output={Path.home()}/slurm_logs/iss_segment_%j.out"
        command = f"sbatch {args} {script_path}"
        print(command)
        system(command)


def segment_roi(data_path, iroi, prefix="DAPI_1", reference="genes_round_1_1"):
    print(f"running segmentation on roi {iroi} from {data_path} using {prefix}")
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    ops_path = processed_path / data_path / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    print(f"stitching {prefix} and aligning to {reference}", flush=True)
    stitched_stack = stitch_and_register(data_path, reference, prefix, roi=iroi)
    print("starting segmentation", flush=True)
    masks = cellpose_segmentation(
        stitched_stack,
        channels=(0, 0),
        flow_threshold=ops["cellpose_flow_threshold"],
        min_pix=0,
        dilate_pix=0,
        rescale=ops["cellpose_rescale"],
        model_type=ops["cellpose_model"],
    )
    np.save(processed_path / data_path / f"masks_{iroi}.npy", masks)
