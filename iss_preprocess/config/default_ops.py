"""
Default parameters for the pipeline.
"""
from ..image import analyze_dark_frames
from flexiznam.config import PARAMETERS
from pathlib import Path

# TODO: Add description of what each option does somewhere. Maybe here
DEFAULT_OPS = {
    "average_clip_value": 2000,
    "average_median_filter": 5,
    "correction_tiles": [(1, 0, 0)],
    "correction_quantile": 0.9999,
    "ref_tile": (1, 0, 0),
    "barcode_ref_tiles": [(1, 0, 0)],
    "ref_ch": 0,
    "ref_round": 0,
    "filter_r": (2, 4),
    "detection_threshold": 0.15,
    "isolation_threshold": 0.05,
    "barcode_correct_channels": "round1_only",
    "barcode_detection_threshold": 0.3,
    "barcode_isolation_threshold": 0.3,
    "barcode_detection_threshold_basecalling": 0.1,
    "hybridisation_correct_channels": True,
    "hyb_spot_detection_threshold": 0.5,
    "codebook": "codebook_83gene_pool.csv",
    "projection": "fstack",
    "hybridisation_projection": "max",
    "spot_extraction_radius": 2,
    "spot_shape_radius": 7,
    "spot_shape_neighbor_filter_size": 9,
    "spot_shape_neighbor_threshold": 15,
    "omp_min_intensity": 0.005,
    "omp_threshold": 0.2,
    "omp_max_genes": 12,
    "omp_alpha": 200,
    "spot_threshold": 0.15,
    "spot_rho": 2,
    "barcode_spot_rho": 0.5,
    "cellpose_model": "cyto",
    "cellpose_rescale": 0.55,
    "cellpose_flow_threshold": 0.4,
}

processed_path = Path(PARAMETERS["data_root"]["processed"])
dark_frame_path = "becalia_iss_microscope/calibration/20221209_dark_frame/20221209_dark_frame_MMStack_Pos0.ome.tif"
DEFAULT_OPS["black_level"], _ = analyze_dark_frames(processed_path / dark_frame_path)
