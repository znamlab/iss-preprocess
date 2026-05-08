"""Pipeline public API exports.

This file re-exports the most commonly used functions so users can call
`iss.pipeline.<function>` directly from notebooks and scripts.
"""

# Submodules that we want accessible as `iss.pipeline.ara_registration`
from . import ara_registration  # noqa: F401

# Stitching and within-acquisition registration utilities
from .stitch import (  # noqa: F401
	calculate_tile_positions,
	get_tile_corners,
	register_adjacent_tiles,
	register_all_rois_within,
	register_within_acquisition,
	stitch_registered,
	stitch_tiles,
)

# High-level pipeline orchestrators and helpers
from .pipeline import (  # noqa: F401
	average,
	call_spots,
	create_all_single_averages,
	create_grand_averages,
	create_single_average,
	overview_for_ara_registration,
	project_and_average,
	register_acquisition,
	register_fluo_acq,
	register_reference_tile,
	segment_and_stitch_mcherry_cells,
	setup_channel_correction,
)

# Spots/cells alignment helpers
from .align_spots_and_cells import (  # noqa: F401
	align_cell_dataframe,
	align_spots,
	merge_and_align_spots,
	merge_and_align_spots_all_rois,
	merge_roi_spots,
	stitch_cell_dataframes,
)

# Tangential/global volume registration helpers
from .volume_registration import (  # noqa: F401
	apply_affine_to_points,
	build_adjacent_slice_pairs,
	build_registered_volume_stack,
	compose_global_slice_transforms,
	discover_volume_data_paths,
	discover_volume_slices,
	export_unregistered_volume_stack,
	get_volume_root,
	load_bad_slices,
	load_global_slice_transforms,
	load_pairwise_registration_state,
	load_unregistered_volume_stack,
	reduce_pairwise_state,
	register_adjacent_slices,
	register_single_pair,
	register_spots_to_global_volume,
	save_pairwise_registration_state,
	warp_with_affine,
	write_bad_slices_to_chamber_ops,
)

__all__ = [
	# submodules
	"ara_registration",
	# stitch API
	"calculate_tile_positions",
	"get_tile_corners",
	"register_adjacent_tiles",
	"register_all_rois_within",
	"register_within_acquisition",
	"stitch_registered",
	"stitch_tiles",
	# pipeline API
	"average",
	"call_spots",
	"create_all_single_averages",
	"create_grand_averages",
	"create_single_average",
	"overview_for_ara_registration",
	"project_and_average",
	"register_acquisition",
	"register_fluo_acq",
	"register_reference_tile",
	"segment_and_stitch_mcherry_cells",
	"setup_channel_correction",
	# alignment helpers
	"align_cell_dataframe",
	"align_spots",
	"merge_and_align_spots",
	"merge_and_align_spots_all_rois",
	"merge_roi_spots",
	"stitch_cell_dataframes",
	# tangential/global volume API
	"apply_affine_to_points",
	"build_adjacent_slice_pairs",
	"build_registered_volume_stack",
	"compose_global_slice_transforms",
	"discover_volume_data_paths",
	"discover_volume_slices",
	"export_unregistered_volume_stack",
	"get_volume_root",
	"load_bad_slices",
	"load_global_slice_transforms",
	"load_pairwise_registration_state",
	"load_unregistered_volume_stack",
	"reduce_pairwise_state",
	"register_adjacent_slices",
	"register_single_pair",
	"register_spots_to_global_volume",
	"save_pairwise_registration_state",
	"warp_with_affine",
	"write_bad_slices_to_chamber_ops",
]
