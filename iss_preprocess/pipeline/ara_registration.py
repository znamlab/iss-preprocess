from pathlib import Path
import numpy as np
import yaml
from skimage.transform import rescale
import bg_atlasapi as bga
from flexiznam import PARAMETERS
from .pipeline import stitch
from ..io import load_section_position, load_metadata, load_stack


def find_roi_position_on_cryostat(data_path, bulb_first=True):
    """Find the A/P position of each ROI relative to the first collected slice

    Args:
        data_path (str): Relative path to the data
        bulb_first (bool, optional): Was the first slice closer to the olfactory
            bulb than the last? Defaults to True.

    Returns:
        roi_slice_pos_um (dict): For each ROI, the slice depth in um relative to the
            first collected slice
        min_step (float): Minimum thickness between two slices
    """
    processed = Path(PARAMETERS["data_root"]["processed"])
    metadata = load_metadata(data_path)
    rois = metadata["ROI"].keys()

    section_info = load_section_position(data_path)
    section_info.sort_values(by="absolute_section", inplace=True)
    if any(np.diff(section_info.absolute_section) > 1):
        raise IOError(
            "I need to know the thickness of all the slices.\n"
            + "Please add missing sections to `section_position.csv`"
        )
    # This assumes that all slices are in the csv file but allows for irregular
    # thickness
    slicing_order = 1 if bulb_first else -1
    increase = section_info.section_thickness_um.values * slicing_order
    section_info["section_position"] = increase.cumsum()

    # find where is each slice of the chamber in the section order of the whole brain
    # the chamber folder should be called chamber_XX
    chamber = int((processed / data_path).name.split("_")[1])
    chamber_pos2section_order = {
        s.chamber_position: s.section_position
        for _, s in section_info[section_info.chamber == chamber].iterrows()
    }
    # find where is each roi in the chamber
    roi_id2chamber_pos = {roi: metadata["ROI"][roi]["chamber_position"] for roi in rois}
    # combine both to find for each ROI the distance sliced since the first
    roi_slice_pos_um = {
        roi: chamber_pos2section_order[roi_id2chamber_pos[roi]] for roi in rois
    }

    return roi_slice_pos_um, section_info.section_thickness_um.min()


