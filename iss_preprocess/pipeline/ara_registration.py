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


def load_registration_reference_metadata(data_path, roi):
    """Load metadata file associated with registration reference of one ROI

    This is the "registration_reference_r{roi}_sl{slice_number}.yml" file that contains
    shape and downsampling info.

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the roi

    Returns:
        metadata (dict): Content of the metadata yml file
    """
    processed = Path(PARAMETERS["data_root"]["processed"])
    reg_folder = processed / data_path / "register_to_ara"

    if not reg_folder.is_dir():
        raise IOError("Registration folder does not exists. Perform registration first")
    metadata_file = list(reg_folder.glob(f"registration_reference_r{roi}_sl*.yml"))
    if not len(metadata_file):
        raise IOError(f"No file found for ROI {roi}")
    elif len(metadata_file) > 1:
        raise IOError(f"Found multiple files for ROI {roi}")
    with open(metadata_file[0], "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    return metadata


def load_coordinate_image(data_path, roi, full_scale=True):
    """Load the 3 channel image of ARA coordinates for `roi`

    TODO: should it go in io?
    TODO: load raw registered image or upsample registered to ref image

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI
    """
    processed = Path(PARAMETERS["data_root"]["processed"])
    reg_folder = processed / data_path / "register_to_ara"

    if not reg_folder.is_dir():
        raise IOError("Registration folder does not exists. Perform registration first")
    coord_folder = reg_folder / "ara_coordinates"
    if not coord_folder.is_dir():
        raise IOError(
            "ARA coordinates folder does not exists." + " Perform registration first"
        )
    coord_file = list(coord_folder.glob(f"*_r{roi}_*Coords.tif"))
    if not len(coord_file):
        raise IOError(f"Cannot find coordinates files for roi {roi}")
    elif len(coord_file) > 1:
        raise IOError(f"Found multiple coordinates files for roi {roi}")
    coord_file = coord_file[0]

    coords = load_stack(str(coord_file))
    if full_scale:
        metadata = load_registration_reference_metadata(data_path, roi)
        scale_factor = metadata["pixel_size"] / metadata["original_pixel_size"]
        coords = rescale(coords, scale_factor)

    return coords


def make_area_image(data_path, roi, atlas_size=10):
    """Generate an image with area ID in each pixel

    Args:
        data_path (str): Relative path to data
        roi (int): Roi number to generate
        atlas_size (int, optional): Pixel size of the atlas used to find area if.
            Defaults to 10.

    Returns:
        area_id (np.array): Image with area id of each pixel
    """
    coord = np.clip(load_coordinate_image(data_path, roi), 0, None)
    coord = np.round(coord * 1000 / atlas_size, 0).astype("uint16")

    atlas_name = "allen_mouse_%dum" % atlas_size
    bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
    for channel, max_val in enumerate(bg_atlas.shape):
        coord[:, :, channel] = np.clip(coord[:, :, channel], 0, max_val - 1)
    area_id = bg_atlas.annotation[coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]]
    return area_id


def spots_ara_infos(data_path, spots, atlas_size):
    """Add ARA coordinates and area ID to spots dataframe

    Args:
        data_path (str): Relative path to data
        spots (pd.DataFrame): Spots dataframe
        atlas_size (int): Atlas size (10, 25 or 50) for find areas borders
    """
    raise NotImplementedError