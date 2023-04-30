from pathlib import Path
import numpy as np
import yaml
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter
import bg_atlasapi as bga
from flexiznam import PARAMETERS
from . import stitch
import cv2
from ..io import (
    load_section_position,
    load_metadata,
    load_stack,
    get_pixel_size,
    save_ome_tiff_pyramid,
)


def find_roi_position_on_cryostat(data_path):
    """Find the A/P position of each ROI relative to the first collected slice

    The section order is guess from the sign of `section_thickness_um`, positive for
    antero-posterior slicing (starting from the olfactory bulb), negative for opposite.

    Args:
        data_path (str): Relative path to the data

    Returns:
        roi_slice_pos_um (dict): For each ROI, the slice depth in um relative to the
            first collected slice
        min_step (float): Minimum thickness between two slices

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    metadata = load_metadata(data_path)
    rois = metadata["ROI"].keys()

    section_info = load_section_position(data_path)
    section_info.sort_values(by="absolute_section", inplace=True)
    constant_thickness = np.sum(np.diff(section_info.section_thickness_um)) == 0
    if any(np.diff(section_info.absolute_section) > 1):
        if not constant_thickness:
            raise IOError(
                "I need to know the thickness of all the slices.\n"
                + "Please add missing sections to `section_position.csv`"
            )
        pos_um = section_info.absolute_section * section_info.section_thickness_um
    else:
        # we have all the slices in the csv, we can deal with variable thickness
        increase = section_info.section_thickness_um.values
        section_info["section_position"] = increase.cumsum()

    # find where is each slice of the chamber in the section order of the whole brain
    # the chamber folder should be called chamber_XX
    chamber = int((processed_path / data_path).name.split("_")[1])
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
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    reg_folder = processed_path / data_path / "register_to_ara"
    chamber = reg_folder.parent.name
    if not reg_folder.is_dir():
        raise IOError("Registration folder does not exists. Perform registration first")
    metadata_file = list(reg_folder.glob(f"{chamber}_r{roi}_sl*.yml"))
    if not len(metadata_file):
        raise IOError(f"No file found for ROI {roi}")
    elif len(metadata_file) > 1:
        raise IOError(f"Found multiple files for ROI {roi}")
    with open(metadata_file[0], "r") as fhandle:
        metadata = yaml.safe_load(fhandle)
    return metadata


def load_coordinate_image(data_path, roi, full_scale=False):
    """Load the 3 channel image of ARA coordinates for `roi`

    TODO: should it go in io?

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI
        full_scale (bool, optional): If true, returns the full scale image, otherwise
            the downsample version used for registration. Defaults to False.

    Returns:
        coords (np.ndarray): 3 channel image of ARA coordinates

    """
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    reg_folder = processed_path / data_path / "register_to_ara"

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


def make_area_image(data_path, roi, atlas_size=10, full_scale=False):
    """Generate an image with area ID in each pixel

    Args:
        data_path (str): Relative path to data
        roi (int): Roi number to generate
        atlas_size (int, optional): Pixel size of the atlas used to find area id.
            Defaults to 10.
        full_scale (bool, optional): If true, returns the full scale image, otherwise
            the downsample version used for registration. Defaults to False.

    Returns:
        area_id (np.array): Image with area id of each pixel

    """
    coord = np.clip(
        load_coordinate_image(data_path, roi, full_scale=full_scale), 0, None
    )
    coord = np.round(coord * 1000 / atlas_size, 0).astype("uint16")

    atlas_name = "allen_mouse_%dum" % atlas_size
    bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
    for channel, max_val in enumerate(bg_atlas.shape):
        coord[:, :, channel] = np.clip(coord[:, :, channel], 0, max_val - 1)
    area_id = bg_atlas.annotation[coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]]
    return area_id


def spots_ara_infos(data_path, spots, roi, atlas_size=10, acronyms=False, inplace=True):
    """Add ARA coordinates and area ID to spots dataframe

    Args:
        data_path (str): Relative path to data
        spots (pd.DataFrame): Spots dataframe
        atlas_size (int, optional): Atlas size (10, 25 or 50) for find areas borders.
            Defaults to 10
        acronyms (bool, optional): Add an acronym column with area name. Defaults to
            False.
        inplace (bool, optional): add the column to spots inplace or return a copy.
            Defaults to True

    Returns:
        spots (pd.DataFrame): reference or copy of spots dataframe with four more
            columns: `ara_x`, `ara_y`, `ara_z`, and `area_id`

    """
    if not inplace:
        spots = spots.copy()
    metadata = load_registration_reference_metadata(data_path, roi)
    coords = load_coordinate_image(data_path, roi)
    spot_xy = spots.loc[:, ["x", "y"]].values / metadata["downsample_ratio"]
    spot_xy = np.round(spot_xy).astype(int)
    spot_coords = coords[spot_xy[:, 1], spot_xy[:, 0], :]
    for i, w in enumerate("xyz"):
        spots[f"ara_{w}"] = spot_coords[:, i]
    area_map = make_area_image(data_path, roi, atlas_size=atlas_size)
    spot_area = area_map[spot_xy[:, 1], spot_xy[:, 0]]
    spots["area_id"] = spot_area

    if acronyms:
        atlas_name = "allen_mouse_%dum" % atlas_size
        bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
        labels = bg_atlas.lookup_df.set_index("id")
        spots["area_acronym"] = "outside"
        valid = spots.area_id != 0
        spots.loc[valid, "area_acronym"] = labels.loc[
            spots.area_id[valid], "acronym"
        ].values

    return spots


def overview_single_roi(
    data_path,
    roi,
    slice_id,
    chan2use=(0, 1, 2, 3),
    sigma_blur=10,
    agg_func=np.nanmean,
    reference_prefix="genes_round_1_1",
    subresolutions=5,
    max_pixel_size=2,
):
    print(f"Data path: {data_path}")
    print(f"Roi: {roi}", flush=True)
    print(f"Slice id: {slice_id}", flush=True)
    print(f"Sigma blur: {sigma_blur}", flush=True)
    sigma_blur = float(sigma_blur)
    chamber = Path(data_path).name
    processed_path = Path(PARAMETERS["data_root"]["processed"])
    registration_folder = processed_path / data_path / "register_to_ara"

    print("Finding shifts")
    ops = np.load(processed_path / data_path / "ops.npy", allow_pickle=True).item()

    print("Finding pixel size")
    pixel_size = get_pixel_size(data_path, reference_prefix)

    if chan2use is None:
        chan2use = [ops["ref_ch"]]
    if isinstance(chan2use, int):
        chan2use = [chan2use]
    chan2use = [int(c) for c in chan2use]

    print("Stitching ROI")
    stitched_stack = stitch.stitch_registered(
        roi=roi,
        filter_r=False,
        data_path=data_path,
        prefix=reference_prefix,
        channels=chan2use,
    )
    print("Aggregating", flush=True)
    stitched_stack = agg_func(stitched_stack, axis=2)

    # get chamber position, and then section position
    log = dict(
        original_dtype=str(stitched_stack.dtype),
        original_shape=list(stitched_stack.shape),
        original_pixel_size=pixel_size,
    )
    ratio = int(max_pixel_size / pixel_size)
    print("... Resize")
    new_shape = (
        stitched_stack.shape[0] // ratio,
        stitched_stack.shape[1] // ratio,
    )
    log["new_shape"] = list(new_shape)

    log["downsample_ratio"] = ratio
    pixel_size *= ratio
    log["pixel_size"] = pixel_size

    print(f"   ..... resizing", flush=True)
    stitched_stack = cv2.resize(
        stitched_stack,
        new_shape[::-1],  # cv2 has (width, height), not (x, y)
        interpolation=cv2.INTER_CUBIC,
    )

    print(f"   ..... filtering", flush=True)

    stitched_stack = gaussian_filter(stitched_stack, sigma_blur)

    target = registration_folder / f"{chamber}_r{roi}_sl{slice_id:03d}.ome.tif"
    logfile = Path(target).with_suffix(".yml")
    print("Saving stitched image", flush=True)

    save_ome_tiff_pyramid(
        target,
        stitched_stack,
        pixel_size=pixel_size,
        subresolutions=subresolutions,
        save_thumbnail=False,
    )
    with open(logfile, "w") as fhandle:
        yaml.dump(log, fhandle)
