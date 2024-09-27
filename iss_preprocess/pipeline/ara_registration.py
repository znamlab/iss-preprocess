from pathlib import Path
import gc
from warnings import warn
import brainglobe_atlasapi as bga
import cv2
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter
from skimage.transform import downscale_local_mean
from image_tools.similarity_transforms import transform_image
from image_tools.registration.phase_correlation import phase_correlation
from znamutils import slurm_it

from ..io import (
    get_pixel_size,
    get_processed_path,
    get_roi_dimensions,
    load_metadata,
    load_ops,
    load_section_position,
    load_stack,
    save_ome_tiff_pyramid,
    write_stack,
)
from . import stitch


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
    chamber = int(Path(data_path).name.split("_")[1])
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
    processed_path = get_processed_path(data_path)
    reg_folder = processed_path / "register_to_ara"
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


def load_registration_reference(data_path, roi):
    """Load the registration reference image of one ROI

    This is the downsampled version of the overview image used for registration.

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI

    Returns:
        ref (np.ndarray): Registration reference image

    """
    processed_path = get_processed_path(data_path)
    reg_folder = processed_path / "register_to_ara"
    chamber = reg_folder.parent.name
    if not reg_folder.is_dir():
        raise IOError("Registration folder does not exists. Perform registration first")
    img_file = list(reg_folder.glob(f"{chamber}_r{roi}_sl*.tif"))
    if not len(img_file):
        raise IOError(f"No file found for ROI {roi}")
    elif len(img_file) > 1:
        raise IOError(f"Found multiple files for ROI {roi}")
    return load_stack(str(img_file[0]))


def load_coordinate_image(
    data_path, roi, full_scale=False, registered=True, return_fname=False
):
    """Load the 3 channel image of ARA coordinates for `roi`

    The reference atlas is first registered to a downsampled version of the overview,
    this is then registered to the normal acquisition. The coordinates of the overview
    can be loaded with `registered=False`.

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI
        full_scale (bool, optional): If true, returns the full scale image, otherwise
            the downsample version used for registration. Defaults to False.
        registered (bool, optional): If True, load the registered coordinates, otherwise
            the coordinates of the overview, before shifting/cropping. Defaults to True.
        return_fname (bool, optional): If True, return the filename of the image.
            Defaults to False.


    Returns:
        coords (np.ndarray): 3 channel image of ARA coordinates

    """
    processed_path = get_processed_path(data_path)
    reg_folder = processed_path / "register_to_ara"

    if not reg_folder.is_dir():
        raise IOError("Registration folder does not exists. Perform registration first")
    coord_folder = reg_folder / "ara_coordinates"
    if not coord_folder.is_dir():
        raise IOError(
            "ARA coordinates folder does not exists." + " Perform registration first"
        )
    if registered:
        coord_file = list(coord_folder.glob(f"*_r{roi}_*registered.tif"))
    else:
        coord_file = list(coord_folder.glob(f"*_r{roi}_*Coords.tif"))

    if not len(coord_file):
        raise IOError(f"Cannot find coordinates files for roi {roi}")
    elif len(coord_file) > 1:
        raise IOError(f"Found multiple coordinates files for roi {roi}")
    coord_file = coord_file[0]

    coords = load_stack(str(coord_file))
    if np.allclose(coords, 0):
        raise IOError(f"Coordinates image for roi {roi} is empty")

    if full_scale:
        metadata = load_registration_reference_metadata(data_path, roi)
        scale_factor = metadata["pixel_size"] / metadata["original_pixel_size"]
        out_shape = (np.array(coords.shape) * scale_factor).astype(int)
        out_shape[2] = coords.shape[2]  # no need to upscale last dimension
        out = np.zeros(
            out_shape,
            dtype=coords.dtype,
        )
        for i in range(coords.shape[2]):
            # cv2 resize takes (width, height) not (row, col)
            out[..., i] = cv2.resize(
                coords[..., i], out_shape[:2][::-1], cv2.INTER_LINEAR
            )
        coords = out
    if return_fname:
        return coords, coord_file
    return coords


def make_area_image(
    data_path, roi, atlas_size=10, full_scale=False, reload=True, registered=True
):
    """Generate an image with area ID in each pixel

    Args:
        data_path (str): Relative path to data
        roi (int): Roi number to generate
        atlas_size (int, optional): Pixel size of the atlas used to find area id.
            Defaults to 10.
        full_scale (bool, optional): If true, returns the full scale image, otherwise
            the downsample version used for registration. Defaults to False.
        reload (bool, optional): If True, reload the area image, otherwise recompute it.
            Valid only if full_scale is False. Defaults to True.
        registered (bool, optional): If True, load the registered coordinates, otherwise
            the coordinates of the overview, before shifting/cropping. Defaults to True.

    Returns:
        area_id (np.array): Image with area id of each pixel

    """
    if full_scale and reload:
        warn("Cannot reload full scale area image. Setting reload to False")
        reload = False
    if not registered and reload:
        warn("Cannot reload non-registered area image. Setting reload to False")
        reload = False
    save_folder = get_processed_path(data_path) / "register_to_ara" / "area_images"
    fname = save_folder / f"area_image_r{roi}_ara{atlas_size}.tif"
    if reload:
        if fname.is_file():
            area_id = load_stack(str(fname))[..., 0]
            return area_id

    coord = np.clip(
        load_coordinate_image(
            data_path, roi, full_scale=full_scale, registered=registered
        ),
        0,
        None,
    )
    coord = np.round(coord * 1000 / atlas_size, 0).astype("uint16")

    atlas_name = "allen_mouse_%dum" % atlas_size
    bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
    for axis, max_val in enumerate(bg_atlas.shape):
        coord[:, :, axis] = np.clip(coord[:, :, axis], 0, max_val - 1)
    area_id = bg_atlas.annotation[coord[:, :, 0], coord[:, :, 1], coord[:, :, 2]]
    if (not full_scale) and registered:
        save_folder.mkdir(exist_ok=True)
        write_stack(area_id, str(fname), bigtiff=True, dtype=area_id.dtype)

    return area_id


def spots_ara_infos(
    data_path,
    spots,
    roi,
    atlas_size=10,
    acronyms=True,
    inplace=True,
    full_scale_coordinates=False,
    reload=True,
    verbose=True,
):
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
        full_scale_coordinates (bool, optional): If true, use the full scale image to
            find coordinates, otherwise the downsample version used for registration.
            Defaults to False.
        reload (bool, optional): If True, reload the area image, otherwise recompute it.
            Valid only if full_scale is False. Defaults to True.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        spots (pd.DataFrame): reference or copy of spots dataframe with four more
            columns: `ara_x`, `ara_y`, `ara_z`, and `area_id`

    """
    if not inplace:
        spots = spots.copy()
    if verbose:
        print("Loading coordinates and area id", flush=True)
    coords = load_coordinate_image(data_path, roi, full_scale=full_scale_coordinates)
    metadata = load_registration_reference_metadata(data_path, roi)
    spot_xy = spots.loc[:, ["x", "y"]].values

    if not full_scale_coordinates:
        spot_xy = spots.loc[:, ["x", "y"]].values / metadata["downsample_ratio"]
    spot_xy = np.round(spot_xy).astype(int)
    if verbose:
        print("Attributing coordinates to spots", flush=True)
    # Filter spots that are outside the valid range
    valid_x = spot_xy[:, 0] < coords.shape[1]
    valid_y = spot_xy[:, 1] < coords.shape[0]
    valid_spots = valid_x & valid_y
    spots_filtered = spots.iloc[valid_spots].copy()
    spot_xy_filtered = spot_xy[valid_spots]

    # Use filtered coordinates for further processing
    spot_coords = coords[spot_xy_filtered[:, 1], spot_xy_filtered[:, 0], :]
    del coords
    gc.collect()
    for i, w in enumerate("xyz"):
        spots_filtered[f"ara_{w}"] = spot_coords[:, i]

    if verbose:
        print("Loading area map", flush=True)
    # we will always load non full scale area map as it takes huge amount of RAM
    area_map = make_area_image(
        data_path, roi, atlas_size=atlas_size, full_scale=False, reload=reload
    )
    if verbose:
        print("Attributing area id to spots", flush=True)

    if full_scale_coordinates:
        # but that means we need to downsample the spots coordinates if it hasn't been
        # done before
        spot_xy_filtered = np.round(
            spot_xy_filtered / metadata["downsample_ratio"]
        ).astype(int)
    # and clip in case of rounding errors
    spot_xy_filtered[:, 0] = np.clip(spot_xy_filtered[:, 0], 0, area_map.shape[1] - 1)
    spot_xy_filtered[:, 1] = np.clip(spot_xy_filtered[:, 1], 0, area_map.shape[0] - 1)
    spot_area = area_map[spot_xy_filtered[:, 1], spot_xy_filtered[:, 0]]
    spots_filtered["area_id"] = spot_area
    del area_map
    gc.collect()
    if acronyms:
        if verbose:
            print("Finding area acronyms", flush=True)
        atlas_name = "allen_mouse_%dum" % atlas_size
        bg_atlas = bga.bg_atlas.BrainGlobeAtlas(atlas_name)
        labels = bg_atlas.lookup_df.set_index("id")
        spots_filtered["area_acronym"] = "outside"
        valid = spots_filtered.area_id != 0
        spots_filtered.loc[valid, "area_acronym"] = labels.loc[
            spots_filtered.area_id[valid], "acronym"
        ].values
    if verbose:
        print("Done", flush=True)
    return spots_filtered


def overview_single_roi(
    data_path,
    roi,
    slice_id,
    prefix,
    chan2use=(0, 1, 2, 3),
    sigma_blur=10,
    agg_func=np.nanmean,
    ref_prefix="genes_round",
    subresolutions=5,
    max_pixel_size=2,
    non_similar_overview=False,
):
    """Stitch and save a single ROI overview for use in atlas registration

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI
        slice_id (int): Slice number to stitch
        prefix (str, optional): Prefix of the acquisition to plot.
        chan2use (tuple, optional): Channels to use for stitching. Defaults to (0, 1, 2, 3).
        sigma_blur (int, optional): Sigma for gaussian blur. Defaults to 10.
        agg_func (function, optional): Aggregation function to apply across channels.
            Defaults to np.nanmean. Unused if `non_similar_overview` is True.
        ref_prefix (str, optional): Prefix of the reference image. Defaults to
            "genes_round".
        subresolutions (int, optional): Number of subresolutions to save. Defaults to 5.
        max_pixel_size (int, optional): Maximum pixel size for the pyramid. Defaults to 2.
        non_similar_overview (bool, optional): If True, stitch the overview tiles with
        the stitch_tiles function rather than stitch_registered which requires tile by tile
        registration to the reference. Defaults to False.

    """
    print(f"Data path: {data_path}")
    print(f"Roi: {roi}", flush=True)
    print(f"Slice id: {slice_id}", flush=True)
    print(f"Sigma blur: {sigma_blur}", flush=True)
    sigma_blur = float(sigma_blur)
    chamber = Path(data_path).name
    registration_folder = get_processed_path(data_path) / "register_to_ara"
    registration_folder.mkdir(exist_ok=True)

    print("Finding shifts")
    ops = load_ops(data_path)

    print("Finding pixel size")
    if ref_prefix == "genes_round":
        ref_round_prefix = f"genes_round_{ops['ref_round']}_1"
    else:
        ref_round_prefix = ref_prefix
    pixel_size = get_pixel_size(data_path, ref_round_prefix)

    if chan2use is None:
        chan2use = [ops["ref_ch"]]
    if isinstance(chan2use, int):
        chan2use = [chan2use]
    chan2use = [int(c) for c in chan2use]

    if non_similar_overview:
        print("Stitching ROI")
        stitched_stack = stitch.stitch_tiles(
            data_path=data_path,
            prefix=prefix,
            roi=roi,
            ich=chan2use,
            correct_illumination=True,
            shifts_prefix=ref_prefix,
        )
    else:
        print("Stitching ROI")
        stitched_stack = stitch.stitch_registered(
            data_path=data_path,
            prefix=prefix,
            roi=roi,
            filter_r=False,
            channels=chan2use,
            ref_prefix=ref_prefix,
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

    print("   ..... resizing", flush=True)
    stitched_stack = cv2.resize(
        stitched_stack,
        new_shape[::-1],  # cv2 has (width, height), not (row, col)
        interpolation=cv2.INTER_CUBIC,
    )

    print("   ..... filtering", flush=True)

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


def crop_overview_registration(data_path, rois=None, overview_prefix="DAPI_1_1"):
    """Crop the registered overview to the same size as the reference

    Args:
        data_path (str): Relative path to data
        rois (list, optional): List of rois to crop. Defaults to None.
        overview_prefix (str, optional): Prefix of the overview image. Defaults to
            "DAPI_1_1".

    Returns:
        imgs (list): List of cropped images
    """

    if rois is None:
        rois = get_roi_dimensions(data_path)[:, 0]
    elif isinstance(rois, int):
        rois = [rois]

    processed_path = get_processed_path(data_path)
    imgs = []
    for roi in rois:
        shifts = np.load(
            processed_path / "reg" / f"{overview_prefix}_roi{roi}_tform_to_ref.npz"
        )

        # account for downsampling
        metadata = load_registration_reference_metadata(data_path, roi)
        ara_downsample_rate = metadata["downsample_ratio"]
        shifts = shifts["shift"] / ara_downsample_rate
        ara_im, ara_coord_file = load_coordinate_image(
            data_path, roi, full_scale=False, registered=False, return_fname=True
        )

        # Do the shift
        target_shifted = transform_image(ara_im, scale=1, angle=0, shift=shifts)

        # Crop to the same size as the reference
        ops = load_ops(data_path)
        corners = stitch.get_tile_corners(
            data_path, prefix=ops["reference_prefix"], roi=roi
        )
        top_right_corner = np.nanmax(corners, axis=(0, 1, 3)) / ara_downsample_rate
        top_right_corner = np.round(top_right_corner).astype(int) + 1
        cropped_image = target_shifted[: top_right_corner[0], : top_right_corner[1]]

        # save the cropped image
        new_name = ara_coord_file.name.replace("Coords", "registered")
        target = ara_coord_file.with_name(new_name)
        write_stack(cropped_image, target, dtype="float32", clip=False)
        print(f"Saved {target}")
        imgs.append(cropped_image)
    return imgs


@slurm_it(conda_env="iss-preprocess", slurm_options={"mem": "32GB"})
def register_overview_to_reference(data_path, roi, channel, overview_prefix="DAPI_1_1"):
    """Register the overview to the reference image

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI
        channel (int): Channel to use for registration.
        downsample (int, optional): Downsample factor. Defaults to 3.
        overview_prefix (str, optional): Prefix of the overview image. Defaults to
            "DAPI_1_1".

    Returns:
        shift (np.ndarray): Shift in x and y
        final_shape (tuple): Final shape of the stitched images
        stitched_fixed (np.ndarray): Stitched reference image
        stitched_moving (np.ndarray): Stitched overview image
    """
    print(f"Registering overview to reference for roi {roi} for {data_path}")
    print("")
    print(f"Using channel {channel}")
    print(f"Overview prefix: {overview_prefix}")
    print("")

    ops = load_ops(data_path)
    reference_prefix = ops["reference_prefix"]
    projection = ops[f"{reference_prefix.split('_')[0].lower()}_projection"]
    if channel is None:
        channel = ops["ref_ch"]

    # for the overview, use the version that was used for the manual registration
    stitched_moving = load_registration_reference(data_path, roi)[..., 0]

    # Stitched reference image
    stitched_fixed = stitch.stitch_tiles(
        data_path,
        reference_prefix,
        suffix=projection,
        roi=roi,
        ich=channel,
        shifts_prefix=reference_prefix,
        correct_illumination=True,
        allow_quick_estimate=False,
    ).astype(
        np.single
    )  # to save memory
    # downsample to match the version used for the manual registration
    metadata = load_registration_reference_metadata(data_path, roi)
    ara_downsample_ratio = metadata["downsample_ratio"]
    print(f"Downsample ratio: {ara_downsample_ratio}")
    stitched_fixed = downscale_local_mean(stitched_fixed, ara_downsample_ratio)

    shapes = np.vstack([stitched_fixed.shape, stitched_moving.shape])
    final_shape = shapes.max(axis=0)
    paddings = final_shape - shapes
    if paddings[1].sum() > 0:
        raise NotImplementedError("Paddings in the moving image. Check shift")
    stitched_fixed = np.pad(
        stitched_fixed, [(p // 2, p // 2 + p % 2) for p in paddings[0]]
    )
    stitched_moving = np.pad(
        stitched_moving, [(p // 2, p // 2 + p % 2) for p in paddings[1]]
    )
    print(f"Shapes: {shapes}")
    print(f"Final shape: {final_shape}")
    print(f"Paddings: {paddings}")

    def prep_stack(stack):
        if stack.dtype != bool:
            mi, ma = np.nanpercentile(stack[stack > 0], (1, 99))
            stack = np.clip(stack, mi, ma)
            stack = (stack - mi) / (ma - mi)
        return stack

    shift, _, _, _ = phase_correlation(
        max_shift=None,
        min_shift=None,
        fixed_image=prep_stack(stitched_fixed),
        moving_image=prep_stack(stitched_moving),
    )
    print(f"Downsampled shift: {shift}")
    # remove whatever padding was added to the fixed image
    shift -= paddings[0] // 2
    print(f"Padding corrected shifts: {shift}")
    scale = 1
    angle = 0
    shift *= ara_downsample_ratio
    print(f"Final shift: {shift}")
    fname = f"{overview_prefix}_roi{roi}_tform_to_ref.npz"
    print(f"Saving {fname} in the reg folder")
    np.savez(
        get_processed_path(data_path) / "reg" / fname,
        angle=angle,
        shift=shift,
        scale=scale,
        stitched_stack_shape=final_shape,
    )
    print(f"DONE")
    return shift, final_shape, stitched_fixed, stitched_moving


@slurm_it(conda_env="iss-preprocess", slurm_options={"time": "3:00:00", "mem": "16GB"})
def check_reg(data_path, save_folder, rois=None):
    import matplotlib.pyplot as plt

    if rois is None:
        rois = get_roi_dimensions(data_path)[:, 0]
    elif isinstance(rois, int):
        rois = [rois]

    fig = plt.figure(figsize=(10, 7))
    crop_overview_registration(data_path, rois=None, overview_prefix="DAPI_1_1")
    ops = load_ops(data_path)
    nrows = 2 if len(rois) > 5 else 1
    ncols = len(rois) // 2
    for roi in rois:
        metadata = load_registration_reference_metadata(data_path, roi)
        ara_downsample_rate = metadata["downsample_ratio"]

        print(f"roi {roi}")
        ax = fig.add_subplot(nrows, ncols, roi)
        anchor_ref = stitch.stitch_registered(
            data_path,
            prefix=ops["reference_prefix"],
            roi=roi,
            channels=[3],
        )
        anchor_ref = downscale_local_mean(anchor_ref[..., 0], ara_downsample_rate)
        vm = np.nanpercentile(anchor_ref, 99)
        ax.imshow(anchor_ref, cmap="gray", vmin=0, vmax=vm)
        ax.set_title(f"roi {roi}")
        area_img = make_area_image(
            data_path, roi, atlas_size=10, full_scale=False, reload=False
        )
        area_img = area_img.astype(float)
        area_img[area_img == 0] = np.nan
        ax.imshow(area_img % 20, cmap="tab20", alpha=0.5)
        bin = area_img > 0
        ax.contour(bin, colors="purple", levels=[0.5], linewidths=1, alpha=0.2)
        ax.axis("off")
    fig.tight_layout()
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)
    chamber = Path(data_path).name
    fig.savefig(save_folder / f"{chamber}_overview_registration.png")
