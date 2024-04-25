import warnings
from os import system
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure
from skimage.filters import threshold_triangle
from skimage.segmentation import expand_labels
from tqdm import tqdm

import iss_preprocess as iss

from ..io import get_pixel_size, get_roi_dimensions, load_metadata, load_ops
from ..segment import cellpose_segmentation, count_spots, spot_mask_value
from . import ara_registration as ara_reg
from . import diagnostics
from .stitch import stitch_registered


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
            args = args + " --partition=ncpu"
        args = (
            args + f" --output={Path.home()}/slurm_logs/{data_path}/iss_segment_%j.out"
        )

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
    ops = load_ops(data_path)
    print(f"stitching {prefix} and aligning to {reference}", flush=True)
    stitched_stack = stitch_registered(
        data_path,
        ref_prefix=reference,
        prefix=prefix,
        roi=iroi,
        channels=ops["segmentation_channels"],
    )
    if stitched_stack.ndim == 3:
        stitched_stack = np.nanmean(stitched_stack, axis=-1)

    print("starting segmentation", flush=True)
    masks = cellpose_segmentation(
        stitched_stack[..., 0],
        channels=ops["segmentation_channels"],
        flow_threshold=ops["cellpose_flow_threshold"],
        min_pix=0,
        dilate_pix=0,
        rescale=ops["cellpose_rescale"],
        model_type=ops["cellpose_model"],
        use_gpu=use_gpu,
    )
    np.save(iss.io.get_processed_path(data_path) / f"masks_{iroi}.npy", masks)
    diagnostics.check_segmentation(
        data_path, iroi, prefix, reference, stitched_stack, masks
    )


def make_cell_dataframe(data_path, roi, masks=None, mask_expansion=5.0, atlas_size=10):
    """Make cell dataframe

    The index will be the mask ID.
    The dataframe will include, for each cell, their centroid, bounding box, and area.
    If atlas_size is not None, it will also include the ID and acronym of the atlas
    area where their centroid is located.

    Args:
        data_path (str): Relative path to data
        roi (int): Number of the ROI to process
        masks (np.array, optional): Array of labels, if None will load `masks_{roi}.npy`
            from the reg folder. Defaults to None.
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.
        atlas_size (int, optional): Size of the atlas to use to load ARA information.
            If None, will not get area information. Defaults to 10.

    """
    big_masks = _get_big_masks(data_path, roi, masks, mask_expansion)

    cell_df = pd.DataFrame(
        measure.regionprops_table(
            big_masks, properties=("label", "centroid", "area", "bbox")
        )
    )
    cell_df.set_index("label", drop=False, inplace=True)
    bbox = ("ymin", "xcol", "ymax", "xmax")
    new_names = {f"bbox-{i}": col_name for i, col_name in enumerate(bbox)}
    new_names.update({"centroid-0": "y", "centroid-1": "x"})
    cell_df.rename(columns=new_names, inplace=True)
    cell_df["roi"] = roi

    # TODO: add coordinate in tile

    if atlas_size is not None:
        ara_reg.spots_ara_infos(
            data_path,
            spots=cell_df,
            atlas_size=atlas_size,
            roi=roi,
            acronyms=True,
            inplace=True,
        )
    cell_folder = iss.io.get_processed_path(data_path) / "cells"
    cell_folder.mkdir(exist_ok=True)
    cell_df.to_pickle(cell_folder / f"cells_df_roi{roi}.pkl")
    return cell_df


def add_mask_id(
    data_path,
    roi,
    mask_expansion=5.0,
    masks=None,
    barcode_dot_threshold=0.15,
    spot_score_threshold=0.1,
    hyb_score_threshold=0.8,
):
    """Load gene, barcode, and hybridisation spots and add a mask_id column to each spots dataframe

    Args:
        data_path (str): Relative path to data
        roi (int): ID of the ROI to load
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.
        masks (np.array, optional): Array of labels. If None will load "masks_{roi}".
             Defaults to None.
        barcode_dot_threshold (float, optional): Threshold for the barcode dot product.
            Only spots above the threshold will be counted. Defaults to 0.15.
        spot_score_threshold (float, optional): Threshold for the OMP score. Only spots
            above the threshold will be counted. Defaults to 0.1.
        hyb_score_threshold (float, optional): Threshold for hybridisation spots. Only
            spots above the threshold will be counted. Defaults to 0.8.

    Returns:
        dict: Dictionary of spots dataframes

    """
    processed_path = iss.io.get_processed_path(data_path)
    big_masks = _get_big_masks(data_path, roi, masks, mask_expansion)

    metadata = load_metadata(data_path=data_path)
    spot_acquisitions = ["genes_round", "barcode_round"]
    thresholds = dict(
        genes_round=("spot_score", spot_score_threshold),
        barcode_round=("dot_product_score", barcode_dot_threshold),
    )
    for hyb in metadata["hybridisation"]:
        spot_acquisitions.append(hyb)
        thresholds[hyb] = ("score", hyb_score_threshold)

    # get the spots dataframes
    spots_dict = dict()
    for prefix in spot_acquisitions:
        print(f"Loading {prefix}", flush=True)
        spot_df = pd.read_pickle(processed_path / f"{prefix}_spots_{roi}.pkl")
        filt_col, threshold = thresholds[prefix]
        spot_df = spot_df[spot_df[filt_col] > threshold]
        # modify spots in place
        spots_dict[prefix] = spot_mask_value(big_masks, spot_df)
    return spots_dict


def segment_spots(
    data_path,
    roi,
    mask_expansion=5.0,
    masks=None,
    barcode_dot_threshold=0.15,
    spot_score_threshold=0.1,
    hyb_score_threshold=0.8,
):
    """Count number of rolonies per cell for barcodes and genes.

    Only rolonies above the relevant threshold will be counted. (Note that genes
    rolonies are already thresholded once after OMP).

    Hybridisation and sequencing datasets will be fused.

    Outputs are saved in the `cells` folder as f"genes_df_roi{roi}.pkl" and
    f"barcode_df_roi{roi}.pkl"

    Args:
        data_path (str): Relative path to data
        roi (int): ID of the ROI to load
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.
        masks (np.array, optional): Array of labels. If None will load "masks_{roi}".
             Defaults to None.
        barcode_dot_threshold (float, optional): Threshold for the barcode dot product.
            Only spots above the threshold will be counted. Defaults to 0.15.
        spot_score_threshold (float, optional): Threshold for the OMP score. Only spots
            above the threshold will be counted. Defaults to 0.1.
        hyb_score_threshold (float, optional): Threshold for hybridisation spots. Only
            spots above the threshold will be counted. Defaults to 0.8.

    Returns:
        barcode_df (pd.DataFrame): Count of rolonies per barcode sequence per cell.
            Index is the mask ID of the cell
        fused_df (pd.DataFrame): Count of rolonies per genes or hybridisation probe per
            cell. Index is the mask ID of the cell

    """
    # add the mask_id column to spots_df
    spots_dict = add_mask_id(
        data_path,
        roi=roi,
        mask_expansion=mask_expansion,
        masks=masks,
        barcode_dot_threshold=barcode_dot_threshold,
        spot_score_threshold=spot_score_threshold,
        hyb_score_threshold=hyb_score_threshold,
    )

    thresholds = dict(
        genes_round=("spot_score", spot_score_threshold),
        barcode_round=("dot_product_score", barcode_dot_threshold),
    )
    for hyb in spots_dict:
        if hyb in thresholds:
            # it is genes or barcode
            continue
        thresholds[hyb] = ("score", hyb_score_threshold)

    # get the spots dataframes
    spots_in_cells = dict()
    for prefix, spot_df in spots_dict.items():
        print(f"Doing {prefix}", flush=True)
        grouping_column = "bases" if prefix.startswith("barcode") else "gene"
        cell_df = count_spots(spots=spot_df, grouping_column=grouping_column)
        spots_in_cells[prefix] = cell_df

    # Save barcodes
    barcode_df = spots_in_cells.pop("barcode_round")
    save_dir = iss.io.get_processed_path(data_path) / "cells"
    save_dir.mkdir(exist_ok=True)
    barcode_df.to_pickle(save_dir / f"barcode_df_roi{roi}.pkl")

    # Fuse genes and hybridisation
    fused_df = spots_in_cells.pop("genes_round")
    for hyb, hyb_df in spots_in_cells.items():
        for gene in hyb_df.columns:
            if gene in fused_df.columns:
                warnings.warn(f"Replacing {gene} with hybridisation")
                fused_df.pop(gene)
        fused_df = fused_df.join(hyb_df, how="outer")
    fused_df[np.isnan(fused_df)] = 0
    fused_df = fused_df.astype(int)
    fused_df.to_pickle(save_dir / f"genes_df_roi{roi}.pkl")

    return barcode_df, fused_df


def _get_big_masks(data_path, roi, masks, mask_expansion):
    """Small internal function to avoid code duplication

    Reload and expand masks if needed

    Args:
        data_path (str): Relative path to data
        roi (int): ID of the ROI to load
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.
        masks (np.array, optional): Array of labels. If None will load "masks_{roi}".
             Defaults to None.

    Returns:
        numpy.ndarray: masks expanded

    """
    if masks is None:
        masks = np.load(iss.io.get_processed_path(data_path) / f"masks_{roi}.npy")
    if mask_expansion is None:
        big_masks = masks
    else:
        pixel_size = get_pixel_size(data_path, prefix="genes_round_1_1")
        big_masks = expand_labels(masks, distance=int(mask_expansion / pixel_size))
    return big_masks


def segment_mcherry_tile(
    data_path,
    prefix,
    roi,
    tilex,
    tiley,
    suffix="unmixed",
    r1=4,
    r2=70,
    area_threshold=200,
    elongation_threshold=0.9,
    circularity_threshold=0.5,
):
    """
    Segment the mCherry channel of an image stack.

    Args:
        data_path (str): Path to the data directory.
        prefix (str): Prefix of the image stack.
        roi (int): Region of interest.
        tilex (int): X coordinate of the tile.
        tiley (int): Y coordinate of the tile.
        suffix (str): Suffix of the image stack.
        r1 (int): Inner radius of the annulus.
        r2 (int): Outer radius of the annulus.
        area_threshold (int): Minimum area of a cell.
        elongation_threshold (float): Maximum eccentricity of a cell.
        circularity_threshold (float): Minimum circularity of a cell.

    Returns:
        filtered_masks (np.ndarray): Binary image of the filtered masks.
        filtered_df (pd.DataFrame): DataFrame of the filtered masks.
        rejected_masks (np.ndarray): Binary image of the rejected masks.
    """

    # Load the unmixed and original mCherry image stacks
    processed_path = iss.io.get_processed_path(data_path)
    unmixed_fname = (
        f"{prefix}_MMStack_{roi}-"
        + f"Pos{str(tilex).zfill(3)}_{str(tiley).zfill(3)}_unmixed.tif"
    )
    unmixed_path = processed_path / prefix / unmixed_fname
    unmixed_stack = iss.io.load_stack(unmixed_path)

    original_fname = (
        f"{prefix}_MMStack_{roi}-"
        + f"Pos{str(tilex).zfill(3)}_{str(tiley).zfill(3)}_{suffix}.tif"
    )
    stack = iss.io.load_stack(processed_path / prefix / original_fname)

    # Apply a hann window filter to the unmixed image to remove halos around cells
    filt = iss.image.filter_stack(unmixed_stack, r1=r1, r2=r2, dtype=float)
    binary = (filt > threshold_triangle(filt))[:, :, 0]

    # Label the connected components in the binary image
    # creating a df with the properties of each cell
    labeled_image = measure.label(binary)
    props = measure.regionprops_table(
        labeled_image,
        intensity_image=stack,
        properties=(
            "label",
            "area",
            "centroid",
            "eccentricity",
            "major_axis_length",
            "minor_axis_length",
            "intensity_max",
            "intensity_mean",
            "intensity_min",
            "perimeter",
            "solidity",
        ),
    )

    props_df = pd.DataFrame(props)
    props_df["circularity"] = (
        4 * np.pi * props_df["area"] / (props_df["perimeter"] ** 2)
    )
    props_df["intensity_ratio"] = (
        props_df["intensity_mean-2"] / props_df["intensity_mean-3"]
    )
    props_df["roi"] = roi
    props_df["tilex"] = tilex
    props_df["tiley"] = tiley

    area_threshold = area_threshold
    # Closer to 1 means more elongated
    elongation_threshold = elongation_threshold
    # Closer to 1 means more circular
    circularity_threshold = circularity_threshold

    filtered_df = props_df[
        (props_df["area"] > area_threshold)
        & (props_df["eccentricity"] <= elongation_threshold)
        & (props_df["circularity"] >= circularity_threshold)
    ]

    rejected_masks_df = props_df[
        ~(
            (props_df["area"] > area_threshold)
            & (props_df["eccentricity"] <= elongation_threshold)
            & (props_df["circularity"] >= circularity_threshold)
        )
    ]

    # Identify all pixels belonging to the filtered labels
    filtered_masks = np.zeros_like(labeled_image, dtype=np.uint16)
    filtered_labels = filtered_df["label"].to_list()
    for label in filtered_labels:
        mask = labeled_image == label
        filtered_masks[mask] = label

    rejected_masks = np.zeros_like(labeled_image, dtype=np.uint8)
    rejected_labels = rejected_masks_df["label"].to_list()
    rejected_mask = np.isin(labeled_image, rejected_labels)
    rejected_masks[rejected_mask] = 255

    mask_dir = processed_path / "cells"
    mask_dir.mkdir(exist_ok=True)
    np.save(
        mask_dir / f"{prefix}_masks_{roi}_{tilex}_{tiley}.npy",
        filtered_masks,
        allow_pickle=True,
    )
    pd.to_pickle(filtered_df, mask_dir / f"{prefix}_df_{roi}_{tilex}_{tiley}.pkl")

    return filtered_masks, filtered_df, rejected_masks


def load_mask_by_coors(
    data_path,
    prefix="mCherry",
    tile_coors=(1, 0, 0),
    load_raw=False,
):
    """Load processed tile images

    Args:
        data_path (str): relative path to dataset.
        prefix (str, optional): Full folder name prefix, including round number.
            Defaults to "mCherry"
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos.
            Defaults to (1,0,0).

    Returns:
        numpy.ndarray: X x Y x channels stack.

    """
    processed_path = iss.io.get_processed_path(data_path)
    tile_roi, tile_x, tile_y = tile_coors
    fname_corrected = f"{prefix}_masks_corrected_{tile_roi}_{tile_x}_{tile_y}.npy"
    fname = f"{prefix}_masks_{tile_roi}_{tile_x}_{tile_y}.npy"
    if load_raw:
        if (processed_path / "cells" / fname).exists():
            masks = np.load(processed_path / "cells" / fname, allow_pickle=True)
        else:
            raise FileNotFoundError(f"Could not find mask file {fname}")
        return masks

    if (processed_path / "cells" / fname_corrected).exists():
        masks = np.load(processed_path / "cells" / fname_corrected, allow_pickle=True)
    elif (processed_path / "cells" / fname).exists():
        masks = np.load(processed_path / "cells" / fname, allow_pickle=True)
    else:
        raise FileNotFoundError(f"Could not find mask file {fname}")

    return masks


def find_edge_touching_masks(masks):
    """
    Finds masks that touch the edge of the image.

    Args:
        masks (np.ndarray): The binary or labeled mask array where each cell is represented
                            by a unique integer, and background is 0.

    Returns:
        edge_touching_labels (list): A list of unique labels that touch the edge of the image.
    """
    # Initialize an empty list to store labels of masks that touch the edge
    edge_touching_labels = []

    # Get the unique labels in the masks excluding the background (0)
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels != 0]

    # Check each label to see if it touches the edge
    for label in unique_labels:
        # Find the mask for the current label
        mask = masks == label

        # Check if the mask touches the top or bottom edge
        if np.any(mask[0, :]) or np.any(mask[-1, :]):
            edge_touching_labels.append(label)
            continue  # Skip to the next label as this one already touches an edge

        # Check if the mask touches the left or right edge
        if np.any(mask[:, 0]) or np.any(mask[:, -1]):
            edge_touching_labels.append(label)

    # Set all masks that touch the edge to 0
    for label in edge_touching_labels:
        masks[masks == label] = 0
    return masks, edge_touching_labels


def get_overlap_regions(
    tile_shape, shifts, tile_ref, tile_right, tile_down, tile_down_right
):
    """
    Determine the coordinates of the overlap region between two adjacent tiles using explicit tile direction.

    Args:
        tile_shape (tuple): The shape (height, width) of the tile.
        shifts (dict): The dictionary containing the shift values for the down and right tiles.
        tile_ref (np.ndarray): The reference tile.
        tile_right (np.ndarray): The right tile.
        tile_down (np.ndarray): The down tile.
        tile_down_right (np.ndarray): The down right tile.

    Returns:
        overlap_ref_vert (np.ndarray): The overlap region between the reference tile and the down tile.
        overlap_down (np.ndarray): The overlap region between the down tile and the reference tile.
        overlap_ref_side (np.ndarray): The overlap region between the reference tile and the right tile.
        overlap_right (np.ndarray): The overlap region between the right tile and the reference tile.
        overlap_ref_with_down_right (np.ndarray): The overlap region between the reference tile and the down right tile.
        overlap_down_right_with_ref (np.ndarray): The overlap region between the down right tile and the reference tile.

    """

    width, height = tile_shape
    shift_down = shifts["shift_down"]
    shift_right = shifts["shift_right"]

    # Get the overlap regions between the reference tile and the down tile
    overlap_ref_vert = tile_ref[
        : (width - int(shift_down[0])), : (height - int(shift_down[1]))
    ]
    overlap_down = tile_down[int(shift_down[0]) :, int(shift_down[1]) :]

    # Get the overlap regions between the reference tile and the right tile
    overlap_ref_side = tile_ref[
        : (-int(shift_right[0])), -(height + int(shift_right[1])) :
    ]
    overlap_right = tile_right[int(shift_right[0]) :, : (height + int(shift_right[1]))]

    # Get the overlap regions between the reference tile and the down right tile
    overlap_ref_with_down_right = tile_ref[
        : (width - int(shift_down[0])), -(height + int(shift_right[1])) :
    ]
    overlap_down_right_with_ref = tile_down_right[
        int(shift_down[0]) :, : (height + int(shift_right[1]))
    ]

    return (
        overlap_ref_vert,
        overlap_down,
        overlap_ref_side,
        overlap_right,
        overlap_ref_with_down_right,
        overlap_down_right_with_ref,
    )


def remove_overlapping_labels(overlap_ref, overlap_shifted, upper_overlap_thresh):
    """
    Dynamically identifies and removes overlapping labels in two adjacent tile overlaps
    based on upper and lower overlap percentage thresholds. If the overlap is above the upper
    threshold, the label in the shifted tile is removed. If the overlap is below
    the lower threshold, the shared region is removed from both masks.

    Args:
        overlap_ref (np.array): The overlap area of the reference tile.
        overlap_shifted (np.array): The overlap area of the shifted tile.
        upper_overlap_thresh (float): The upper threshold percentage for considering mask overlap significant.

    Returns:
        overlapping_pairs (list): A list of tuples containing the labels that overlapped and their respective percentages.
    """

    # Continuously process until no more labels meet the criteria for adjustments
    labels_changed = True
    overlapping_pairs = []
    while labels_changed:
        labels_changed = False  # Reset flag for this iteration
        unique_labels_1 = np.unique(overlap_ref[overlap_ref != 0])
        unique_labels_2 = np.unique(overlap_shifted[overlap_shifted != 0])

        for label1 in unique_labels_1:
            mask1 = overlap_ref == label1
            for label2 in unique_labels_2:
                mask2 = overlap_shifted == label2
                if np.any(mask1 & mask2):  # Check if there's any overlap
                    overlap_area = np.sum(mask1 & mask2)
                    percent_overlap_tile1 = (overlap_area / np.sum(mask1)) * 100
                    percent_overlap_tile2 = (overlap_area / np.sum(mask2)) * 100
                    overlapping_pairs.append(
                        (label1, label2, (percent_overlap_tile1, percent_overlap_tile2))
                    )
                    # If overlap is above the upper threshold, remove the label from the shifted tile
                    if (
                        percent_overlap_tile1 > upper_overlap_thresh
                        or percent_overlap_tile2 > upper_overlap_thresh
                    ):
                        print(f"Removing label {label2} from shifted_tile")
                        overlap_shifted[mask2] = 0
                        labels_changed = True
                    # If overlap is below the lower threshold, remove the shared region from both masks
                    elif (
                        percent_overlap_tile1 > upper_overlap_thresh
                        and percent_overlap_tile2 > upper_overlap_thresh
                    ):
                        print(
                            f"Removing shared region from labels {label1} and {label2}"
                        )
                        shared_mask = mask1 & mask2
                        overlap_ref[shared_mask] = 0
                        overlap_shifted[shared_mask] = 0
                        labels_changed = True

                    if labels_changed:
                        break  # Exit the inner loop to reevaluate conditions due to the changes
            if labels_changed:
                break  # Exit the outer loop to restart the evaluation with updated overlaps
    return overlapping_pairs


def remove_all_overlapping_masks(data_path, prefix, upper_overlap_thresh):
    processed_path = iss.io.get_processed_path(data_path)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix)
    ops = iss.io.load_ops(data_path)

    # Remove all old files with "masks_corrected" in the name
    import glob

    for f in glob.glob(str(processed_path / "cells" / f"{prefix}_masks_corrected*")):
        Path(f).unlink()

    # First remove masks at the edges of all the tiles
    for roi in roi_dims[:, 0]:
        for tilex in tqdm(
            range(roi_dims[roi - 1, 1]),
            desc=f"ROI {roi} X-axis",
            total=roi_dims[roi - 1, 1],
        ):
            for tiley in tqdm(
                range(roi_dims[roi - 1, 2]),
                desc=f"Tile {tilex} Y-axis",
                leave=False,
                total=roi_dims[roi - 1, 2],
            ):
                coors = (roi, tilex, tiley)
                tile = load_mask_by_coors(
                    data_path,
                    tile_coors=coors,
                    prefix=prefix,
                    load_raw=True,
                )
                corrected_masks, _ = find_edge_touching_masks(tile)
                # Save the edge corrected masks
                fname = f"{prefix}_masks_corrected_{roi}_{tilex}_{tiley}.npy"
                np.save(
                    processed_path / "cells" / fname, corrected_masks, allow_pickle=True
                )

    # Now remove overlapping masks
    all_overlapping_pairs = []
    for roi in roi_dims[:, 0]:
        for tilex in tqdm(
            reversed(range(roi_dims[roi - 1, 1])),
            desc=f"ROI {roi} X-axis (overlap check)",
            total=roi_dims[roi - 1, 1],
        ):
            for tiley in tqdm(
                reversed(range(roi_dims[roi - 1, 2])),
                desc=f"Tile {tilex} Y-axis (overlap check)",
                leave=False,
                total=roi_dims[roi - 1, 2],
            ):
                ref_coors = (roi, tilex, tiley)

                # Load reference tile
                tile_ref = load_mask_by_coors(
                    data_path, tile_coors=ref_coors, prefix=prefix
                )

                # Check if adjacent down tile exists and load it
                down_offset = 1 if ops["y_tile_direction"] == "bottom_to_top" else -1
                down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + down_offset)
                if down_coors[2] < 0:
                    tile_down = np.zeros_like(tile_ref)
                else:
                    tile_down = load_mask_by_coors(
                        data_path, tile_coors=down_coors, prefix=prefix
                    )

                # Check if adjacent right tile exists and load it
                right_offset = 1 if ops["x_tile_direction"] == "left_to_right" else -1
                right_coors = (ref_coors[0], ref_coors[1] + right_offset, ref_coors[2])
                if right_coors[1] < 0:
                    tile_right = np.zeros_like(tile_ref)
                else:
                    tile_right = load_mask_by_coors(
                        data_path, tile_coors=right_coors, prefix=prefix
                    )

                # Check if adjacent down right tile exists and load it
                down_right_coors = (
                    ref_coors[0],
                    ref_coors[1] + right_offset,
                    ref_coors[2] + down_offset,
                )
                if down_right_coors[1] < 0 or down_right_coors[2] < 0:
                    tile_down_right = np.zeros_like(tile_ref)
                else:
                    tile_down_right = load_mask_by_coors(
                        data_path, tile_coors=down_right_coors, prefix=prefix
                    )

                # Create code that loads adjacent tiles and checks for masks that overlap
                shifts_fname = (
                    iss.io.get_processed_path(data_path)
                    / "reg"
                    / f"{prefix}_shifts.npz"
                )
                shifts = np.load(shifts_fname)

                # As I have referenced overlap regions directly from the original tiles
                # deleting the overlapping labels also deletes them from the original tiles
                (
                    overlap_ref_vert,
                    overlap_down,
                    overlap_ref_side,
                    overlap_right,
                    overlap_ref_with_down_right,
                    overlap_down_right_with_ref,
                ) = get_overlap_regions(
                    shifts["tile_shape"],
                    shifts,
                    tile_ref,
                    tile_right,
                    tile_down,
                    tile_down_right,
                )
                overlapping_pairs = remove_overlapping_labels(
                    overlap_ref_side, overlap_right, upper_overlap_thresh
                )
                # print(f"Pairs of masks and overlaps {overlapping_pairs}")
                overlapping_pairs_down = remove_overlapping_labels(
                    overlap_ref_vert, overlap_down, upper_overlap_thresh
                )
                # print(f"Pairs of masks and overlaps {overlapping_pairs_down}")
                overlapping_pairs_down_right = remove_overlapping_labels(
                    overlap_ref_with_down_right,
                    overlap_down_right_with_ref,
                    upper_overlap_thresh,
                )
                # print(f"Pairs of masks and overlaps {overlapping_pairs_ref_down_right}")
                for pairs_list in [
                    overlapping_pairs,
                    overlapping_pairs_down,
                    overlapping_pairs_down_right,
                ]:
                    if pairs_list:  # Checks if the list is not empty
                        all_overlapping_pairs.extend(pairs_list)

                # Save the corrected masks
                for tile in [
                    (ref_coors, tile_ref),
                    (down_coors, tile_down),
                    (right_coors, tile_right),
                    (down_right_coors, tile_down_right),
                ]:
                    tile_roi, tile_x, tile_y = tile[0]
                    if tile_x >= 0 and tile_y >= 0:
                        fname = (
                            f"{prefix}_masks_corrected_{tile_roi}_{tile_x}_{tile_y}.npy"
                        )
                        np.save(
                            processed_path / "cells" / fname, tile[1], allow_pickle=True
                        )
    all_overlapping_pairs_array = np.array(
        [
            (label1, label2, overlap1, overlap2, sum_mask1, sum_mask2)
            for label1, label2, (
                overlap1,
                overlap2,
            ), sum_mask1, sum_mask2 in all_overlapping_pairs
        ]
    )
    np.save(
        processed_path / "cells" / f"{prefix}_overlapping_pairs.npy",
        all_overlapping_pairs_array,
        allow_pickle=True,
    )
    return all_overlapping_pairs
