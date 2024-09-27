import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import dilation
from scipy.ndimage import binary_erosion, binary_dilation
import iss_preprocess as iss


# TODO move to IO or pipeline/segment
def get_cell_masks(
    data_path, roi, projection="corrected", mask_expansion=None, reload=True
):
    """Small wrapper to get cell masks from a given data path.

    Wrap to ensure we use the same projection for all calls

    Args:
        data_path (str): Path to acquisition data (chamber folder)
        roi (int): Region of interest
        projection (str, optional): Projection to use. Defaults to "corrected".
        mask_expansion (int, optional): Expansion of the mask. If None, reads from ops.
            Defaults to None.
        reload (bool, optional): If True, reload the masks. Defaults to True.

    Returns:
        np.ndarray: Cell masks
    """
    ops = iss.io.load_ops(data_path)
    if mask_expansion is None:
        mask_expansion = ops["mask_expansion"]

    seg_prefix = f"{ops['segmentation_prefix']}_masks"
    target = f"{data_path}/cells/{seg_prefix}_{roi}_{mask_expansion}.tif"
    target = iss.io.get_processed_path(target)

    if not reload and target.exists():
        masks = iss.io.load_stack(target)[..., 0]
    else:
        masks = iss.pipeline.stitch_registered(
            data_path,
            prefix=seg_prefix,
            roi=roi,
            projection=projection,
        )[..., 0]
        if mask_expansion > 0:
            masks = iss.pipeline.segment.get_big_masks(data_path, masks, mask_expansion)
        iss.io.write_stack(masks, target, dtype=masks.dtype)
    return masks


def cellpose_segmentation(
    img,
    channels,
    flow_threshold=0.4,
    min_pix=0,
    dilate_pix=0,
    model_type="cyto3",
    pretrained_model=None,
    use_gpu=False,
    debug=False,
    **kwargs,
):
    """Segment cells using Cellpose.

    Args:
        img (np.array): reference image
        channels (tuple): channels to use for segmentation.
        flow_threshold (float, optional): flow threshold for cellpose cell detection.
            Defaults to 0.4.
        min_pix (int, optional): minimum number of pixels to keep mask. Defaults to 0.
        dilate_pix (int, optional): number of rounds of binary dilation to grow masks.
            Defaults to 0.
        rescale (float, optional): rescale factor for cellpose model. Defaults to 0.55.
        model_type (str, optional): Cellpose mode to use. Defaults to "cyto".
        use_gpu (bool, optional): Defaults to False.
        debug (bool, optional): If True, return flows and styles. Defaults to False.
        **kwargs (optional): Other kwargs are forwarded to CellposeModel.eval

    Returns:
        numpy.ndarray of masks

    """
    from cellpose.models import CellposeModel

    model = CellposeModel(
        gpu=use_gpu, model_type=model_type, pretrained_model=pretrained_model
    )
    masks, flows, styles = model.eval(
        img,
        channels=channels,
        flow_threshold=flow_threshold,
        tile=True,
        **kwargs,
    )
    if min_pix > 0:
        print(f"Filtering masks with less than {min_pix} pixels")
        nmasks = np.max(masks)
        npix = np.empty(nmasks)
        for mask in tqdm(range(nmasks), total=nmasks):
            npix[mask] = np.sum(masks == mask + 1)
            if npix[mask] < min_pix:
                masks[masks == mask + 1] = 0
        print(f"Filtered {np.sum(npix < min_pix)} masks")
        print(f"Average mask size: {np.mean(npix[npix > min_pix])} pixels")

    if dilate_pix > 0:
        print(f"Dilating masks by {dilate_pix} pixels")
    for i in range(dilate_pix):
        masks_dilated = dilation(masks)
        masks[masks == 0] = masks_dilated[masks == 0]
    if debug:
        return masks, flows, styles
    return masks


def spot_mask_value(masks, spots):
    """Find the mask value of each spot

    Args:
        masks (numpy.array): cell masks. Must be positive integers
        spots (pandas.DataFrame): table of spot locations. Must have a x and y columns

    Returns:
        pandas.DataFrame: spots, modfied inplace to add a "mask_id" column

    """
    xy = np.round(spots.loc[:, ["x", "y"]].values).astype(int)
    # clip values outside of mask. Can happen because of registration shift
    for i in range(2):
        xy[:, i] = np.clip(xy[:, i], 0, masks.shape[::-1][i] - 1)
    mask_val = masks[xy[:, 1], xy[:, 0]]
    # add that to the spots df
    spots["mask_id"] = mask_val
    return spots


def count_spots(spots, grouping_column, masks=None, mask_id_column="mask_id"):
    """
    Count number of rolonies within each mask and return a DataFrame of counts.

    Args:
        spots (pandas.DataFrame): table of spot locations for each group
        grouping_column (str): name of the column to group counts, usually 'gene' or
            'bases'
        masks (numpy.ndarray, optional): cell masks. Must be positive integers. Can be
            None If spots already includes a "mask_id" columns. Defaults to None.
        mask_id_column (str, optional): name of the column containing the mask id.
            Defaults to "mask_id".

    Returns:
        A DataFrame of counts by unique values of `grouping_column`.

    """
    if masks is None:
        assert mask_id_column in spots.columns
    else:
        spots = spot_mask_value(masks, spots)
    # count the number of occurence of each ("mask_id", genes or barcode) pair
    cell_df = pd.DataFrame(
        spots.loc[:, [mask_id_column, grouping_column]]
        .groupby([mask_id_column, grouping_column])
        .aggregate(len)
    )

    # formating
    cell_df = cell_df.unstack(grouping_column)
    cell_df[np.isnan(cell_df)] = 0

    return cell_df[0].astype(int)


def project_mask(masks, min_pix_size):
    """Project masks to a single plane.

    Args:
        masks (np.array): 3D array of masks.
        min_pix_size (int): Minimum number of pixels in the center plane to keep a mask.


    Returns:
        np.array: 2D array of projected masks.
    """
    print("Separating masks")
    binary_masks, n_planes_per_mask = _separate_masks(masks, min_pix_size=min_pix_size)
    del masks

    # First merge all the masks that do not overlap
    n_masks = np.sum(binary_masks, axis=0)
    projected_mask = np.zeros_like(n_masks)
    # Find part with overlapping masks
    overlapping = {}
    print("Finding overlapping masks")
    for i_m, m in tqdm(enumerate(binary_masks), total=len(binary_masks)):
        if not np.any(m):
            # mask that are too small are emptied by _separate_masks
            continue
        if n_masks[m].max() == 1:
            projected_mask[m] = i_m + 1
        else:
            ovl_masks_index = np.unique(np.where(binary_masks[:, m])[0])
            # remove masks found only on a single plane
            ovl_masks_index = ovl_masks_index[n_planes_per_mask[ovl_masks_index] > 1]
            # check if we have any overlap left after removing single plane masks
            if np.sum(ovl_masks_index != i_m) > 0:
                overlapping[i_m + 1] = ovl_masks_index + 1
            else:
                # no overlap, keep the mask
                projected_mask[m] = i_m + 1

    print("Merging overlapping masks")
    for source, to_compare in tqdm(overlapping.items()):
        to_compare = to_compare[to_compare != 0]
        # also remove source
        to_compare = to_compare[to_compare != source]
        assert len(to_compare) > 0, f"No overlapping masks for {source}"
        for target in to_compare:
            projected_mask = _fuse_masks(
                source, target, binary_masks, projected_mask, min_pix_size
            )

    return projected_mask


def _separate_masks(masks, min_pix_size):
    """Innner function to separate masks and keep only one plane per mask.

    For each mask, find the center plane and keep only that one.

    Args:
        masks (np.array): 3D array of masks.
        min_pix_size (int): Minimum number of pixels in the center plane to keep a mask.

    Returns:
        binary_masks (np.array): 2D array of binary masks.
        n_planes_per_mask (np.array): Number of planes where the mask can be found.
    """

    mask_ids = np.unique(masks[masks != 0])

    # separate masks, keeping only one plane per mask
    binary_masks = np.zeros((len(mask_ids), *masks.shape[1:]), dtype=bool)
    n_planes_per_mask = np.zeros(len(mask_ids), dtype=int)
    for i_m, mask_id in tqdm(enumerate(mask_ids), total=len(mask_ids)):
        # Find the center z plane of the mask
        mask = masks == mask_id
        npx_per_plane = np.sum(mask, axis=(1, 2))
        # If the mask is only on a single plane, skip it
        if np.sum(npx_per_plane > 0) == 1:
            continue
        mask_z = np.where(npx_per_plane > npx_per_plane.max() * 0.3)[0]
        center_z = mask_z[len(mask_z) // 2]
        n_planes_per_mask[i_m] = len(mask_z)
        # if the mask is too small, skip it
        if np.sum(mask[center_z]) < min_pix_size:
            continue
        binary_masks[i_m] = mask[center_z]
    return binary_masks, n_planes_per_mask


def _fuse_masks(source, target, binary_masks, projected_mask, min_pix_size):
    """Inner function to fuse two masks.

    If the masks barely overlap, we ignore the intersection.
    If the masks overlap but are not the same, we try to remove the intersection
    and keep the rest of the mask.

    Args:
        source (int): ID of the source mask.
        target (int): ID of the target mask.
        binary_masks (np.array): 2D array of binary masks.
        projected_mask (np.array): 2D array of projected masks.
        min_pix_size (int): Minimum number of pixels in the center plane to keep a mask.

    Returns:
        np.array: 2D array of projected masks.
    """
    s_mask = binary_masks[source - 1].copy()
    t_mask = binary_masks[target - 1].copy()

    intersection = np.logical_and(s_mask, t_mask)
    union = np.logical_or(s_mask, t_mask)
    i_over_u = np.sum(intersection) / np.sum(union)
    t_in_s = np.sum(intersection) / np.sum(t_mask)
    s_in_t = np.sum(intersection) / np.sum(s_mask)
    barely_overlap = i_over_u < 0.01
    overlap_but_2cells = (i_over_u < 0.6) and (s_in_t < 0.7) and (t_in_s < 0.7)
    if barely_overlap or overlap_but_2cells:
        # barely overlapping, we can ignore the intersection
        s_mask[intersection] = 0
        t_mask[intersection] = 0
        # erode and dilate to remove weird shape
        s_mask = binary_erosion(s_mask, iterations=5)
        s_mask = binary_dilation(s_mask, iterations=5)
        t_mask = binary_erosion(t_mask, iterations=5)
        t_mask = binary_dilation(t_mask, iterations=5)
        if np.sum(t_mask) > min_pix_size:
            projected_mask[t_mask] = target
        if np.sum(s_mask) > min_pix_size:
            projected_mask[s_mask] = source

    if (np.sum(t_mask) < min_pix_size) or (np.sum(s_mask) < min_pix_size):
        return projected_mask

    if t_in_s > 0.7:
        clean_s = s_mask.copy()
        clean_s[t_mask] = 0
        clean_s = binary_erosion(clean_s, iterations=5)
        clean_s = binary_dilation(clean_s, iterations=5)
        projected_mask[s_mask] = 0
        projected_mask[clean_s] = source
        projected_mask[t_mask] = target
    if s_in_t > 0.7:
        clean_t = t_mask.copy()
        clean_t[s_mask] = 0
        clean_t = binary_erosion(clean_t, iterations=5)
        clean_t = binary_dilation(clean_t, iterations=5)
        projected_mask[t_mask] = 0
        projected_mask[clean_t] = target
        projected_mask[s_mask] = source

    return projected_mask


def remove_overlapping_labels(overlap_ref, overlap_shifted, upper_overlap_thresh=0.3):
    """Remove overlapping labels in two adjacent tile overlaps.

    Dynamically identifies and removes overlapping labels in place, in two adjacent tile
    overlaps based on upper and lower overlap percentage thresholds. If the overlap is
    above the upper threshold, the label in the shifted tile is removed. If the overlap
    is below the lower threshold, the shared region is removed from both masks.

    Args:
        overlap_ref (np.array): The overlap area of the reference tile.
        overlap_shifted (np.array): The overlap area of the shifted tile.
        upper_overlap_thresh (float, optional): The upper threshold percentage for
            considering mask overlap significant. Defaults to 0.3.

    Returns:
        overlapping_pairs (pandas.DataFrame): A dataframe containing the labels that
            overlapped, their respective percentages and what happened to them.
    """

    # Continuously process until no more labels meet the criteria for adjustments
    labels_changed = True
    overlapping_pairs = []
    while labels_changed:
        labels_changed = False  # Reset flag for this iteration
        unique_labels_1 = np.unique(overlap_ref[overlap_ref != 0])

        for label1 in unique_labels_1:
            mask1 = overlap_ref == label1
            overlapping_masks = np.unique(overlap_shifted[mask1])
            overlapping_masks = overlapping_masks[overlapping_masks != 0]
            for label2 in overlapping_masks:
                mask2 = overlap_shifted == label2
                if not np.any(mask1 & mask2):  # Check if there's any overlap
                    continue
                overlap_area = np.sum(mask1 & mask2)
                overlap_tile1 = overlap_area / np.sum(mask1)
                overlap_tile2 = overlap_area / np.sum(mask2)
                overlap_above_thresh = int(overlap_tile1 > upper_overlap_thresh) + int(
                    overlap_tile2 > upper_overlap_thresh
                )
                overlapping_pairs.append(
                    dict(
                        ref_label=label1,
                        match_label=label2,
                        ref_overlap=overlap_tile1,
                        match_overlap=overlap_tile2,
                        overlap_above=overlap_above_thresh,
                        delete_match=overlap_above_thresh > 0,
                        erase_overlap=overlap_above_thresh == 0,
                    )
                )
                # If overlap is above the upper threshold, remove the label from the
                # shifted tile
                if overlap_above_thresh > 0:
                    print(f"Removing label {label2} from shifted_tile")
                    overlap_shifted[mask2] = 0
                    labels_changed = True
                # If overlap is below the lower threshold, remove the shared region from
                # both masks
                else:
                    print(f"Removing shared region from labels {label1} and {label2}")
                    shared_mask = mask1 & mask2
                    overlap_ref[shared_mask] = 0
                    overlap_shifted[shared_mask] = 0
                    labels_changed = True

                if labels_changed:
                    # Exit the inner loop to reevaluate conditions due to the changes
                    break
            if labels_changed:
                # Exit the outer loop to restart the evaluation with updated overlaps
                break
    return overlapping_pairs
