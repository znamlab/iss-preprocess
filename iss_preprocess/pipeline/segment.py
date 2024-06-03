import warnings
from tqdm import tqdm
import glob
from os import system
from pathlib import Path
from znamutils import slurm_it
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_triangle
from skimage.segmentation import expand_labels

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


def segment_all_tiles(
    data_path,
    prefix="DAPI_1",
    use_raw_stack=True,
    use_gpu=True,
    use_rois=None,
    tile_list=None,
):
    """Start batch jobs for segmentation for each tile.

    Args:
        data_path (str): Relative path to data.
        prefix (str, optional): acquisition prefix to use for segmentation.
            Defaults to "DAPI_1".
        use_raw_stack (bool, optional): Whether to use the raw stack and do 3d
            segmentation. Defaults to True.
        use_gpu (bool, optional): Whether to use GPU. Defaults to True.
        use_rois (list, optional): List of ROIs to process. If None, will use all ROIs.
            Defaults to None.
        tile_list (list, optional): List of tiles to process. If provided will ignore
            use_rois. If None, will use all tiles.

    Returns:
        list: List of job IDs for the slurm jobs.
    """
    # create the list of tiles to process
    if tile_list is None:
        roi_dims = get_roi_dimensions(data_path)
        if use_rois is None:
            ops = iss.io.load_ops(data_path)
            use_rois = ops.get("use_rois", roi_dims[:, 0])
        tile_list = []
        for r in use_rois:
            _, nx, ny = roi_dims[roi_dims[:, 0] == r][0]
            tile_list.extend(
                [(r, ix, iy) for ix in range(nx + 1) for iy in range(ny + 1)]
            )

    slurm_folder = Path.home() / "slurm_logs" / data_path / "segmentation"
    slurm_folder.mkdir(exist_ok=True, parents=True)
    job_ids = segment_tile(
        data_path,
        prefix=prefix,
        use_gpu=use_gpu,
        use_raw_stack=use_raw_stack,
        use_slurm=True,
        slurm_folder=slurm_folder,
        batch_param_names=["roi", "tx", "ty"],
        batch_param_list=tile_list,
    )
    return job_ids


@slurm_it(
    conda_env="iss-preprocess",
    slurm_options={
        "partition": "gpu",
        "gpus-per-node": 1,
        "mem": "64GB",
        "time": "3:00:00",
    },
)
def segment_tile(
    data_path,
    roi=None,
    tx=None,
    ty=None,
    prefix="DAPI_1",
    use_raw_stack=True,
    use_gpu=False,
):
    """Run segmentation on a single roi.

    Args:
        data_path (str): Relative path to data.
        roi (int): ROI to process.
        tx (int): Tile x coordinate.
        ty (int): Tile y coordinate.
        prefix (str, optional): Acquisition prefix to use for segmentation.
            Defaults to "DAPI_1".
        use_raw_stack (bool, optional): Whether to use the raw stack and do 3d
            segmentation. Defaults to True.
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.
    """
    # we make tile_coors keyword only to avoid confusion with the slurm_it decorator
    print(f"Segmenting {data_path} {prefix} roi {roi} x={tx} y={ty}", flush=True)
    tile_coors = (roi, tx, ty)
    print("Loading data")
    ops = iss.io.load_ops(data_path)
    img = get_stack_for_cellpose(data_path, prefix, tile_coors, use_raw_stack)
    if use_raw_stack:
        stitch_threshold = 0.3
        z_axis = 3
    else:
        stitch_threshold = 0
        z_axis = None

    print(f"segmenting {data_path} {tile_coors} {prefix}")
    pretrained_model = ops["cellpose_pretrained_model"]
    if pretrained_model is not None:
        pretrained_model = iss.io.get_processed_path(pretrained_model)
    masks = cellpose_segmentation(
        img,
        z_axis=z_axis,
        channel_axis=2,
        use_gpu=use_gpu,
        channels=[0, 1],  # channel selection is made in get_stack_for_cellpose
        flow_threshold=ops["cellpose_flow_threshold"],
        min_pix=ops["cellpose_min_pix"],
        dilate_pix=ops["cellpose_dilate_pix"],
        diameter=ops["cellpose_diameter"],
        rescale=ops["cellpose_rescale"],
        model_type=ops["cellpose_model_type"],
        pretrained_model=pretrained_model,
        debug=False,
        stitch_threshold=stitch_threshold,
        normalize=dict(normalize=True, norm3D=False),
        cellprob_threshold=0.0,
        do_3D=False,
        anisotropy=None,
    )

    target = iss.io.get_processed_path(data_path) / "cells"
    target.mkdir(exist_ok=True)
    tile_name = "_".join(map(str, tile_coors))
    fname = f"{prefix}_masks_{tile_name}.npy"
    if use_raw_stack:
        # project masks to a single plane
        _, _, projected_mask = project_mask(masks, min_pix_size=ops["cellpose_min_pix"])
        # save raw 3D masks
        raw_target = target / "raw_masks"
        raw_target.mkdir(exist_ok=True)
        np.save(raw_target / fname, masks)
        masks = projected_mask

    np.save(target / fname, masks)
    print(f"Saved masks to {target}")

    return img, masks


def get_stack_for_cellpose(data_path, prefix, tile_coors, use_raw_stack=True):
    """Load the stack to segment with cellpose.

    This will load a stack with 2 channels from the raw data or the registered stack.

    Args:
        data_path (str): Relative path to data.
        prefix (str): Acquisition prefix to use for segmentation.
        tile_coors (tuple): Coordinates of the tile to segment.
        use_raw_stack (bool, optional): Whether to use the raw stack or the projected
            stack. Defaults to True.

    Returns:
        numpy.ndarray: X x Y x channels (x Z) stack.
    """
    ops = iss.io.load_ops(data_path)
    channels = ops["cellpose_channels"]
    if use_raw_stack:
        raw_stack = iss.pipeline.load_and_register_raw_stack(
            data_path, prefix, tile_coors
        )
        raw_stack = np.nan_to_num(raw_stack, 0)
        raw_stack = np.clip(raw_stack, 0, 2**16 - 1).astype(np.uint16)
        z_shift = ops.get("cellpose_channel_z_shift", 0)
        if z_shift:
            shape = np.array(raw_stack.shape)
            shape[2] = 2
            shape[3] -= z_shift
            img = np.zeros(shape, dtype=raw_stack.dtype)
            if z_shift > 0:
                img[..., 0, :] = raw_stack[..., channels[0], :-z_shift]
                img[..., 1, :] = raw_stack[..., channels[1], z_shift:]
            else:
                img[..., 0, :] = raw_stack[..., channels[0], -z_shift:]
                img[..., 1, :] = raw_stack[..., channels[1], :z_shift]
            channels = [0, 1]
        else:
            img = raw_stack[..., channels, :]
    else:
        img = iss.pipeline.load_and_register_tile(data_path, tile_coors, prefix)
        img = img[..., channels]
    return img


def project_mask(masks, min_pix_size):
    """Project masks to a single plane.

    Args:
        masks (np.array): 3D array of masks.
        min_pix_size (int): Minimum number of pixels in the center plane to keep a mask.


    Returns:
        np.array: 2D array of projected masks.
    """
    print("Separating masks")
    binary_masks, individual_masks = _separate_masks(masks, min_pix_size=min_pix_size)

    # First merge all the masks that do not overlap
    n_masks = np.sum(binary_masks, axis=0)
    projected_mask = np.zeros_like(n_masks)
    # Find part with overlapping masks
    overlapping = {}
    print("Finding overlapping masks")
    for i_m, m in tqdm(enumerate(binary_masks), total=len(binary_masks)):
        if n_masks[m].max() == 1:
            projected_mask[m] = i_m + 1
        else:
            overlapping[i_m + 1] = np.unique(individual_masks[:, m])

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

    return binary_masks, individual_masks, projected_mask


def _separate_masks(masks, min_pix_size):
    """Innner function to separate masks and keep only one plane per mask.

    For each mask, find the center plane and keep only that one.

    Args:
        masks (np.array): 3D array of masks.
        min_pix_size (int): Minimum number of pixels in the center plane to keep a mask.

    Returns:
        binary_masks (np.array): 2D array of binary masks.
        individual_masks (np.array): 2D array of masks with unique values.
    """

    mask_ids = np.unique(masks[masks != 0])

    # separate masks, keeping only one plane per mask
    binary_masks = np.zeros((len(mask_ids), *masks.shape[1:]), dtype=bool)
    individual_masks = np.zeros((len(mask_ids), *masks.shape[1:]), dtype="int16")
    for i_m, mask_id in tqdm(enumerate(mask_ids), total=len(mask_ids)):
        # Find the center z plane of the mask
        mask = masks == mask_id
        npx_per_plane = np.sum(mask, axis=(1, 2))
        mask_z = np.where(npx_per_plane > npx_per_plane.max() * 0.3)[0]
        center_z = mask_z[len(mask_z) // 2]
        # if the mask is too small, skip it
        if np.sum(mask[center_z]) < min_pix_size:
            continue
        binary_masks[i_m] = mask[center_z]
        individual_masks[i_m] = mask[center_z] * (i_m + 1)
    return binary_masks, individual_masks


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
    big_masks = get_big_masks(data_path, roi, masks, mask_expansion)

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
    barcode_df=None,
    barcode_dot_threshold=0.15,
    spot_score_threshold=0.1,
    hyb_score_threshold=0.8,
    load_genes=True,
    load_hyb=True,
    load_barcodes=True,
):
    """Load gene, barcode, and hybridisation spots and add a mask_id column to each
    spots dataframe

    Args:
        data_path (str): Relative path to data
        roi (int): ID of the ROI to load
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.
        masks (np.array, optional): Array of labels. If None will load "masks_{roi}".
             Defaults to None.
        barcode_df (pd.DataFrame, optional): Rabies barcode dataframe, if None, will
            load "barcode_df_roi{roi}.pkl". Defaults to None.
        barcode_dot_threshold (float, optional): Threshold for the barcode dot product.
            Only spots above the threshold will be counted. Defaults to 0.15.
        spot_score_threshold (float, optional): Threshold for the OMP score. Only spots
            above the threshold will be counted. Defaults to 0.1.
        hyb_score_threshold (float, optional): Threshold for hybridisation spots. Only
            spots above the threshold will be counted. Defaults to 0.8.
        load_genes (bool, optional): Whether to load gene spots. Defaults to True.
        load_hyb (bool, optional): Whether to load hybridisation spots. Defaults to True
        load_barcodes (bool, optional): Whether to load barcode spots. Defaults to True.


    Returns:
        dict: Dictionary of spots dataframes

    """
    processed_path = iss.io.get_processed_path(data_path)
    big_masks = get_big_masks(data_path, roi, masks, mask_expansion)

    metadata = load_metadata(data_path=data_path)
    spot_acquisitions = []
    if load_genes:
        spot_acquisitions.append("genes_round")
    if load_barcodes:
        spot_acquisitions.append("barcode_round")
    thresholds = dict(
        genes_round=("spot_score", spot_score_threshold),
        barcode_round=("dot_product_score", barcode_dot_threshold),
    )
    if load_hyb:
        for hyb in metadata["hybridisation"]:
            spot_acquisitions.append(hyb)
            thresholds[hyb] = ("score", hyb_score_threshold)

    # get the spots dataframes
    spots_dict = dict()
    for prefix in spot_acquisitions:
        if prefix == "barcode_round" and barcode_df is not None:
            spot_df = barcode_df
        else:
            print(f"Loading {prefix}", flush=True)
            spot_df = pd.read_pickle(processed_path / f"{prefix}_spots_{roi}.pkl")
        filt_col, threshold = thresholds[prefix]
        if threshold is not None:
            spot_df = spot_df[spot_df[filt_col] > threshold]
        # modify spots in place
        spots_dict[prefix] = spot_mask_value(big_masks, spot_df)
    return spots_dict


def segment_spots(
    data_path,
    roi,
    mask_expansion=5.0,
    masks=None,
    barcode_df=None,
    barcode_dot_threshold=None,
    spot_score_threshold=0.1,
    hyb_score_threshold=0.8,
    load_genes=True,
    load_hyb=True,
    load_barcodes=True,
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
        barcode_df (pd.DataFrame, optional): Rabies barcode dataframe, if None, will
            load "barcode_df_roi{roi}.pkl". Defaults to None.
        barcode_dot_threshold (float, optional): Threshold for the barcode dot product.
            Only spots above the threshold will be counted. Defaults to 0.15.
        spot_score_threshold (float, optional): Threshold for the OMP score. Only spots
            above the threshold will be counted. Defaults to 0.1.
        hyb_score_threshold (float, optional): Threshold for hybridisation spots. Only
            spots above the threshold will be counted. Defaults to 0.8.
        load_genes (bool, optional): Whether to load gene spots. Defaults to True.
        load_hyb (bool, optional): Whether to load hybridisation spots. Defaults to True
        load_barcodes (bool, optional): Whether to load barcode spots. Defaults to True.

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
        barcode_df=barcode_df,
        barcode_dot_threshold=barcode_dot_threshold,
        spot_score_threshold=spot_score_threshold,
        hyb_score_threshold=hyb_score_threshold,
        load_genes=load_genes,
        load_hyb=load_hyb,
        load_barcodes=load_barcodes,
    )

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


def get_big_masks(data_path, roi, masks, mask_expansion):
    """Small internal function to avoid code duplication

    Reload and expand masks if needed

    Args:
        data_path (str): Relative path to data
        roi (int): ID of the ROI to load
        masks (np.array, optional): Array of labels. If None will load "masks_{roi}".
             Defaults to None.
        mask_expansion (float, optional): Distance in um to expand masks before counting
            rolonies per cells. None for no expansion. Defaults to 5.


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
    max_val = big_masks.max()
    # find the smallest integer type that can hold the max value
    if max_val < 256:
        big_masks = big_masks.astype(np.uint8)
    elif max_val < 65536:
        big_masks = big_masks.astype(np.uint16)
    else:
        big_masks = big_masks.astype(np.uint32)
    return big_masks


def segment_mcherry_tile(
    data_path,
    prefix,
    roi,
    tilex,
    tiley,
    suffix="unmixed",
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

    Returns:
        filtered_masks (np.ndarray): Binary image of the filtered masks.
        filtered_df (pd.DataFrame): DataFrame of the filtered masks.
        rejected_masks (np.ndarray): Binary image of the rejected masks.
    """

    # Load the unmixed and original mCherry image stacks
    processed_path = iss.io.get_processed_path(data_path)
    ops = load_ops(data_path)
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
    filt = iss.image.filter_stack(
        unmixed_stack, r1=ops["mcherry_r1"], r2=ops["mcherry_r2"], dtype=float
    )
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

    # TODO: these are a lot of threshold and we don't have an easy way to set them
    # adapt to detect more mask here and filter later.
    filtered_df = props_df[
        (props_df["area"] > ops["min_area_threshold"])
        & (props_df["area"] < ops["max_area_threshold"])
        & (props_df["circularity"] >= ops["min_circularity_threshold"])
        & (props_df["circularity"] <= ops["max_circularity_threshold"])
        & (props_df["eccentricity"] <= ops["max_elongation_threshold"])
        & (props_df["solidity"] >= ops["min_solidity_threshold"])
        & (props_df["solidity"] < ops["max_solidity_threshold"])
        & (props_df["intensity_mean-3"] < ops["max_bg_intensity_threshold"])
    ]

    rejected_masks_df = props_df[
        ~(props_df["area"] > ops["min_area_threshold"])
        & (props_df["area"] < ops["max_area_threshold"])
        & (props_df["circularity"] >= ops["min_circularity_threshold"])
        & (props_df["circularity"] >= ops["max_circularity_threshold"])
        & (props_df["eccentricity"] <= ops["max_elongation_threshold"])
        & (props_df["solidity"] >= ops["min_solidity_threshold"])
        & (props_df["solidity"] < ops["max_solidity_threshold"])
        & (props_df["intensity_mean-3"] < ops["max_bg_intensity_threshold"])
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
    Dynamically identifies and removes overlapping labels in place, in two adjacent tile overlaps
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
                        (label1, label2, percent_overlap_tile1, percent_overlap_tile2)
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


@slurm_it(conda_env="iss-preprocess", print_job_id=True)
def remove_all_overlapping_masks(data_path, prefix, upper_overlap_thresh):
    """
    Remove masks that overlap in adjacent tiles.

    Args:
        data_path (str): Relative path to the data.
        prefix (str): Prefix of the image stack.
        upper_overlap_thresh (float): The upper threshold percentage for considering mask overlap significant.

    Returns:
        all_overlapping_pairs (list): A list of tuples containing the labels that overlapped and their respective percentages.
    """
    processed_path = iss.io.get_processed_path(data_path)
    roi_dims = iss.io.get_roi_dimensions(data_path, prefix)
    ops = iss.io.load_ops(data_path)
    # Remove all old files with "masks_corrected" in the name
    for f in glob.glob(str(processed_path / "cells" / f"{prefix}_masks_corrected*")):
        Path(f).unlink()

    # First remove masks at the edges of all the tiles
    for roi in roi_dims[:, 0]:
        for tilex in tqdm(
            range(roi_dims[roi - 1, 1] + 1),
            desc=f"ROI {roi} X-axis",
            total=roi_dims[roi - 1, 1],
        ):
            for tiley in tqdm(
                range(roi_dims[roi - 1, 2] + 1),
                desc=f"Tile {tilex} Y-axis",
                leave=False,
                total=roi_dims[roi - 1, 2],
            ):
                coors = (roi, tilex, tiley)
                tile = iss.io.load.load_mask_by_coors(
                    data_path,
                    tile_coors=coors,
                    prefix=prefix,
                    suffix="",
                )
                corrected_masks, _ = find_edge_touching_masks(tile)
                # Save the edge corrected masks
                fname = f"{prefix}_masks_corrected_{roi}_{tilex}_{tiley}.npy"
                np.save(
                    processed_path / "cells" / fname, corrected_masks, allow_pickle=True
                )

    # Now remove overlapping masks
    all_overlapping_pairs = []
    if (ops["x_tile_direction"] != "right_to_left") or (
        ops["y_tile_direction"] != "top_to_bottom"
    ):
        warnings.warn(
            "This function is only tested for right_to_left and top_to_bottom tile direction"
        )
    for roi in roi_dims[:, 0]:
        # TODO: this might fail with other microscope

        for tilex in tqdm(
            reversed(range(roi_dims[roi - 1, 1] + 1)),
            desc=f"ROI {roi} X-axis (overlap check)",
            total=roi_dims[roi - 1, 1],
        ):
            for tiley in tqdm(
                reversed(range(roi_dims[roi - 1, 2] + 1)),
                desc=f"Tile {tilex} Y-axis (overlap check)",
                leave=False,
                total=roi_dims[roi - 1, 2],
            ):
                ref_coors = (roi, tilex, tiley)
                # Load reference tile
                tile_ref = iss.io.load.load_mask_by_coors(
                    data_path,
                    tile_coors=ref_coors,
                    prefix=prefix,
                    suffix="corrected",
                )

                # Check if adjacent down tile exists and load it
                down_offset = 1 if ops["y_tile_direction"] == "bottom_to_top" else -1
                down_coors = (ref_coors[0], ref_coors[1], ref_coors[2] + down_offset)
                if down_coors[2] < 0:
                    tile_down = np.zeros_like(tile_ref)
                else:
                    tile_down = iss.io.load.load_mask_by_coors(
                        data_path,
                        tile_coors=down_coors,
                        prefix=prefix,
                        suffix="corrected",
                    )

                # Check if adjacent right tile exists and load it
                right_offset = 1 if ops["x_tile_direction"] == "left_to_right" else -1
                right_coors = (ref_coors[0], ref_coors[1] + right_offset, ref_coors[2])
                if right_coors[1] < 0:
                    tile_right = np.zeros_like(tile_ref)
                else:
                    tile_right = iss.io.load.load_mask_by_coors(
                        data_path,
                        tile_coors=right_coors,
                        prefix=prefix,
                        suffix="corrected",
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
                    tile_down_right = iss.io.load.load_mask_by_coors(
                        data_path,
                        tile_coors=down_right_coors,
                        prefix=prefix,
                        suffix="corrected",
                    )

                # Create code that loads adjacent tiles and checks for masks that overlap
                # we use the prefix shift, not the reference as we want to be sure to
                # find duplicates and we will not stitch
                shifts_fname = (
                    iss.io.get_processed_path(data_path)
                    / "reg"
                    / f"{prefix}_shifts.npz"  # f"{ops['reference_prefix']}_shifts.npz"
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
                    # Checks if the list is not empty
                    all_overlapping_pairs.extend(pairs_list)

                # Save the corrected masks
                for tile in [
                    (ref_coors, tile_ref),
                    (down_coors, tile_down),
                    (right_coors, tile_right),
                    (down_right_coors, tile_down_right),
                ]:
                    tile_coors, mask = tile
                    tile_roi, tile_x, tile_y = tile_coors
                    if tile_x >= 0 and tile_y >= 0:
                        fname = (
                            f"{prefix}_masks_corrected_{tile_roi}_{tile_x}_{tile_y}.npy"
                        )
                        np.save(
                            processed_path / "cells" / fname, mask, allow_pickle=True
                        )

    np.save(
        processed_path / "cells" / f"{prefix}_overlapping_pairs.npy",
        np.vstack(all_overlapping_pairs),
        allow_pickle=True,
    )
    return all_overlapping_pairs


def remove_non_cell_masks(data_path, roi, tilex, tiley):
    """
    Remove masks that are not cells based on the clustering results.

    Args:
        data_path (str): Relative path to the data.
        roi (int): Region of interest.
        tilex (int): X coordinate of the tile.
        tiley (int): Y coordinate of the tile.
    """
    processed_path = iss.io.get_processed_path(data_path)
    mask_dir = processed_path / "cells"
    df_thresh = pd.read_pickle(mask_dir / "df_thresh.pkl")
    tile = np.load(
        mask_dir / f"mCherry_1_masks_{roi}_{tilex}_{tiley}.npy", allow_pickle=True
    )
    image_df = df_thresh[
        (df_thresh["roi"] == roi)
        & (df_thresh["tilex"] == tilex)
        & (df_thresh["tiley"] == tiley)
    ]
    # Remove bad masks
    for label in np.unique(tile):
        if label == 0:
            continue
        elif label in image_df["label"].astype(np.uint16).values:
            # Check if the mask is in the bad cluster (0)
            if image_df[image_df["label"] == label]["cluster_label"].values[0] != 0:
                continue
        else:
            tile[tile == label] = 0

    # Save the edge corrected masks
    fname = f"mCherry_1_cell_masks_{roi}_{tilex}_{tiley}.npy"
    np.save(processed_path / "cells" / fname, tile, allow_pickle=True)


def find_mcherry_cells(data_path):
    """
    Find cell clusters in the mCherry channel using a GMM to cluster
    cells based on their morphological features. Then remove non-cell
    masks based on the clustering results and save remaining masks.

    Args:
        data_path (str): Relative path to the data.
    """
    processed_path = iss.io.get_processed_path(data_path)
    df_dir = processed_path / "cells"
    df_files = glob.glob(str(df_dir / "*.pkl"))
    dfs = [pd.read_pickle(f) for f in df_files]
    df = pd.concat(dfs)
    if df.empty:
        raise ValueError("No masks found in any tile.")

    scaler = StandardScaler()

    features = [
        "area",
        "circularity",
        "solidity",
        "intensity_mean-3",
        "intensity_mean-2",
    ]

    df_norm = (df[features] - df[features].min()) / (
        df[features].max() - df[features].min()
    )
    scaled_features = scaler.fit_transform(df_norm[features])
    df_scaled_features = pd.DataFrame(scaled_features, columns=features)

    # TODO: Remove hardcoded cluster centers (use percentiles?)
    cluster_centers_scaled = np.array(
        [
            [-0.81560289, -1.16570977, -1.16885992, 0.68591332, -0.47768646],
            [-0.08201876, 0.48188625, 0.38447341, -0.41695244, -0.42873761],
            [0.97601349, 0.61105821, 0.74187513, -0.18499336, 1.06977134],
        ]
    )

    # Fit GMM
    n_components = 3
    gmm = GaussianMixture(
        n_components=n_components,
        means_init=cluster_centers_scaled,
        random_state=42,
        verbose=2,
    )
    gmm.fit(df_scaled_features[features])
    labels = gmm.predict(df_scaled_features[features])
    df["cluster_label"] = labels + 1
    df.to_pickle(processed_path / "cells" / "df_thresh.pkl")

    iss.pipeline.batch_process_tiles(data_path, script="remove_non_cell_masks")
