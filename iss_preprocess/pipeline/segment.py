from os import system
import numpy as np
import pandas as pd
import iss_preprocess as iss
from pathlib import Path
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels
from ..segment import cellpose_segmentation, count_spots, spot_mask_value
from .stitch import stitch_registered
from ..io import get_roi_dimensions, load_ops, get_pixel_size, load_metadata
from . import diagnostics
from . import ara_registration as ara_reg
import warnings


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
            args = args + " --partition=cpu"
        args = args + f" --output={Path.home()}/slurm_logs/{data_path}/iss_segment_%j.out"

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
        data_path, ref_prefix=reference, prefix=prefix, roi=iroi
    )
    print("starting segmentation", flush=True)
    masks = cellpose_segmentation(
        stitched_stack[..., 0],
        channels=(0, 0),
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
        regionprops_table(big_masks, properties=("label", "centroid", "area", "bbox"))
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
