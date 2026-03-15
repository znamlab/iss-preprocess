import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from znamutils import slurm_it

from ..io import get_processed_path, load_ops
from ..pipeline.stitch import stitch_registered

__all__ = [
    "create_tissue_mask",
]
# more mem
@slurm_it(conda_env="iss-preprocess", slurm_options=dict(mem="128G"))
def create_tissue_mask(
        data_path, 
        roi,
        initial_threshold_percentile = 7,
        kernel_size = 500,
        margin_size_relative = 0.025,
        reload = False,
        plot = True,
        plot_cells = True,
        cells_df_path = "filtered_cells_with_global_coords.pkl",
    ):
    """
    Create a binary mask of the tissue.

    Args:
        data_path (str): path to the data
        initial_threshold_percentile (int): initial threshold percentile for tissue detection
        kernel_size (int): size of the kernel for morphological operations
        margin_size_relative (float): relative size of the margin for tissue detection

    Returns:
        numpy.ndarray: binary mask of the tissue
    """
    
    processed_path = get_processed_path(data_path)
    mouse_folder = processed_path.parent
    
    # load mask if it already exists
    save_folder = f"{mouse_folder}/tissue_masks"
    os.makedirs(save_folder, exist_ok=True)
    mask_path = f"{save_folder}/tissue_mask_{roi}.npy"
    if reload and os.path.exists(mask_path):
        print(f"Loading existing tissue mask for ROI {roi} from {mask_path}")
        return np.load(mask_path)

    threshold_percentile = initial_threshold_percentile

    ops = load_ops(data_path)

    stitched_stack = stitch_registered(
        data_path,
        "barcode_round_2_1",
        roi=roi,
        channels=ops["ref_ch"],
    )
    img8 = cv2.normalize(stitched_stack[:,:,0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img8 = cv2.GaussianBlur(img8, (0, 0), 50)
    retry = False

    while True:
        thresh = np.percentile(img8, threshold_percentile)
        binarised = img8 > thresh

        # dilate to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binarised = cv2.erode(binarised.astype(np.uint8), kernel, iterations=2)
        
        # find connected components
        labels = label(binarised, connectivity=2)

        # get areas of connected components
        regions = regionprops(labels)
        areas = np.array([r.area for r in regions])
        biggest = np.argmax(areas)

        # define edge
        margin_x = int(binarised.shape[0] * margin_size_relative)  # pixels
        margin_y = int(binarised.shape[1] * margin_size_relative)  # pixels
        x_opposite = binarised.shape[0] - margin_x
        y_opposite = binarised.shape[1] - margin_y

        # only keep biggest component
        largest_component_mask = (labels == (biggest + 1))  # labels are 1-indexed
        if len(areas) > 1:
            # any other area more than 20% of biggest and doesn't touch edge, keep it too
            for i, area in enumerate(areas):
                if i != biggest and area > 0.1 * areas[biggest]:
                    # get map of this component
                    component_mask = (labels == (i + 1))
                    # compensate for previous erosion by dilating the component mask
                    component_mask = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
                    # check if it touches edge     
                    close_to_edge = (
                        np.any(component_mask[:margin_x, :]) |
                        np.any(component_mask[x_opposite:, :]) |
                        np.any(component_mask[:, :margin_y]) |
                        np.any(component_mask[:, y_opposite:])
                    )
                    if not close_to_edge:
                        largest_component_mask = largest_component_mask | (labels == (i + 1))

        dilated_mask = cv2.dilate(largest_component_mask.astype(np.uint8), kernel, iterations=2)
        dilated_mask = binary_fill_holes(dilated_mask).astype(dilated_mask.dtype)

        # if the mask is too small, lower threshold and repeat

        close_to_edge = (
            np.any(dilated_mask[:margin_x, :]) |
            np.any(dilated_mask[x_opposite:, :]) |
            np.any(dilated_mask[:, :margin_y]) |
            np.any(dilated_mask[:, y_opposite:])
        )
        if dilated_mask.sum() < 0.15 * binarised.size:
            if not retry:
                print(f"Mask for ROI {roi} is too small ({dilated_mask.sum()} pixels), lowering threshold and retrying")
                retry = True
            threshold_percentile -= 0.5
            if threshold_percentile < 0.1:
                print(f"Threshold percentile dropped below 0.1, giving up on this ROI")
                break

        elif close_to_edge:
            if not retry:
                print(f"Mask for ROI {roi} is close to edge, lowering threshold and retrying")
                retry = True
            threshold_percentile += 1
            if threshold_percentile > 10:
                print(f"Threshold percentile above 10, giving up on broadening this ROI")
                break
        
        else:
            break

    # if within margin of edge, flag for manual review
    if plot:


        # plot mask overlaid on image with cells
        plt.figure(figsize=(10, 10))
        plt.imshow(dilated_mask, alpha=0.5)
        plt.imshow(stitched_stack[:,:,0], 
                cmap="gray", 
                alpha=0.5, 
                vmin=0, 
                vmax=np.quantile(stitched_stack[:,:,0], 0.99)
        )
        if plot_cells:
            assert os.path.exists(f"{mouse_folder}/{cells_df_path}"), f"Cells DataFrame not found at {mouse_folder}/{cells_df_path}"
            cells_df = pd.read_pickle(f"{mouse_folder}/{cells_df_path}")
            plt.scatter(cells_df[cells_df["roi"] == roi]["x"], 
                        cells_df[cells_df["roi"] == roi]["y"], 
                        s=1, 
                        color="red", 
                        label="cells"
            )
        if close_to_edge:
            plt.title(f"ROI {roi} - final threshold {threshold_percentile} - Mask close to edge - Manual Review Needed")
        else:
            plt.title(f"ROI {roi} - final threshold {threshold_percentile} - Mask OK")
        plt.legend()

        plt.savefig(f"{save_folder}/tissue_mask_{roi}.png", dpi=300)
        plt.close()
    np.savez_compressed(f"{save_folder}/tissue_mask_{roi}.npz", mask=dilated_mask.astype(bool))

    return dilated_mask