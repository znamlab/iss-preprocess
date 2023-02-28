import numpy as np
import pandas as pd
from skimage.morphology import dilation


def cellpose_segmentation(
    img,
    channels=(0, 0),
    flow_threshold=0.4,
    min_pix=0,
    dilate_pix=0,
    rescale=0.55,
    model_type="cyto",
    use_gpu=False,
    **kwargs
):
    """Segment cells using Cellpose.

    Args:
        img (np.array): reference image
        channels (tuple, optional): channels to use for segmentation. Defaults to (0,0)
        flow_threshold (float, optional): flow threshold for cellpose cell detection.
            Defaults to 0.4.
        min_pix (int, optional): minimum number of pixels to keep mask. Defaults to 0.
        dilate_pix (int, optional): number of rounds of binary dilation to grow masks.
            Defaults to 0.
        rescale (float, optional): rescale factor for cellpose model. Defaults to 0.55.
        model_type (str, optional): Cellpose mode to use. Defaults to "cyto".
        use_gpu (bool, optional): Defaults to False.
        **kwargs (optional): Other kwargs are forwarded to CellposeModel.eval

    Returns:
        numpy.ndarray of masks

    """
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=use_gpu, model_type=model_type, net_avg=False)
    masks, flows, styles = model.eval(
        img,
        rescale=rescale,
        channels=channels,
        flow_threshold=flow_threshold,
        tile=True,
        **kwargs
    )
    if min_pix > 0:
        nmasks = np.max(masks)
        npix = np.empty(nmasks)
        for mask in range(nmasks):
            npix[mask] = np.sum(masks == mask + 1)
            if npix[mask] < min_pix:
                masks[masks == mask + 1] = 0

    for i in range(dilate_pix):
        masks_dilated = dilation(masks)
        masks[masks == 0] = masks_dilated[masks == 0]

    return masks


def rolonie_mask_value(masks, spots):
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


def count_rolonies(spots, grouping_column, masks=None):
    """
    Count number of rolonies within each mask and return a DataFrame of counts.

    Args:
        spots (pandas.DataFrame): table of spot locations for each group
        grouping_column (str): name of the column to group counts, usually 'gene' or
            'bases'
        masks (numpy.ndarray, optional): cell masks. Must be positive integers. Can be 
            None If spots already includes a "mask_id" columns. Defaults to None.

    Returns:
        A DataFrame of counts by unique values of `grouping_column`.

    """
    if masks is None:
        assert "mask_id" in spots.columns
    else:
        spots = rolonie_mask_value(masks, spots)
    # count the number of occurence of each ("mask_id", genes or barcode) pair
    cell_df = pd.DataFrame(
        spots.loc[:, ["mask_id", grouping_column]]
        .groupby(["mask_id", grouping_column])
        .aggregate(len)
    )
    
    # formating
    cell_df = cell_df.unstack(grouping_column)
    cell_df[np.isnan(cell_df)] = 0

    return cell_df[0].astype(int)
