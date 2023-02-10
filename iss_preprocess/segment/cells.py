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


def count_rolonies(masks, spots):
    """
    Count number of rolonies within each mask and return a DataFrame of gene counts.

    Args:
        masks (numpy.ndarray): cell masks
        spots (pandas.DataFrame): table of spot locations for each gene

    Returns:
        A DataFrame of gene counts.

    """
    gene_names = spots["gene"].unique()
    nmasks = np.max(masks)
    gene_matrix = np.zeros((nmasks + 1, len(gene_names)))
    gene_df = pd.DataFrame(gene_matrix, columns=gene_names)
    for gene in gene_names:
        this_gene = spots[spots["gene"] == gene]
        mask_ids = masks[
            this_gene["y"].round().to_numpy().astype(int),
            this_gene["x"].round().to_numpy().astype(int),
        ]
        for mask in mask_ids:
            gene_df.loc[mask, gene] += 1
    return gene_df
