import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cellpose.io import imread
from cellpose.models import CellposeModel
from cellpose import plot
from skimage.morphology import dilation


def cellpose_segmentation(fname, channels=(3, 2), flow_threshold=2,
                          min_pix=500, vis=False, dilate_pix=50, rescale=0.55):
    """
    Segment cells using Cellpose.

    Args:
        fname (str): path to file containing reference image
        channels (tuple): channels to use for segmentation
        flow_threshold (float): flow threshold for cellpose cell detection
        min_pix (int): minimum number of pixels to keep mask
        vis (bool): whether to plot masks
        dilate_pix (int): number of rounds of binary dilation to grow masks
        rescale (float): rescale factor for cellpose model

    Returns:
        nummpy.ndarray of masks

    """
    model = CellposeModel(gpu=False, model_type='cyto', net_avg=True, torch=True)
    img = imread(fname)
    masks, flows, styles = model.eval(
        img,
        rescale=rescale,
        channels=channels,
        flow_threshold=flow_threshold,
        verbose=True
    )

    nmasks = np.max(masks)
    npix = np.empty(nmasks)
    for mask in range(nmasks):
        npix[mask] = np.sum(masks == mask + 1)
        if npix[mask] < min_pix:
            masks[masks == mask + 1] = 0

    for i in range(dilate_pix):
        masks_dilated = dilation(masks)
        masks[masks == 0] = masks_dilated[masks == 0]

    if vis:
        plt.figure(figsize=(15, 15))
        plt.imshow(plot.mask_rgb(masks))
        plt.show()

    return masks


def count_rolonies(masks, rolony_locations, gene_names):
    """
    Count number of rolonies within each mask and return a DataFrame of gene counts.

    Args:
        masks (numpy.ndarray): cell masks
        rolony_locations (list): list of DataFrames with spot locations for each gene
        gene_names: names of genes

    Returns:
        A DataFrame of gene counts.

    """
    nmasks = np.max(masks)
    gene_matrix = np.zeros((nmasks + 1, len(gene_names)))
    gene_df = pd.DataFrame(gene_matrix, columns=gene_names)
    for gene, spots in zip(gene_names, rolony_locations):
        mask_ids = masks[spots['y'].round().to_numpy().astype(int), spots['x'].round().to_numpy().astype(int)]
        for mask in mask_ids:
            gene_df.loc[mask, gene] += 1
    return gene_df
