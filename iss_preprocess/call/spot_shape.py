from math import floor

import cv2
import numpy as np
import pandas as pd
from skimage.morphology import disk

from ..segment import detect_spots


def get_spot_shape(g, spot_xy=7, neighbor_filter_size=9, neighbor_threshold=15):
    """
    Get average spot shape.

    Args:
        g (numpy.ndarray): X x Y x Ngenes OMP output
        spot_xy (int): spot radius to extract
        neighbor_filter_size (int): size of the square filter used for counting pixels
        in initial spot selection
        neighbor_threshold (int): minimum number of positive pixels for a spot to be
        included in the average

    Returns:
        numpy.ndarray: (spot_xy + 1) x (spot_xy+1) mean spot image.

    """
    spot_sign_image = np.zeros((spot_xy * 2 + 1, spot_xy * 2 + 1))
    nspots = 0
    for igene in range(g.shape[2]):
        print(f"processing {igene} of {g.shape[2]}...")
        gene_spots = detect_spots(g[:, :, igene], threshold=0)
        neighborhood = np.ones((neighbor_filter_size, neighbor_filter_size))
        g_filt = cv2.filter2D(
            (g[:, :, igene] > 0).astype(float),
            -1,
            neighborhood,
            borderType=cv2.BORDER_REPLICATE,
        )
        pos_neighbors = g_filt[gene_spots["y"], gene_spots["x"]]
        use_spots = np.where(pos_neighbors >= neighbor_threshold)[0]

        for spot in use_spots:
            spot_x = int(gene_spots.iloc[spot]["x"])
            spot_y = int(gene_spots.iloc[spot]["y"])
            if (
                spot_xy < spot_x < g.shape[1] - spot_xy - 1
                and spot_xy < spot_y < g.shape[0] - spot_xy - 1
            ):
                spot_sign_image += np.sign(
                    g[
                        spot_y - spot_xy : spot_y + spot_xy + 1,
                        spot_x - spot_xy : spot_x + spot_xy + 1,
                        igene,
                    ]
                )
                nspots += 1
    return spot_sign_image, nspots


def apply_symmetry(spot_sign_image):
    """
    Generates a circularly symmetric spot image by averaging pixels at the same distance
    from the centre.

    Args:
        spot_sign_image (numpy.ndarray): inputs spot image

    Returns:
        numpy.ndarray: circularly symmetric spot image

    """
    X, Y = np.meshgrid(
        np.arange(spot_sign_image.shape[0]), np.arange(spot_sign_image.shape[1])
    )
    X = X - floor(spot_sign_image.shape[0] / 2)
    Y = Y - floor(spot_sign_image.shape[1] / 2)
    D = X**2 + Y**2
    unique_ds = np.unique(D)
    symmetric_spot_sign_image = np.empty(spot_sign_image.shape)
    for unique_d in unique_ds:
        symmetric_spot_sign_image[D == unique_d] = np.mean(
            spot_sign_image[D == unique_d]
        )
    return symmetric_spot_sign_image


def find_gene_spots(
    g, spot_sign_image, gene_names, rho=2, spot_score_threshold=0.05, disk_radius=2
):
    """
    Finds gene spots and extracts additional gene coefficient statistics.

    Args:
        g (numpy.ndarray): X x Y x Ngenes OMP output
        spot_sign_image (numpy.ndarray): Average spot sign image for filtering
        gene_names (list): List of gene names corresponding to g's third dimension
        rho (float): Weight multiplier for positive spot pixels (default: 2)
        spot_score_threshold (float): Minimum score threshold for including spots
        (default: 0.05)
        disk_radius (int): Radius of the disk to extract gene coefficients (default: 2)

    Returns:
        list: List of pandas.DataFrame with spot coordinates and scores for each gene.
        pandas.DataFrame: DataFrame with spot x gene coefficients.
    """
    ngenes = g.shape[2]
    all_genes = []
    all_coefficients = []

    for igene in range(ngenes):
        print(
            f"Finding spots for gene {igene + 1} of {ngenes} ({gene_names[igene]})..."
        )
        gene_spots = detect_spots_by_shape(
            g[:, :, igene], spot_sign_image, threshold=0, rho=rho
        )
        gene_spots = gene_spots.iloc[
            (gene_spots["spot_score"] > spot_score_threshold).to_numpy().astype(bool)
        ]
        all_genes.append(gene_spots)

        # Extract coefficients for all genes at each spot
        spot_coeffs = []

        for _, spot in gene_spots.iterrows():
            y, x = int(spot["y"]), int(spot["x"])
            mask = disk(disk_radius)

            # Get values from OMP output for all genes
            coefficients = g[
                np.clip(y - mask.shape[0] // 2, 0, g.shape[0] - 1) : np.clip(
                    y + mask.shape[0] // 2 + 1, 0, g.shape[0] - 1
                ),
                np.clip(x - mask.shape[1] // 2, 0, g.shape[1] - 1) : np.clip(
                    x + mask.shape[1] // 2 + 1, 0, g.shape[1] - 1
                ),
                :,
            ]
            # Compute the mean coefficient value for each gene
            avg_coeffs = np.mean(coefficients, axis=(0, 1))
            spot_coeffs.append(avg_coeffs)

        # Create a DataFrame with spot x gene coefficients
        spot_gene_df = pd.DataFrame(spot_coeffs, columns=gene_names)
        spot_gene_df["spot_x"] = gene_spots["x"].values
        spot_gene_df["spot_y"] = gene_spots["y"].values
        spot_gene_df["gene"] = gene_names[igene]
        all_coefficients.append(spot_gene_df)
    all_coefficients = pd.concat(all_coefficients, ignore_index=True)
    return all_genes, all_coefficients


def detect_spots_by_shape(im, spot_sign_image, threshold=0, rho=2):
    """
    Detect spots in an image based on similarity to a spot sign image.

    Args:
        im (numpy.ndarray): input image
        spot_sign_image (numpy.ndarray): average spot sign image to use as a template
        in filtering spots
        threshold (float): threshold for initial spot detection. Default: 0.
        rho (float): multiplier that defines the relative weight assigned to
        positive spot pixels.
            Default: 2.

    Returns:
        pandas.DataFrame: spot coordinates and scores

    """
    spots = detect_spots(im, threshold=threshold)

    neg_max = np.sum(np.sign(spot_sign_image) == -1)
    pos_max = np.sum(np.sign(spot_sign_image) == 1)
    pos_filter = (np.sign(spot_sign_image) == 1).astype(float)
    neg_filter = (np.sign(spot_sign_image) == -1).astype(float)
    filt_pos = cv2.filter2D(
        (im > 0).astype(float), -1, pos_filter, borderType=cv2.BORDER_REPLICATE
    )
    filt_neg = cv2.filter2D(
        (im < 0).astype(float), -1, neg_filter, borderType=cv2.BORDER_REPLICATE
    )
    pos_pixels = filt_pos[spots["y"], spots["x"]]
    neg_pixels = filt_neg[spots["y"], spots["x"]]
    spot_score = (neg_pixels + pos_pixels * rho) / (neg_max + pos_max * rho)
    spots["spot_score"] = spot_score
    spots["pos_pixels"] = pos_pixels
    spots["neg_pixels"] = neg_pixels
    return spots
