import cv2
import numpy as np
from ..segment import detect_spots
from math import floor


def get_spot_shape(g, spot_xy=7, neighbor_filter_size=9, neighbor_threshold=15):
    """
    Get average spot shape.

    Args:
        g (numpy.ndarray): X x Y x Ngenes OMP output
        spot_xy (int): spot radius to extract
        neighbor_filter_size (int): size of the square filter used for counting pixels in initial spot selection
        neighbor_threshold (int): minimum number of positive pixels for a spot to be included in the average

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

    return spot_sign_image / nspots


def apply_symmetry(spot_sign_image):
    """
    Generates a circularly symmetric spot image by averaging pixels at the same distance from the centre.

    Args:
        spot_sign_image (numpy.ndarray): inputs spot image

    Returns:
    numpy.ndarray: circularly s

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


def find_gene_spots(g, spot_sign_image, rho=2, spot_score_threshold=0.05):
    """
    Finds gene spots based on similarity to the spot sign image.

    We first detect spots by finding peaks in the OMP out put images. For each
    spot we compute a score based on its similarity with average spot sign image.
    This score is defined as
        (n_neg + rho * n_pos) / (max_neg + rho * max_pos)

    Args:
        g (numpy.ndarray): X x Y x Ngenes OMP output
        spot_sign_image (numpy.ndarray): average spot sign image to use as a template in filtering spots
        rho (float): multiplier that defines the relative weight assigned to positive spot pixels.
            Default: 2.
        spot_score_threshold (float): minimum score threshold for including spots.
            Default: 0.05

    Returns:

    """
    ngenes = g.shape[2]
    all_genes = []
    for igene in range(ngenes):
        print(f"findings spots for gene {igene} of {ngenes}...")
        gene_spots = detect_spots_by_shape(
            g[:, :, igene], spot_sign_image, threshold=0, rho=rho
        )
        gene_spots = gene_spots.iloc[(gene_spots["spot_score"] > spot_score_threshold).to_numpy().astype(bool)]
        all_genes.append(gene_spots)
    return all_genes


def detect_spots_by_shape(im, spot_sign_image, threshold=0, rho=2):
    neg_max = np.sum(np.sign(spot_sign_image) == -1)
    pos_max = np.sum(np.sign(spot_sign_image) == 1)
    spots = detect_spots(im, threshold=threshold)
    pos_filter = (np.sign(spot_sign_image) == 1).astype(float)
    neg_filter = (np.sign(spot_sign_image) == -1).astype(float)
    filt_pos = cv2.filter2D(
        (im > 0).astype(float),
        -1,
        pos_filter,
        borderType=cv2.BORDER_REPLICATE,
    )
    filt_neg = cv2.filter2D(
        (im < 0).astype(float),
        -1,
        neg_filter,
        borderType=cv2.BORDER_REPLICATE,
    )
    pos_pixels = filt_pos[spots["y"], spots["x"]]
    neg_pixels = filt_neg[spots["y"], spots["x"]]
    spot_score = (neg_pixels + pos_pixels * rho) / (neg_max + pos_max * rho)
    spots["spot_score"] = spot_score
    spots["pos_pixels"] = pos_pixels
    spots["neg_pixels"] = neg_pixels
    return spots
