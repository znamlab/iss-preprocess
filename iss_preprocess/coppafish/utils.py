import warnings
from math import floor

import numpy as np
import scipy


def ftrans2(b, t=None) -> np.ndarray:
    """
    Produces a 2D convolve kernel that corresponds to the 1D convolve kernel, `b`, using the transform, `t`.
    Copied from [MATLAB `ftrans2`](https://www.mathworks.com/help/images/ref/ftrans2.html).
    Args:
        b: `float [Q]`.
            1D convolve kernel.
        t: `float [M x N]`.
            Transform to make `b` a 2D convolve kernel.
            If `None`, McClellan transform used.
    Returns:
        `float [(M-1)*(Q-1)/2+1 x (N-1)*(Q-1)/2+1]`.
            2D convolve kernel.
    """
    if t is None:
        # McClellan transformation
        t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8

    # Convert the 1-D convolve_2d b to SUM_n a(n) cos(wn) form
    n = int(round((len(b) - 1) / 2))
    b = b.reshape(-1, 1)
    b = np.rot90(np.fft.fftshift(np.rot90(b)))
    a = np.concatenate((b[:1], 2 * b[1 : n + 1]))

    inset = np.floor((np.array(t.shape) - 1) / 2).astype(int)

    # Use Chebyshev polynomials to compute h
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = inset[0]
    cols = inset[1]
    h[rows, cols] += a[0, 0] * p0
    for i in range(2, n + 1):
        p2 = 2 * scipy.signal.convolve2d(t, p1)
        rows = rows + inset[0]
        cols = cols + inset[1]
        p2[rows, cols] -= p0
        rows = inset[0] + np.arange(p1.shape[0])
        cols = (inset[1] + np.arange(p1.shape[1])).reshape(-1, 1)
        hh = h.copy()
        h = a[i] * p2
        h[rows, cols] += hh
        p0 = p1.copy()
        p1 = p2.copy()
    h = np.rot90(h)
    return h


def hanning_diff(r1: int, r2: int) -> np.ndarray:
    """
    Gets difference of two hanning window 2D convolve kernel.
    Central positive, outer negative with sum of `0`.
    Args:
        r1: radius in pixels of central positive hanning convolve kernel.
        r2: radius in pixels of outer negative hanning convolve kernel.
    Returns:
        `float [2*r2+1 x 2*r2+1]`.
            Difference of two hanning window 2D convolve kernel.
    """
    h_outer = np.hanning(2 * r2 + 3)[1:-1]  # ignore zero values at first and last index
    h_outer = -h_outer / h_outer.sum()
    h_inner = np.hanning(2 * r1 + 3)[1:-1]
    h_inner = h_inner / h_inner.sum()
    h = h_outer.copy()
    h[r2 - r1 : r2 + r1 + 1] += h_inner
    h = ftrans2(h)
    return h


def annulus(r0, r_xy) -> np.ndarray:
    """
    Gets structuring element used to assess if spot isolated.
    Args:
        r0: Inner radius within which values are all zero.
        r_xy: Outer radius in xy direction.
            Can be float not integer because all values with `radius < r_xy1` and `> r0` will be set to `1`.
        r_z: Outer radius in z direction. Size in z-pixels.
            None means 2D annulus returned.
    Returns:
        `int [2*floor(r_xy1)+1, 2*floor(r_xy1)+1, 2*floor(r_z1)+1]`.
            Structuring element with each element either `0` or `1`.

    adapted from coppafish

    """
    r_xy1_int = floor(r_xy)
    y, x = np.meshgrid(
        np.arange(-r_xy1_int, r_xy1_int + 1), np.arange(-r_xy1_int, r_xy1_int + 1)
    )
    m = x**2 + y**2
    # only use upper radius in xy direction as z direction has different pixel size.
    annulus = r_xy**2 >= m
    annulus = np.logical_and(annulus, m > r0**2)
    return annulus.astype(int)


def scaled_k_means(
    x: np.ndarray,
    initial_cluster_mean: np.ndarray,
    score_thresh=0,
    min_cluster_size=10,
    n_iter=100,
):
    """
    Does a clustering that minimizes the norm of ```x[i] - g[i] * cluster_mean[cluster_ind[i]]```
    for each data point ```i``` in ```x```, where ```g``` is the gain which is not explicitly computed.
    Args:
        x: ```float [n_points x n_dims]```.
            Data set of vectors to build cluster means from.
        initial_cluster_mean: ```float [n_clusters x n_dims]```.
            Starting point of mean cluster vectors.
        score_thresh: `float` or give different score for each cluster as `float [n_clusters]`
            Scalar between ```0``` and ```1```.
            Points in ```x``` with dot product to a cluster mean vector greater than this
            contribute to new estimate of mean vector.
        min_cluster_size: If less than this many points assigned to a cluster,
            that cluster mean vector will be set to ```0```.
        n_iter: Maximum number of iterations performed.
    Returns:
        - norm_cluster_mean - ```float [n_clusters x n_dims]```.
            Final normalised mean cluster vectors.
        - cluster_eig_value - ```float [n_clusters]```.
            First eigenvalue of outer product matrix for each cluster.
        - cluster_ind - ```int [n_points]```.
            Index of cluster each point was assigned to. ```-1``` means fell below score_thresh and not assigned.
        - top_score - ```float [n_points]```.
            `top_score[i]` is the dot product score between `x[i]` and `norm_cluster_mean[cluster_ind[i]]`.
        - cluster_ind0 - ```int [n_points]```.
            Index of cluster each point was assigned to on first iteration.
            ```-1``` means fell below score_thresh and not assigned.
        - top_score0 - ```float [n_points]```.
            `top_score0[i]` is the dot product score between `x[i]` and `initial_cluster_mean[cluster_ind0[i]]`.
    """
    # normalise starting points and original data
    norm_cluster_mean = initial_cluster_mean / np.linalg.norm(
        initial_cluster_mean, axis=1
    ).reshape(-1, 1)
    x_norm = x / np.linalg.norm(x, axis=1).reshape(-1, 1)
    n_clusters = initial_cluster_mean.shape[0]
    n_points, n_dims = x.shape
    cluster_ind = (
        np.ones(x.shape[0], dtype=int) * -2
    )  # set all to -2 so won't end on first iteration
    cluster_eig_val = np.zeros(n_clusters)

    if len(np.array([score_thresh]).flatten()) == 1:
        # if single threshold, set the same for each cluster
        score_thresh = np.ones(n_clusters) * score_thresh
    elif isinstance(score_thresh, list):
        score_thresh = np.array(score_thresh)
        assert len(score_thresh) == n_clusters, "score_thresh must be length n_clusters"

    for i in range(n_iter):
        cluster_ind_old = cluster_ind.copy()

        # project each point onto each cluster. Use normalized so we can interpret score
        score = x_norm @ norm_cluster_mean.transpose()
        cluster_ind = np.argmax(score, axis=1)  # find best cluster for each point
        top_score = score[np.arange(n_points), cluster_ind]
        top_score[np.where(np.isnan(top_score))[0]] = (
            score_thresh.min() - 1
        )  # don't include nan values
        cluster_ind[top_score < score_thresh[cluster_ind]] = -1  # unclusterable points
        if i == 0:
            top_score0 = top_score.copy()
            cluster_ind0 = cluster_ind.copy()

        if (cluster_ind == cluster_ind_old).all():
            break

        for c in range(n_clusters):
            my_points = x[
                cluster_ind == c
            ]  # don't use normalized, to avoid overweighting weak points
            n_my_points = my_points.shape[0]
            if n_my_points < min_cluster_size:
                norm_cluster_mean[c] = 0
                warnings.warn(
                    f"Cluster c only had {n_my_points} vectors assigned to it.\n "
                    f"This is less than min_cluster_size = {min_cluster_size} so setting this cluster to 0."
                )
                continue
            eig_vals, eigs = np.linalg.eig(
                my_points.transpose() @ my_points / n_my_points
            )
            best_eig_ind = np.argmax(eig_vals)
            norm_cluster_mean[c] = eigs[:, best_eig_ind] * np.sign(
                eigs[:, best_eig_ind].mean()
            )  # make them positive
            cluster_eig_val[c] = eig_vals[best_eig_ind]

    return (
        norm_cluster_mean,
        cluster_eig_val,
        cluster_ind,
        top_score,
        cluster_ind0,
        top_score0,
    )
