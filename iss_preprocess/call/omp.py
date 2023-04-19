import numba
import numpy as np
from . import rois_to_array, BASES
from ..vis import plot_gene_templates


def make_gene_templates(cluster_means, codebook, vis=False):
    """
    Make dictionary of fluorescence values for each gene by finding well-matching
    spots.

    Args:
        rois (list): list of ROI objects containing fluorescence traces
        codebook (pandas.DataFrame): gene codes, containing 'gii', 'seq', and 'gene'
            columns.

    Returns:
        N x genes numpy.ndarray containing dictionary of fluorescence values for
            each gene.
        List of detected gene names.

    """
    gene_dict = []
    for seq in codebook["seq"]:
        base_ids = [np.where(b == BASES)[0][0] for b in seq]
        gene_barcode = [
            cluster_mean[base, :] for base, cluster_mean in zip(base_ids, cluster_means)
        ]
        gene_dict.append(np.concatenate(gene_barcode))
    gene_dict = np.stack(gene_dict, axis=1)
    gene_dict /= np.linalg.norm(gene_dict, axis=0)
    unique_genes = codebook["gene"]

    if vis:
        plot_gene_templates(gene_dict, unique_genes, BASES)

    return gene_dict, unique_genes


def refine_gene_templates(rois, gene_dict, unique_genes, thresh=0.8, vis=False):
    """
    Refine gene templates by finding spots that match the template and averaging
    their fluorescence values.

    TODO: This function is currently unused. Needs to be updated to work with
    new data structures.

    Args:
        rois (list): list of ROI objects containing fluorescence traces
        gene_dict (N x genes numpy.ndarray): dictionary of fluorescence values for
            each gene.
        unique_genes (list): list of gene names.
        thresh (float): threshold for matching spots to gene template.
            Default: 0.8.
        vis (bool): whether to visualize gene templates. Default: False.

    Returns:
        N x genes numpy.ndarray containing dictionary of fluorescence values for
            each gene.

    """
    x = rois_to_array(rois, normalize=False)
    x_ = np.reshape(x, (28, -1))
    x_ /= np.linalg.norm(x_, axis=0)
    gene_dict_refined = []
    for igene in range(gene_dict.shape[1]):
        matching_spots = np.dot(gene_dict[:, igene], x_) > thresh
        if np.sum(matching_spots) > 0:
            gene_dict_refined.append(np.mean(x_[:, matching_spots], axis=1))
        else:
            gene_dict_refined.append(gene_dict[:, igene])
    gene_dict_refined = np.stack(gene_dict_refined, axis=1)
    gene_dict_refined /= np.linalg.norm(gene_dict_refined, axis=0)

    if vis:
        plot_gene_templates(gene_dict_refined, unique_genes)

    return gene_dict_refined


def make_background_vectors(nrounds=7, nchannels=4):
    """
    Create background vectors for OMP algorithm. There is one vector for each channel.
    Each vector has fluorescence in one channel across all rounds. Vectors are
    normalized to have unit norm.

    Args:
        nrounds (int): number of rounds
        nchannels (int): number of channels.

    Returns:
        round x channel numpy.ndarray of background vectors.

    """
    b = []
    for ich in range(nchannels):
        vec = np.zeros((nrounds, nchannels))
        vec[:, ich] = 1
        b.append(vec.flatten())
    m = np.stack(b).T
    m /= np.linalg.norm(m, axis=0)
    return m


def barcode_spots_dot_product(
    spots, cluster_means, norm_shift=0, sequence_column="sequence"
):
    """
    Compute dot product between synthetic trace and observed trace for each spot.
    
    The synthetic trace is estimated using the provided bleeedthrough matrix.
    The observed trace is first background subtracted using the same approach as 
    used in the OMP algorithm.

    Args:
        spots (pandas.DataFrame): barcode spot table containing 'trace' column
            with fluorescence values.
        cluster_means (numpy.ndarray): Nrounds x Nchannels x Nclusters bleedthrough
            matrix of fluorescence values for each cluster (i.e. base).
        norm_shift (float): small value added to the norm of the observed trace. This
            penalizes the dot product score for spots with very low signal.
        sequence_column (str): name of column in spots table containing the sequence.
            Default is 'sequence', but could also be 'corrected_sequence'.
    
    Returns:
        List of dot product scores for each spot.
    
    """
    nrounds = cluster_means.shape[0]
    nchannels = cluster_means.shape[1]
    background_vectors = make_background_vectors(nrounds=nrounds, nchannels=nchannels)
    dot_product_scores = []
    for i, spot in spots.iterrows():
        if np.all(np.isfinite(spot["trace"])):
            synthetic_trace = cluster_means[
                np.arange(nrounds), :, spot[sequence_column]
            ].flatten()
            synthetic_trace /= np.linalg.norm(synthetic_trace)
            y = spot["trace"].flatten()
            norm_y = np.linalg.norm(y)
            y /= norm_y + norm_shift
            coefs_background, _, _, _ = np.linalg.lstsq(background_vectors, y)
            r = y - np.dot(background_vectors, coefs_background)
            dot_product_scores.append(np.dot(r, synthetic_trace))
        else:
            dot_product_scores.append(np.nan)
    return dot_product_scores


@numba.jit(nopython=True)
def omp(y, X, background_vectors=None, max_comp=None, tol=0.05):
    """
    Run Orthogonal Matching Pursuit to identify components present in the input
    signal.

    The algorithm works by iteratively. At each step we find the component that has
    the highest dot product with the residual of the input signal. After selecting
    a component, coefficients for all included components are estimated by least
    squares regression and the residuals are updated. The component is retained
    if it reduces the norm of the residuals by at least a fraction of the original
    norm specified by the tolerance parameter.

    Background vectors are automatically included.

    Algorithm stops when the tolerance threshold is reach or the number of
    components reaches `max_comp`.

    Args:
        y (numpy.ndarray): length N input signal.
        X (numpy.ndarray): N x M dictionary of M components.
        background_vectors (numpy.ndarray): N x O dictionary of background components.
        max_comp (int): maximum number of components to include.
        tol (float): tolerance threshold that determines the minimum fraction of
            the residual norm to retain a component.

    Returns:
        Length M + O array of component coefficients
        Length N array of residuals

    """
    norm_y = np.linalg.norm(y)
    # initialize residuals vector
    r = y
    if background_vectors is not None:
        X = np.concatenate((background_vectors, X), axis=1)
    if norm_y == 0:
        return np.zeros(X.shape[1]), r
    ichosen = np.zeros(X.shape[1], dtype=numba.boolean)
    if background_vectors is not None:
        # initial coefficients for background vectors
        ichosen[: background_vectors.shape[1]] = True
        coefs, _, _, _ = np.linalg.lstsq(X[:, ichosen], y)
        r = y - np.dot(X[:, ichosen], coefs)
    if not max_comp:
        max_comp = X.shape[1]
    # iterate until maximum number of components is reached
    while np.sum(ichosen) < max_comp:
        # find the largest dot product among components not yet included
        not_chosen = np.nonzero(np.logical_not(ichosen))[0]
        dot_product = np.dot(X[:, not_chosen].transpose(), y)
        best_match = not_chosen[np.argmax(np.abs(dot_product))]
        ichosen[best_match] = True
        # fit coefficients, including new component and calculate new residuals
        coefs_new, _, _, _ = np.linalg.lstsq(X[:, ichosen], y)
        r_new = y - np.dot(X[:, ichosen], coefs_new)
        # check if the reduction in residual norm passes the threshold
        if (np.linalg.norm(r) - np.linalg.norm(r_new)) / norm_y > tol:
            # if it does, update coefficients and residuals
            coefs = coefs_new
            r = r_new
        else:
            # if not, deselect the component and break loop
            ichosen[best_match] = False
            break
    # prepare output vector
    coefs_out = np.zeros(X.shape[1])
    if np.any(ichosen):
        coefs_out[ichosen] = coefs
    return coefs_out, r


@numba.jit(nopython=True)
def omp_weighted(
    y,
    X,
    background_vectors=None,
    max_comp=None,
    tol=0.05,
    alpha=120.0,
    beta_squared=1.0,
    weighted=True,
    refit_background=False,
    norm_shift=0.0,
):
    """
    Run Orthogonal Matching Pursuit to identify components present in the input
    signal.

    The algorithm works by iteratively. At each step we find the component that has
    the highest dot product with the residual of the input signal. After selecting
    a component, coefficients for all included components are estimated by least
    squares regression and the residuals are updated. The component is retained
    if it reduces the norm of the residuals by at least a fraction of the original
    norm specified by the tolerance parameter.

    Background vectors are automatically included.

    Algorithm stops when the tolerance threshold is reach or the number of
    components reaches `max_comp`.

    Args:
        y (numpy.ndarray): length N input signal.
        X (numpy.ndarray): N x M dictionary of M components.
        background_vectors (numpy.ndarray): N x O dictionary of background components.
        max_comp (int): maximum number of components to include.
        tol (float): tolerance threshold that determines the minimum fraction of
            the residual norm to retain a component.
        alpha (float): parameter for weighted OMP.
        beta_squared (float): parameter for weighted OMP.
        weighted (bool): whether to use weighted OMP. Default is True.
        refit_background (bool): whether to refit background coefficients on every iteration.
            Default is True.        
        norm_shift (float): additional shift to add to the norm of the pixel trace. Larger values
            reduce false positive gene calls in dim pixels. Default is 0.

    Returns:
        Length M + O array of component coefficients
        Length N array of residuals

    """
    norm_y = np.linalg.norm(y)
    if norm_y == 0:
        return np.zeros(X.shape[1]), y
    y /= norm_y + norm_shift
    # initialize residuals vector
    nbackground = background_vectors.shape[1]
    if background_vectors is not None:
        Xfull = np.concatenate((background_vectors, X), axis=1)
    ichosen = np.zeros(Xfull.shape[1], dtype=numba.boolean)
    if background_vectors is not None:
        # initial coefficients for background vectors
        ichosen[: background_vectors.shape[1]] = True
        coefs_background, _, _, _ = np.linalg.lstsq(Xfull[:, ichosen], y)
        if not refit_background:
            y = y - np.dot(Xfull[:, ichosen], coefs_background)
            r = y
        else:
            # TODO: double check if this is correct
            r = y - np.dot(Xfull[:, ichosen], coefs_background)
    else:
        r = y
    if not max_comp:
        max_comp = X.shape[1]
    coefs = coefs_background
    if not refit_background:
        Xweighted = np.empty(X.shape, dtype=np.float64)
    else:
        Xweighted = np.empty(Xfull.shape, dtype=np.float64)

    # iterate until maximum number of components is reached
    while np.sum(ichosen) < max_comp:
        # find the largest dot product among components not yet included
        not_chosen = np.nonzero(np.logical_not(ichosen))[0]
        if weighted:
            sigma_squared = beta_squared + alpha * np.sum(
                Xfull[:, ichosen] ** 2 * coefs ** 2, axis=1
            )
            weights_sq = (1 / sigma_squared) / np.mean(1 / sigma_squared)
            dot_product = np.abs(
                np.dot(Xfull[:, not_chosen].transpose(), r * weights_sq)
            )
        else:
            dot_product = np.abs(np.dot(Xfull[:, not_chosen].transpose(), r))
        best_match = not_chosen[np.argmax(dot_product)]
        if (
            ichosen[best_match] == True or np.max(dot_product) < tol
        ):  # gene already added or background gene
            break
        else:
            ichosen[best_match] = True
        if weighted:
            weights = np.sqrt(weights_sq)
            # fit coefficients, including new component and calculate new residuals
            if not refit_background:
                for ix in range(X.shape[0]):
                    Xweighted[ix, :] = X[ix, :] * weights[ix]
                coefs_new, _, _, _ = np.linalg.lstsq(
                    Xweighted[:, ichosen[nbackground:]], y * weights
                )
            else:
                for ix in range(Xfull.shape[0]):
                    Xweighted[ix, :] = Xfull[ix, :] * weights[ix]
                coefs_new, _, _, _ = np.linalg.lstsq(Xweighted[:, ichosen], y * weights)
        else:
            if not refit_background:
                coefs_new, _, _, _ = np.linalg.lstsq(X[:, ichosen[nbackground:]], y)
            else:
                coefs_new, _, _, _ = np.linalg.lstsq(Xfull[:, ichosen], y)
        if not refit_background:
            r = y - np.dot(X[:, ichosen[nbackground:]], coefs_new)
            coefs = np.concatenate((coefs_background, coefs_new))
        else:
            r = y - np.dot(Xfull[:, ichosen], coefs_new)
            coefs = coefs_new

    # prepare output vector
    coefs_out = np.zeros(Xfull.shape[1])
    if np.any(ichosen):
        coefs_out[ichosen] = coefs
    return coefs_out, r


def run_omp(
    stack,
    gene_dict,
    tol=0.05,
    weighted=True,
    refit_background=True,
    alpha=120.0,
    beta_squared=1.0,
    norm_shift=0.0,
    max_comp=None,
    min_intensity=0,
):
    """
    Apply the OMP algorithm to every pixel of the provided image stack.

    Args:
        stack (numpy.ndarray): X x Y x C x R image stack.
        gene_dict (numpy.ndarray): N x M dictionary, where N = R * C and M is the
            number of genes.
        tol (float): tolerance threshold for OMP algorithm.
        weighted (bool): whether to use weighted OMP. Default is True.
        refit_background (bool): whether to refit background coefficients on every iteration.
            Default is True.
        alpha (float): parameter for weighted OMP.
        beta_squared (float): parameter for weighted OMP.
        norm_shift (float): additional shift to add to the norm of the pixel trace. Larger values
            reduce false positive gene calls in dim pixels. Default is 0.
        max_comp (int): maximum number of components to use in OMP. Default is None, in which case
            OMP proceeds until the tolerance threshold is reached.
        min_intensity (float): minimum intensity for a pixel to be considered. Calculated as
            the mean absolute value of the pixel trace. Default is 0.

    Returns:
        Gene coefficient matrix of shape X x Y x M
        Background coefficient matrix of shape X x Y x C
        Residual stack, shape X x Y x (R * C)

    """
    stack = np.moveaxis(stack, 2, 3)

    background_vectors = make_background_vectors(
        nrounds=stack.shape[2], nchannels=stack.shape[3]
    )
    n_background_vectors = background_vectors.shape[1]
    stack = stack.astype(float)

    @numba.jit(nopython=True, parallel=True)
    def omp_loop_(stack_, background_vectors_, gene_dict_, tol_):
        g = np.zeros(
            (
                stack_.shape[0],
                stack_.shape[1],
                gene_dict_.shape[1] + background_vectors_.shape[1],
            )
        )
        r = np.zeros(
            (stack_.shape[0], stack_.shape[1], stack_.shape[2] * stack_.shape[3])
        )
        for ix in numba.prange(stack_.shape[0]):
            if ix % 100 == 0:
                print(f"finished row {ix} of {stack_.shape[0]}")
            for iy in range(stack_.shape[1]):
                if np.mean(np.abs(stack_[ix, iy, :, :])) > min_intensity:
                    g[ix, iy, :], r[ix, iy, :] = omp_weighted(
                        stack_[ix, iy, :, :].flatten(),
                        gene_dict_,
                        background_vectors=background_vectors_,
                        tol=tol_,
                        weighted=weighted,
                        refit_background=refit_background,
                        alpha=alpha,
                        beta_squared=beta_squared,
                        norm_shift=norm_shift,
                        max_comp=max_comp,
                    )
        return g, r

    g, r = omp_loop_(stack, background_vectors, gene_dict, tol)

    return g[:, :, n_background_vectors:], g[:, :, :n_background_vectors], r
