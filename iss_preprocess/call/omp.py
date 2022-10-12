import numba
import numpy as np
from . import basecall_rois, call_genes, rois_to_array
from itertools import compress


def make_gene_templates(rois, codebook, max_errors=0, rounds=()):
    """
    Make dictionary of fluorescence values for each gene by finding well-matching
    spots.

    Args:
        rois (list): list of ROI objects containing fluorescence traces
        codebook (pandas.DataFrame): gene codes, containing 'gii', 'seq', and 'gene'
            columns.
        max_errors (int): maximum number of error to include a spot. Default 0.
        rounds (list): list of rounds to include. If empty, all rounds are used.

    Returns:
        N x genes numpy.ndarray containing dictionary of fluorescence values for
            each gene.
        List of detected gene names.

    """
    # rounds x channels x rois matrix
    f = rois_to_array(rois, normalize=False)
    # only include ROIs imaged on all rounds
    valid_rois = np.isfinite(f[0,0,:])
    sequences, _, _ = basecall_rois(list(compress(rois, valid_rois)), separate_rounds=False, rounds=rounds)
    f = f[:, :, valid_rois]
    genes, errors = call_genes(sequences, codebook)
    errors = np.array(errors)
    genes = np.array(genes)
    # ignore spots that have too many mismatches
    genes[errors > max_errors] = 'nan'
    unique_genes = np.unique(genes)
    unique_genes = unique_genes[unique_genes != 'nan']

    f = np.moveaxis(f, 2, 0)
    f = np.reshape(f, (f.shape[0], -1))
    gene_dict = np.empty((f.shape[1], len(unique_genes)))
    for igene, gene in enumerate(unique_genes):
        gene_dict[:, igene] = f[genes == gene, :].mean(axis=0)
    gene_dict /= np.linalg.norm(gene_dict, axis=0)
    return gene_dict, unique_genes


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
        ichosen[:background_vectors.shape[1]] = True
        coefs,_,_,_ = np.linalg.lstsq(X[:, ichosen], y)
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
        coefs_new,_,_,_ = np.linalg.lstsq(X[:, ichosen], y)
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
def omp_weighted(y, X, background_vectors=None, max_comp=None, tol=0.05,
                 alpha=120., beta_squared=1., weighted=True, refit_background=False,
                 norm_shift=0.):
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
        ichosen[:background_vectors.shape[1]] = True
        coefs_background,_,_,_ = np.linalg.lstsq(Xfull[:, ichosen], y)
        if not refit_background:
            y = y - np.dot(Xfull[:, ichosen], coefs_background)
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
        if weighted:
            sigma_squared = beta_squared + alpha * np.sum(Xfull[:, ichosen]**2 * coefs**2, axis=1)
            weights_sq = (1 / sigma_squared) / np.mean(1 / sigma_squared)
            dot_product = np.abs(np.dot(Xfull.transpose(), r * weights_sq))
        else:
            dot_product = np.abs(np.dot(Xfull.transpose(), r))
        best_match = np.argmax(dot_product)
        if ichosen[best_match] == True or dot_product[best_match] < tol: # gene already added or background gene
            break
        else:
            ichosen[best_match] = True
        if weighted:
            weights = np.sqrt(weights_sq)
            # fit coefficients, including new component and calculate new residuals
            if not refit_background:
                for ix in range(X.shape[0]):
                    Xweighted[ix, :] = X[ix,:] * weights[ix]
                coefs_new,_,_,_ = np.linalg.lstsq(Xweighted[:, ichosen[nbackground:]], y * weights)
            else:
                for ix in range(Xfull.shape[0]):
                    Xweighted[ix, :] = Xfull[ix,:] * weights[ix]
                coefs_new,_,_,_ = np.linalg.lstsq(Xweighted[:, ichosen], y * weights)
        else:
            if not refit_background:
                coefs_new,_,_,_ = np.linalg.lstsq(X[:, ichosen[nbackground:]], y)
            else:
                coefs_new,_,_,_ = np.linalg.lstsq(Xfull[:, ichosen], y)
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


def run_omp(stack, gene_dict, tol=0.05):
    """
    Apply the OMP algorithm to every pixel of the provided image stack.

    Args:
        stack (numpy.ndarray): X x Y x R x C image stack.
        gene_dict (numpy.ndarray): N x M dictionary, where N = R * C and M is the
            number of genes.
        tol (float): tolerance threshold for OMP algorithm.

    Returns:
        Gene coefficient matrix of shape X x Y x M
        Background coefficient matrix of shape X x Y x C
        Residual stack, shape X x Y x (R * C)

    """
    background_vectors = make_background_vectors(nrounds=stack.shape[2], nchannels=stack.shape[3])
    n_background_vectors = background_vectors.shape[1]
    stack = stack.astype(float)

    @numba.jit(nopython=True, parallel=True)
    def omp_loop_(stack_, background_vectors_, gene_dict_, tol_):
        g = np.empty((stack_.shape[0], stack_.shape[1], gene_dict_.shape[1] + background_vectors_.shape[1]))
        r = np.empty((stack_.shape[0], stack_.shape[1], stack_.shape[2] * stack_.shape[3]))
        for ix in numba.prange(stack_.shape[0]):
            if ix % 100 == 0:
                print(f'finished row {ix} of {stack_.shape[0]}')
            for iy in numba.prange(stack_.shape[1]):
                g[ix, iy, :], r[ix, iy, :] = omp(
                    stack_[ix, iy, :, :].flatten(),
                    gene_dict_,
                    background_vectors=background_vectors_,
                    tol=tol_
                )
        return g, r
    g,r = omp_loop_(stack, background_vectors, gene_dict, tol)

    return g[:,:,n_background_vectors:], g[:,:,:n_background_vectors], r
