import numba
import numpy as np
from . import basecall_rois, call_genes


def make_gene_templates(rois, codebook, max_errors=0):
    """

    Args:
        rois:
        codebook:
        max_errors:
        min_spots:

    Returns:

    """
    bases = basecall_rois(rois, separate_rounds=False, rounds=range(7))
    genes, errors = call_genes(bases, codebook)
    errors = np.array(errors)
    genes = np.array(genes)
    genes[errors > max_errors] = 'nan'
    unique_genes = np.unique(genes)
    unique_genes = unique_genes[unique_genes != 'nan']
    # rounds x channels x rois matrix
    f = np.stack([roi.trace for roi in rois], axis=2)
    # normalize by mean intensity
    # f = f / np.mean(f, axis=1)[:,np.newaxis,:] # not sure if this makes sense here?
    f = np.moveaxis(f, 2, 0)[:,:-1,:]
    f = np.reshape(f, (f.shape[0], -1))
    gene_dict = np.empty((f.shape[1], len(unique_genes)))
    for igene, gene in enumerate(unique_genes):
        gene_dict[:, igene] = f[genes == gene, :].mean(axis=0)
    gene_dict /= np.linalg.norm(gene_dict, axis=0)
    return gene_dict, unique_genes


def make_background_vectors(nrounds=7, nchannels=4):
    """

    Args:
        nrounds:
        nchannels:

    Returns:

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

    Args:
        y:
        X:
        background_vectors:
        max_comp:
        tol:

    Returns:

    """
    if not max_comp:
        max_comp = X.shape[1]
    norm_y = np.linalg.norm(y)
    r = y
    if background_vectors is not None:
        X = np.concatenate((background_vectors, X), axis=1)
    ichosen = np.zeros(X.shape[1], dtype=numba.boolean)
    if background_vectors is not None:
        ichosen[:background_vectors.shape[1]] = True
        coefs,_,_,_ = np.linalg.lstsq(X[:, ichosen], y)
        r = y - np.dot(X[:, ichosen], coefs)

    while np.sum(ichosen) < max_comp:
        not_chosen = np.nonzero(np.logical_not(ichosen))[0]
        dot_product = np.dot(X[:, not_chosen].transpose(), y)
        best_match = not_chosen[np.argmax(dot_product)]
        ichosen[best_match] = True
        coefs_new,_,_,_ = np.linalg.lstsq(X[:, ichosen], y)
        r_new = y - np.dot(X[:, ichosen], coefs_new)
        if (np.linalg.norm(r) - np.linalg.norm(r_new)) / norm_y > tol:
            coefs = coefs_new
            r = r_new
        else:
            ichosen[best_match] = False
            break

    coefs_out = np.zeros(X.shape[1])
    if np.any(ichosen):
        coefs_out[ichosen] = coefs
    return coefs_out, r


def run_omp(stack, gene_dict):
    """

    Args:
        stack:
        gene_dict:

    Returns:

    """
    background_vectors = make_background_vectors()
    n_background_vectors = background_vectors.shape[1]

    g = np.empty((stack.shape[0], stack.shape[1], gene_dict.shape[1]+n_background_vectors))
    r = np.empty((stack.shape[0], stack.shape[1], 28))
    stack = stack.astype(float)
    for ix in range(stack.shape[0]):
        if ix % 100 == 0:
            print(f'finished row {ix} of {stack.shape[0]}')
        for iy in range(stack.shape[1]):
            g[ix, iy, :], r[ix, iy, :] = omp(
                stack[ix,iy,:,:].flatten(),
                gene_dict,
                background_vectors=background_vectors
            )

    return g, r