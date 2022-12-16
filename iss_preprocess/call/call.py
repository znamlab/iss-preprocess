import numpy as np
from math import ceil
from skimage.draw import disk
from ..segment import ROI
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FastICA
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from ..coppafish import scaled_k_means

# BASES = np.array(['G','T','A','C'])
# BASES = np.array(["A", "C", "T", "G"])
BASES = np.array(["G", "T", "A", "C"])


def get_cluster_means(rois, vis=False):
    x = rois_to_array(rois, normalize=False)  # round x channels x spots
    nrounds = x.shape[0]
    nch = x.shape[1]
    if vis:
        _, ax1 = plt.subplots(nrows=2, ncols=ceil(nrounds / 2))
        _, ax2 = plt.subplots(nrows=2, ncols=ceil(nrounds / 2))
    cluster_means = []
    cluster_means = []
    cluster_intensity = np.zeros((nrounds, nch))
    for iround in range(nrounds):
        norm_cluster_mean, _, cluster_ind, _, _, _ = scaled_k_means(
            x[iround, :, :].T, np.eye(nch)
        )
        cluster_means.append(norm_cluster_mean)
        for ich in range(nch):
            cluster_intensity[iround, ich] = np.mean(x[iround, ich, cluster_ind == ich])
        if vis:
            plt.sca(ax1.flatten()[iround])
            plt.imshow(norm_cluster_mean)
            plt.xlabel("channels")
            plt.ylabel("clusters")
            plt.yticks(range(4))
            plt.title(f"round {iround+1}")
            plt.sca(ax2.flatten()[iround])
            for ich in range(nch):
                plt.plot(
                    x[iround, 0, cluster_ind == ich],
                    x[iround, 1, cluster_ind == ich],
                    ".",
                    markersize=1,
                )
    # normalize intensity to first round
    cluster_intensity = cluster_intensity / cluster_intensity[0, :]
    for iround in range(nrounds):
        for ich in range(nch):
            cluster_means[iround][ich, :] = (
                cluster_means[iround][ich, :] * cluster_intensity[iround, ich]
            )
    return cluster_means


def call_hyb_spots(spots, stack, nprobes=3, vis=False):
    """
    Assign spots to hybridization probes, adjusting for cross-talk. Fluorescence
    values are extracted and first processed using ICA. The ICA output is used
    to fit a GMM and assign components for each spot. Finally, since the order of
    GMM components is arbitrary, we assign a channel number for each component
    based on the channel where it has the highest mean fluorescence.

    Args:
        spots (DataFrame): table of spot locations
        stack (numpy.ndarray): X x Y x C stack.
        nprobes (int): number probes used. Determines the number of ICA and GMM components.
        vis (bool): whether to make diagnostic plots.

    Returns:
        numpy.array of channel IDs for each spot. IDs are selected for each component
            based on the channel with the highest mean fluorescence for spots
            assigned to this component.

    """
    hyb_rois = extract_spots(spots, stack[:, :, np.newaxis, :])
    f = rois_to_array(hyb_rois)
    # first perform ICA to separate fluorophores
    ica = FastICA(n_components=nprobes, random_state=1, max_iter=1000).fit_transform(
        f[0, :, :].T
    )
    # then cluster the ICA output with a GMM
    gmm = GaussianMixture(n_components=nprobes, random_state=0).fit(ica)
    labels = gmm.predict(ica)
    # assign each GMM component to a component based on the highest fluorescence channel
    channel_id = np.array(
        [
            np.argmax(np.mean(f[0, :, labels == label], axis=0))
            for label in np.unique(labels)
        ]
    )

    if vis:
        # visualise ICA components
        plt.figure(figsize=(10, 10))
        for xch in range(f.shape[1]):
            for ych in range(f.shape[1]):
                plt.subplot(f.shape[1], f.shape[1], xch * f.shape[1] + ych + 1)
                plt.scatter(ica[:, xch], ica[:, ych], c=labels, s=5)
        # plot component assignments for the original fluorescence values
        plt.figure(figsize=(10, 10))
        f = rois_to_array(hyb_rois, normalize=False)
        for xch in range(f.shape[1]):
            for ych in range(f.shape[1]):
                plt.subplot(f.shape[1], f.shape[1], xch * f.shape[1] + ych + 1)
                plt.scatter(f[:, xch], f[:, ych], c=labels, s=5)

    return channel_id[labels]


def extract_spots(spots, stack):
    """
    Create ROIs based on spot locations and extract their fluorescence traces.

    Args:
        spots (pandas.DataFrame):
        stack (numpy.ndarray): X x Y x R x C stack.

    Returns:
        List of ROI objects.

    """
    rois = []
    for _, spot in spots.iterrows():
        rr, cc = disk((spot["y"], spot["x"]), spot["size"], shape=stack.shape[0:2])
        roi = ROI(xpix=rr, ypix=cc, shape=stack.shape[0:2])
        roi.trace = stack[roi.xpix, roi.ypix, :, :].mean(axis=0)
        rois.append(roi)

    return rois


def rois_to_array(rois, normalize=True):
    # rounds x channels x rois matrix
    x = np.stack([roi.trace for roi in rois], axis=2)
    invalid_rois = np.any(np.mean(x, axis=1) == 0, axis=0)
    if np.any(invalid_rois):
        print(
            """
            WARNING: Zeros encountered for all channels in some ROIs.
            This normally occurs when an ROI is not imaged on all rounds.
            Corresponding values of the fluorescence matrix will be set to NaN.
            """
        )
        x[:, :, invalid_rois] = np.nan
    # normalize by mean intensity
    if normalize:
        valid_rois = np.logical_not(invalid_rois)
        x[:, :, valid_rois] /= np.mean(x[:, :, valid_rois], axis=1)[:, np.newaxis, :]
    return x


def basecall_rois(rois, separate_rounds=True, rounds=(), nsamples=None):
    """
    Assign bases using a Gaussian Mixture Model.

    Args:
        rois (list): list of ROI objects.
        separate_rounds (bool): whether to run basecalling separately on each
            round or on all rounds together. Default True.
        rounds: numpy.array of rounds to include.
        nsamples (int): number of samples to include for fitting GMM. If None,
            all data are used for fitting. Default None.

    Returns:
        ROIs x rounds of base IDs.

    """

    def predict_bases(data_, nsamples_):
        if nsamples_ and nsamples_ < data.shape[0]:
            data_idx = np.random.choice(data_.shape[0], nsamples_, replace=False)
            gmm = GaussianMixture(n_components=4, random_state=0).fit(
                data_[data_idx, :]
            )
        else:
            gmm = GaussianMixture(n_components=4, random_state=0).fit(data_)
        # GMM components are arbitrarily ordered. We assign each component to a
        # base based on its maximum channel.
        base_id = np.argmax(gmm.means_, axis=1)  # the base id for each component
        base_means = gmm.means_[np.argsort(base_id), :]
        labels = gmm.predict(data_)
        return base_id[labels], base_means

    x = rois_to_array(rois)
    if rounds:
        x = x[rounds, :, :]
    if separate_rounds:
        bases = np.empty((x.shape[2], x.shape[0]), dtype=int)
        base_means = np.empty((x.shape[1], x.shape[1], x.shape[0]))
        for round in range(x.shape[0]):
            data = x[round, :, :].transpose()
            bases[:, round], base_means[:, :, round] = predict_bases(data, nsamples)
    else:
        data = np.moveaxis(x, 0, 2).reshape((4, -1)).transpose()
        bases, base_means = predict_bases(data, nsamples)
        bases = np.reshape(bases, (x.shape[2], x.shape[0]))

    return bases, base_means, x


def call_genes(sequences, codebook):
    """
    Assigns sequences to genes based on the provided codebook.

    Args:
        sequences (numpy.ndarray): ROIs x rounds array of base IDs generated by
            `basecall_rois`.
        codebook (pandas.DataFrame): gene codes, containing 'gii', 'seq', and 'gene'
            columns.

    Returns:
        List of most closely matching gene names.
        List of edit distances.

    """
    genes = []
    errors = []
    for s in sequences:
        seq = BASES[s]
        dist_series = codebook["seq"].apply(lambda x: hamming(list(x), seq) * len(x))
        dist = dist_series.min()
        genes.append(codebook.iloc[dist_series.argmin()]["gene"])
        errors.append(dist)
    return genes, errors
