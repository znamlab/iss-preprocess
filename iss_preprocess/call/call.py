import numpy as np
from math import ceil
from skimage.draw import disk
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import hamming
import matplotlib.pyplot as plt
from ..coppafish import scaled_k_means

# BASES = np.array(['G','T','A','C'])
# BASES = np.array(["A", "C", "T", "G"])
BASES = np.array(["G", "T", "A", "C"])


def get_cluster_means(spots, vis=False, score_thresh=0):
    x = np.stack(spots["trace"], axis=2)  # round x channels x spots
    nrounds = x.shape[0]
    nch = x.shape[1]
    if vis:
        _, ax1 = plt.subplots(nrows=1, ncols=nch)
        _, ax2 = plt.subplots(nrows=2, ncols=ceil(nrounds / 2))
    cluster_means = []
    cluster_means = []
    cluster_intensity = np.zeros((nrounds, nch))
    for iround in range(nrounds):
        norm_cluster_mean, _, cluster_ind, _, _, _ = scaled_k_means(
            x[iround, :, :].T, np.eye(nch), score_thresh=score_thresh
        )
        cluster_means.append(norm_cluster_mean)
        for ich in range(nch):
            # TODO: should this be a norm instead of a mean?
            cluster_intensity[iround, ich] = np.mean(x[iround, ich, cluster_ind == ich])
        if vis:
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
    if vis:
        for ich in range(nch):
            plt.sca(ax1[ich])
            plt.imshow(np.stack(cluster_means, axis=2)[ich, :, :])
            plt.xlabel("rounds")
            plt.ylabel("channels")
            plt.title(f"channel {ich+1}")
        plt.tight_layout()
    return cluster_means


def extract_spots(spots, stack):
    """
    Extract fluorescence traces of spots and assign them to a column of the DataFrame.

    Args:
        spots (pandas.DataFrame):
        stack (numpy.ndarray): X x Y x R x C stack.

    """
    traces = []
    for _, spot in spots.iterrows():
        rr, cc = disk((spot["y"], spot["x"]), spot["size"], shape=stack.shape[0:2])
        traces.append(stack[rr, cc, :, :].mean(axis=0).T)
    spots["trace"] = traces


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


def correct_barcode_sequences(spots, max_edit_distance=2):
    sequences = np.stack(spots["sequence"].to_numpy())
    unique_sequences, counts = np.unique(sequences, axis=0, return_counts=True)
    # sort sequences according to abundance
    order = np.flip(np.argsort(counts))
    unique_sequences = unique_sequences[order]
    counts = counts[order]

    corrected_sequences = unique_sequences.copy()
    reassigned = np.zeros(corrected_sequences.shape[0])
    for i, sequence in enumerate(unique_sequences):
        # if within edit distance and lower in the list (i.e. lower abundance),
        # then update the sequence
        edit_distance = np.sum((unique_sequences - sequence) != 0, axis=1)
        sequences_to_correct = np.logical_and(
            edit_distance <= max_edit_distance, np.logical_not(reassigned)
        )
        sequences_to_correct[: i + 1] = False
        corrected_sequences[sequences_to_correct, :] = sequence
        reassigned[sequences_to_correct] = True

    for original_sequence, new_sequence in zip(unique_sequences, corrected_sequences):
        if not np.array_equal(original_sequence, new_sequence):
            sequences[
                np.all((sequences - original_sequence) == 0, axis=1), :
            ] = new_sequence

    spots["corrected_sequence"] = [seq for seq in sequences]
    spots["corrected_bases"] = [
        "".join(BASES[seq]) for seq in spots["corrected_sequence"]
    ]
    return spots
