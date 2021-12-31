from skimage.draw import disk
import iss_preprocess as iss
import numpy as np
from sklearn.mixture import GaussianMixture


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
        rr, cc = disk([spot['y'], spot['x']], spot['size'], shape=stack.shape[0:2])
        roi = iss.segment.ROI(xpix=rr, ypix=cc, shape=stack.shape[0:2])
        roi.trace = stack[roi.xpix,roi.ypix,:,:].mean(axis=0)
        rois.append(roi)

    return rois


def basecall_rois(rois, separate_rounds=True, rounds=[]):
    """
    Assign bases using a Gaussian Mixture Model.

    Args:
        rois (list): list of ROI objects.
        separate_rounds (bool): whether to run basecalling separately on each
            round or on all rounds together. Default True.
        rounds: numpy.array of rounds to include.

    Returns:
        ROIs x rounds of base IDs.

    """
    def predict_bases(data):
        gmm = GaussianMixture(n_components=4, random_state=0).fit(data)
        # GMM components are arbitrarily ordered. We assign each component to a
        # based on its maximum channel.
        base_id = np.argmax(gmm.means_, axis=1)
        labels = gmm.predict(data)
        return base_id[labels]

    # rounds x channels x rois matrix
    x = np.stack([roi.trace for roi in rois], axis=2)
    # normalize by mean intensity
    x = x / np.mean(x, axis=1)[:,np.newaxis,:]
    if rounds:
        x = x[rounds,:,:]
    if separate_rounds:
        bases = np.empty((x.shape[2], x.shape[0]))
        for round in range(x.shape[0]):
            data = x[round,:,:].transpose()
            bases[:, round] = predict_bases(data)
    else:
        data = np.moveaxis(x, 0, 2).reshape((4,-1)).transpose()
        bases = predict_bases(data)
        bases = np.reshape(bases,(x.shape[2], x.shape[0]))

    return bases