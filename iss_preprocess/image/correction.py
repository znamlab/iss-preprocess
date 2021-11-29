import numpy as np
from sklearn.mixture import GaussianMixture

def correct_offset(tiles, n_components=5, method='min'):
    """
    Estimate image offset for each channel as the minimum value or using a
    Gaussian mixture model and substract it from input images.

    """
    channels = tiles.C.unique()
    for channel in channels:
        this_channel = tiles[(tiles['C'] == channel) & (tiles['Z']==0)]['data']
        data = np.concatenate(this_channel.to_numpy()).reshape(-1, 1)
        if method=='min':
            offset = np.min(data)
        else:
            gm = GaussianMixture(n_components=n_components, random_state=0).fit(
                data[:10:,:]
            )
            offset = np.min(gm.means_)
        v = tiles[tiles['C'] == channel]['data'].transform(lambda x: x-offset)
        tiles.update(v)
    return tiles
