import numpy as np
from sklearn.mixture import GaussianMixture

def correct_offset(tiles, method='metadata', metadata=None, n_components=5):
    """
    Estimate image offset for each channel as the minimum value or using a
    Gaussian mixture model and substract it from input images.

    Args:
        tiles (DataFrame): individual tiles
        method (str): method for determining the offset, one of either:
            `metadata`: uses the values recorded in the image metadata
            `min`: uses the minimum for each channel
            `gmm`: fits a Gaussian mixture model and uses the smallest mean
        metadata (ElementTree): XML element tree with

    """
    if metadata:
        channels_metadata = metadata.findall(
            './Metadata/Information/Image/Dimensions/Channels/Channel'
        )

    channels = tiles.C.unique()
    for channel in channels:
        this_channel = tiles[(tiles['C'] == channel) & (tiles['Z']==0)]['data']
        data = np.concatenate(this_channel.to_numpy()).reshape(-1, 1)
        if method=='metadata' and metadata:
            offset = float(channels_metadata[channel].find('./DetectorSettings/Offset').text)
        elif method=='min':
            offset = np.min(data)
        else:
            gm = GaussianMixture(n_components=n_components, random_state=0).fit(
                data[:10:,:]
            )
            offset = np.min(gm.means_)
        v = tiles[tiles['C'] == channel]['data'].transform(lambda x: x.astype(float)-offset)
        tiles.update(v)
    return tiles    
