import numpy as np


class ROI:
    """
    ROI class to manage binary masks of regions of interest.

    Stores locations of included pixels rather than full masks for memory
    efficiency.

    """

    def __init__(self, xpix=None, ypix=None, shape=None, mask=None, trace=None):
        """
        Create an ROI object.

        Can be instantiated either by providing a `mask` or `xpix`, `ypix`, and
        `shape`. The latter are ignored if mask is provided.

        Args:
            xpix (numpy.array): x-indices of mask pixels
            ypix (numpy.array): numpy.array of y-indices of mask pixels
            shape (tuple): shape of the image to mask
            mask (numpy.ndarray): binary mask of included pixels.
            trace (numpy.ndarray): fluorescence trace of the ROI.

        """
        if mask is None:
            assert xpix.size == ypix.size
            assert np.max(xpix) < shape[0]
            assert np.max(ypix) < shape[1]
            self.xpix = xpix
            self.ypix = ypix
            self.shape = shape
        else:
            self.mask = mask
        self.trace = trace

    @property
    def mask(self):
        """Binary mask of the ROI"""
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.xpix, self.ypix] = True
        return mask

    @mask.setter
    def mask(self, new_mask):
        """
        Updates binary mask

        `self.trace` is set to None in case it is no longer valid.

        """
        self.xpix, self.ypix = np.nonzero(new_mask)
        self.shape = new_mask.shape
        self.trace = None

    @property
    def npix(self):
        """Number of pixels in the ROI"""
        return self.xpix.size
