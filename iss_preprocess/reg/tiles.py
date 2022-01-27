from skimage.registration import phase_cross_correlation
import numpy as np
import scipy.fft


def phase_corr(reference, target, max_shift=None, whiten=True):
    """
    Compute phase correlation of two images.

    Args:
        reference (numpy.ndarray): reference image
        target (numpy.ndarray): target image
        max_shift (int): the range over which to search for the maximum of the
            cross-correlogram
        whiten (bool): whether or not to whiten the FFTs of the images. If True,
            the method performs phase correlation, otherwise cross correlation

    Returns:
        shift: numpy.array of the location of the peak of the cross-correlogram
        cc: numpy.ndarray of the cross-correlagram itself.

    """
    f1 = scipy.fft.fft2(reference)
    f2 = scipy.fft.fft2(target)
    if whiten:
        f1 = f1 / np.abs(f1)
        f2 = f2 / np.abs(f2)
    cc = np.abs(scipy.fft.ifft2(f1 * np.conj(f2)))
    if max_shift:
        cc[max_shift:-max_shift, max_shift:-max_shift] = 0
    cc = scipy.fft.fftshift(cc)

    shift = np.unravel_index(np.argmax(cc), reference.shape) - np.array(reference.shape)/2
    return shift, cc


def register_tiles(tiles, ch_to_align=0, reg_fraction=0.1, method='scipy',
                   normalization='phase', upsample_factor=1, offset=(456., 456.),
                   max_orthogonal_shift=20, max_shift=None):
    """
    Stitch tiles together using phase correlation registration.
    The current mean projection is used as a registration template for each z-stack.

    Args:
        tiles (DataFrame): pandas DataFrame containing individual tile data
        ch_to_align (int): index of channel to use for registration
        reg_fraction (float): fraction of tile pixels along tile boundary to use for
            registration. This should be similar to tile size * overlap ratio.
        method (str): The method of registration, if set to 'None', fixed offset values 
            will be used instead.
        normalization (str): NOTE: from scikit-image - Which form of normalization
            is better is application-dependent. For example, the phase correlation method works
            well in registering images under different illumination, but is not very
            robust to noise. In a high noise scenario, the unnormalized method may be
            preferable.
            TO CONSIDER, CHANGE DEFAULT TO = None ?
        upsample_factor (int): Factor by which overlap region is scaled up for subpixel 
        offset (tuple): An alternative fixed pixel offset for stitching
        max_orthogonal_shift (float): largest shift allowed along the
            orthogonal axis (e.g. up/down when aligning tiles left/right of
            each other. If it is exceeded, shift is set to 0. 
            Setting this too strictly will result in many bad default 0 stitches 

    Returns:
        numpy.ndarray: X x Y x C x Z array of stitched tiles.

    """
    xs = tiles.X.unique()
    ys = tiles.Y.unique()
    nx = len(xs)
    ny = len(ys)
    xpix = tiles.iloc[0].data.shape[0]
    ypix = tiles.iloc[0].data.shape[1]
    reg_pix = int(xpix * reg_fraction)

    # calculate shifts
    tile_pos = np.zeros((2, nx, ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            #Creating ragged nested ndarrays is deprecated. Suggested fix is to make dtype=object
            this_tile = tiles[
                (tiles['X'] == xs[ix]) &
                (tiles['Y'] == ys[iy]) &
                (tiles['C'] == ch_to_align)
            ]['data'].to_numpy(dtype=object)
            this_tile = np.stack(this_tile, axis=2).max(axis=2)
            # align tiles in rows left to right
            if ix+1<nx:
                east_tile = tiles[
                    (tiles['X'] == xs[ix+1]) &
                    (tiles['Y'] == ys[iy]) &
                    (tiles['C'] == ch_to_align)
                ]['data'].to_numpy(dtype=object)
                east_tile = np.stack(east_tile, axis=2).max(axis=2)
                if method == 'scipy':
                    shift = phase_cross_correlation(
                        this_tile[:, -reg_pix:],
                        east_tile[:, :reg_pix],
                        upsample_factor=upsample_factor,
                        normalization=normalization
                    )[0] + [0, ypix-reg_pix]
                    if np.abs(shift[0]) > max_orthogonal_shift:
                        shift[0] = 0
                elif method == 'custom':
                    shift = phase_corr(
                        this_tile[:, -reg_pix:],
                        east_tile[:, :reg_pix],
                        max_shift=max_shift,
                        whiten=True if normalization == 'phase' else False
                    )[0] + [0, ypix-reg_pix]
                else:
                    # limit the maximum y shift
                    shift = [0, offset[1]]
                tile_pos[:, ix+1, iy] = shift + tile_pos[:, ix, iy]
            # align first tile in each row to the one above
            if ix==0 and iy+1<ny:
                south_tile = tiles[
                    (tiles['X'] == xs[ix]) &
                    (tiles['Y'] == ys[iy+1]) &
                    (tiles['C'] == ch_to_align)
                ]['data'].to_numpy(dtype=object)
                south_tile = np.stack(south_tile, axis=2).max(axis=2)
                if method == 'scipy':
                    shift = phase_cross_correlation(
                        this_tile[-reg_pix:, :],
                        south_tile[:reg_pix, :],
                        upsample_factor=upsample_factor,
                        normalization=normalization
                    )[0] + [xpix-reg_pix, 0]
                    if np.abs(shift[1]) > max_orthogonal_shift:
                        shift[1] = 0
                elif method == 'custom':
                    shift = phase_corr(
                        this_tile[-reg_pix:, :],
                        south_tile[:reg_pix, :],
                        max_shift=max_shift,
                        whiten=True if normalization == 'phase' else False
                    )[0] + [xpix-reg_pix, 0]
                else:
                    shift = [offset[0], 0]
                tile_pos[:, ix, iy+1] = shift + tile_pos[:, ix, iy]
    # round shifts and make sure there are no negative values
    tile_pos = np.rint(tile_pos).astype(int)
    tile_pos[0,:,:] = tile_pos[0,:,:] - np.min(tile_pos[0,:,:])
    tile_pos[1,:,:] = tile_pos[1,:,:] - np.min(tile_pos[1,:,:])
    # make stitched stack
    channels = tiles.C.unique()
    zs = tiles.Z.unique()
    nch = len(channels)
    nz = len(zs)
    im = np.zeros((
        np.max(tile_pos[0])+xpix,
        np.max(tile_pos[1])+ypix,
        nch,
        nz))
    for ich, ch in enumerate(channels):
        for iz, z in enumerate(zs):
            for ix, x in enumerate(xs):
                for iy, y in enumerate(ys):
                    this_tile = tiles[
                        (tiles['X'] == xs[ix]) &
                        (tiles['Y'] == ys[iy]) &
                        (tiles['C'] == ch) &
                        (tiles['Z'] == z)
                    ]['data'].to_numpy(dtype=object)[0]
                    im[tile_pos[0,ix,iy]:tile_pos[0,ix,iy]+xpix,
                        tile_pos[1,ix,iy]:tile_pos[1,ix,iy]+ypix, ich, iz] = this_tile
    return im, tile_pos
