from skimage.registration import phase_cross_correlation
import numpy as np
import scipy.fft

def phase_corr(im1, im2):
    f1 = fft.fft2(im1)
    f2 = fft.fft2(im2)
    wf1 = f1 / np.abs(f1)
    wf2 = f2 / np.abs(f2)
    cc = fft.ifft2(wf1 * np.conj(wf2))
    return np.unravel_indx(np.amax(cc), im1.shape)

def register_tiles(tiles, ch_to_align=0, reg_pix=64, overlap_ratio=0.9,
    method='phasecorr', offset=(456., 456.)):
    """
    Stitch tiles together using phase correlation registration.

    Current mean projection is used as a registration template for each z-stack.

    Args:
        tiles (DataFrame): pandas DataFrame containing individual tile data
        ch_to_align (int): index of channel to use for registration
        reg_pix (int): number of pixels along tile boundary to use for
            registration. This should be similar to tile size * overlap ratio.

    Returns:
        numpy.ndarray: X x Y x C x Z array of stitched tiles.
    """
    xs = tiles.X.unique()
    ys = tiles.Y.unique()
    nx = len(xs)
    ny = len(ys)
    xpix = tiles.iloc[0].data.shape[0]
    ypix = tiles.iloc[0].data.shape[1]
    # calculate shifts
    tile_pos = np.zeros((2, nx, ny))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            this_tile = tiles[
                (tiles['X'] == xs[ix]) &
                (tiles['Y'] == ys[iy]) &
                (tiles['C'] == ch_to_align)
            ]['data'].to_numpy()
            this_tile = np.stack(this_tile, axis=2).max(axis=2)
            # align tiles in rows left to right
            if ix+1<nx:
                east_tile = tiles[
                    (tiles['X'] == xs[ix+1]) &
                    (tiles['Y'] == ys[iy]) &
                    (tiles['C'] == ch_to_align)
                ]['data'].to_numpy()
                east_tile = np.stack(east_tile, axis=2).max(axis=2)
                if method=='phasecorr':
                    shift = phase_cross_correlation(
                        this_tile[:, -reg_pix:],
                        east_tile[:, :reg_pix],
                        upsample_factor=5,
                        overlap_ratio=overlap_ratio
                    )[0] + [0, ypix-reg_pix]
                    if shift[0]>5:
                        shift[0]=0
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
                ]['data'].to_numpy()
                south_tile = np.stack(south_tile, axis=2).max(axis=2)
                if method=='phasecorr':
                    shift = phase_cross_correlation(
                        this_tile[-reg_pix:, :],
                        south_tile[:reg_pix, :],
                        upsample_factor=5,
                        overlap_ratio=overlap_ratio
                    )[0] + [xpix-reg_pix, 0]
                    if shift[1]>5:
                        shift[1]=0
                else:
                    shift = [offset[0], 0]
                tile_pos[:, ix, iy+1] = shift + tile_pos[:, ix, iy]
    # round shifts and make sure there are no negative values
    tile_pos_float = tile_pos.copy()
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
                    ]['data'].to_numpy()[0]
                    im[tile_pos[0,ix,iy]:tile_pos[0,ix,iy]+xpix,
                        tile_pos[1,ix,iy]:tile_pos[1,ix,iy]+ypix, ich, iz] = this_tile
    return im, tile_pos
