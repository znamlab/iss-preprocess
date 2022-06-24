import numpy as np
from ..image import correct_offset
from ..reg import register_tiles
from ..io import get_tiles, reorder_channels


def load_image(fname, ops):
    tiles, metadata = get_tiles(fname)
    tiles = correct_offset(tiles, method='metadata', metadata=metadata)
    im, tile_pos = register_tiles(
        tiles,
        reg_fraction=ops['tile_reg_fraction'],
        method='custom',
        max_shift=ops['max_shift'],
        ch_to_align=-1
    )
    im = reorder_channels(im, metadata)
    if im.ndim>3:
        im = np.mean(im, axis=3)
    return im