import numpy as np
import pandas as pd
from ..image import correct_offset, fstack_channels, filter_stack
from ..reg import register_tiles
from ..io import load_stack, get_tiles, get_tile_ome, reorder_channels, write_stack
from ..segment import detect_isolated_spots
from ..call import extract_spots, make_gene_templates


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


def setup_omp(fname, nchannels=4, nrounds=7):
    stack = load_stack(fname)
    stack = np.reshape(stack, (stack.shape[0], stack.shape[1], nchannels, nrounds))
    stack = np.moveaxis(stack, 2, 3)
    stack = filter_stack(stack)

    spots = detect_isolated_spots(
        np.reshape(stack, (stack.shape[0], stack.shape[1], -1)),
        detection_threshold=40,
        isolation_threshold=30
    )

    rois = extract_spots(spots, stack)
    codebook = pd.read_csv(
        '/Users/znamenp/code/iss-preprocess/iss_preprocess/call/codebook_83gene_pool.csv',
        header=None,
        names=['gii', 'seq', 'gene']
    )
    gene_dict, unique_genes = make_gene_templates(rois, codebook, vis=True)

    norm_shift = np.sqrt(
        np.median(
            np.sum(
                np.reshape(stack,(stack.shape[0], stack.shape[1], -1))**2,
                axis=2
            )
        )
    )
    return gene_dict, unique_genes, norm_shift


def project_tile(fnames):
    for fname in fnames:
        print(f'loading {fname}')
    im = get_tile_ome(fname + '.ome.tif', fname + '_metadata.txt')
    print('computing projection')
    im_proj = fstack_channels(im, sth=10)
    #im_proj = np.max(im, axis=3)
    write_stack(im_proj, fname + '_proj.tif', bigtiff=True)