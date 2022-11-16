import numpy as np
import pandas as pd
import glob
import multiprocessing as mp
from flexiznam.config import PARAMETERS
from pathlib import Path
from os.path import isfile

from ..image import correct_offset, fstack_channels, filter_stack
from ..reg import register_tiles
from ..io import load_stack, get_tiles, get_tile_ome, reorder_channels, write_stack
from ..segment import detect_isolated_spots
from ..call import extract_spots, make_gene_templates


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


def check_files(data_path, nrounds=7):
    """Check that TIFFs are present for all imaging rounds and return their list

    Args:
        data_path (str): relative path to the raw data
        nrounds (int, optional): number of sequencing rounds to look for

    Returns:
        bool: whether matching TIFFs for found for all rounds
        list: list of tiff paths for round 1

    """
    raw_path = Path(PARAMETERS['data_root']['raw'])
    data_path = raw_path / data_path

    tiffs = sorted(glob.glob(str(data_path / 'round_01_1/*.tif')))
    success = True
    # check that all files exist
    for iround in range(nrounds):
        for tiff in tiffs:
            fname = tiff.replace('round_01', f'round_{str(iround+1).zfill(2)}')
            if not isfile(fname):
                print(f'{fname} does not exist')
                success = False
    return success, tiffs


def project_tile(fname, overwrite=False):
    """Calculates extended depth of field and max intensity projections for a single tile.

    Args:
        fname (str): path to tile *without* `'.ome.tif'` extension.
        overwrite (bool): whether to repeat if already completed

    """
    raw_path = Path(PARAMETERS['data_root']['raw'])
    processed_path = Path(PARAMETERS['data_root']['processed'])
    save_path_fstack = processed_path / (fname + '_proj.tif')
    save_path_max = processed_path / (fname + '_max.tif')
    if save_path_fstack.exists() or save_path_max.exists():
        print(f'{fname} already projected...\n')
        return
    print(f'loading {fname}\n')
    im = get_tile_ome(raw_path / (fname + '.ome.tif'), raw_path / (fname + '_metadata.txt'))
    print('computing projection\n')
    im_fstack = fstack_channels(im, sth=8)
    im_max = np.max(im, axis=3)
    (processed_path / fname).parent.mkdir(parents=True, exist_ok=True)
    write_stack(im_fstack, save_path_fstack, bigtiff=True)
    write_stack(im_max, save_path_max, bigtiff=True)


def load_processed_tile(data_path, tile_coors=(1,0,0), nrounds=7, suffix='proj'):
    """Load processed tile images across rounds 

    Args:
        data_path (str): relative path to dataset.
        tile_coors (tuple, optional): Coordinates of tile to load: ROI, Xpos, Ypos. 
            Defaults to (1,0,0).
        nrounds (int, optional): Number of rounds to load. Defaults to 7.
        suffix (str, optional): File name suffix. Defaults to '_proj'.

    Returns:
        numpy.ndarray: X x Y x channels x rounds stack.

    """
    tile_roi, tile_x, tile_y = tile_coors
    processed_path = Path(PARAMETERS['data_root']['processed'])
    ims = []
    for iround in range(nrounds):
        dirname = f'round_{str(iround+1).zfill(2)}_1'
        fname = f'round_{str(iround+1).zfill(2)}_1_MMStack_{tile_roi}-' + \
            f'Pos{str(tile_x).zfill(3)}_{str(tile_y).zfill(3)}_{suffix}.tif'
        ims.append(load_stack(processed_path / data_path / dirname / fname))
    return np.stack(ims, axis=3)


def preprocess_images(data_path, nrounds=7):
    success, tiffs = check_files(data_path, nrounds=nrounds)
    if not success:
        print('Some files were missing... aborting...')
        return

    fnames = []
    for tiff in tiffs:
        for iround in range(nrounds):
            fname = tiff \
                .replace('round_01', f'round_{str(iround+1).zfill(2)}') \
                .replace('.ome.tif', '')
            fnames.append(str(Path(fname).relative_to(PARAMETERS['data_root']['raw'])))

    max_workers = 16
    pool = mp.Pool(np.min((mp.cpu_count(), max_workers)))
    results = pool.map(project_tile, fnames)
    pool.close()