import numpy as np
import pandas as pd
import glob
import multiprocessing as mp
from skimage.registration import phase_cross_correlation
from flexiznam.config import PARAMETERS
from pathlib import Path
from os.path import isfile
from ..image import fstack_channels, filter_stack
from ..reg import register_channels_and_rounds, align_channels_and_rounds
from ..io import load_stack, get_tile_ome, write_stack
from ..segment import detect_isolated_spots
from ..call import extract_spots, make_gene_templates, run_omp, find_gene_spots


def setup_omp(stack, codebook_name='codebook_83gene_pool.csv', detection_threshold=40,
                isolation_threshold=30):
    """Prepare variables required to run the OMP algorithm.

    Args:
        stack (numpy.ndarray): X x Y x R x C image stack.

    Returns:
        numpy.ndarray: N x M dictionary, where N = R * C and M is the
            number of genes.
        list: gene names.
        float: norm shift for the OMP algorithm, estimated as median norm of all pixels.

    """
    spots = detect_isolated_spots(
        np.reshape(stack, (stack.shape[0], stack.shape[1], -1)),
        detection_threshold=detection_threshold,
        isolation_threshold=isolation_threshold
    )
    
    rois = extract_spots(spots, stack)
    codebook = pd.read_csv(
        Path(__file__).parent.parent / 'call' / codebook_name,
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


def project_tile_by_coors(data_path, prefix, tile_coors, overwrite=False):
    fname = f'{prefix}_MMStack_{tile_coors[0]}-Pos{str(tile_coors[1]).zfill(3)}_{str(tile_coors[2]).zfill(3)}'
    tile_path = str(Path(data_path) / prefix / fname)
    project_tile(tile_path, overwrite=overwrite)


def project_tile(fname, overwrite=False, sth=8):
    """Calculates extended depth of field and max intensity projections for a single tile.

    Args:
        fname (str): path to tile *without* `'.ome.tif'` extension.
        overwrite (bool): whether to repeat if already completed

    """
    raw_path = Path(PARAMETERS['data_root']['raw'])
    processed_path = Path(PARAMETERS['data_root']['processed'])
    save_path_fstack = processed_path / (fname + '_fstack.tif')
    save_path_max = processed_path / (fname + '_max.tif')
    if not overwrite and (save_path_fstack.exists() or save_path_max.exists()):
        print(f'{fname} already projected...\n')
        return
    print(f'loading {fname}\n')
    im = get_tile_ome(raw_path / (fname + '.ome.tif'), raw_path / (fname + '_metadata.txt'))
    print('computing projection\n')
    im_fstack = fstack_channels(im, sth=sth)
    im_max = np.max(im, axis=3)
    (processed_path / fname).parent.mkdir(parents=True, exist_ok=True)
    write_stack(im_fstack, save_path_fstack, bigtiff=True)
    write_stack(im_max, save_path_max, bigtiff=True)


def load_processed_tile(data_path, tile_coors=(1,0,0), nrounds=7, suffix='proj', prefix='round'):
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
        dirname = f'{prefix}_{str(iround+1).zfill(2)}_1'
        fname = f'{prefix}_{str(iround+1).zfill(2)}_1_MMStack_{tile_roi}-' + \
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


def register_reference_tile(data_path, tile_coors=(0,0,0)):
    stack = load_processed_tile(data_path, tile_coors)
    tforms = register_channels_and_rounds(stack)

    processed_path = Path(PARAMETERS['data_root']['processed'])
    save_path = processed_path / data_path / 'tforms.npy'
    np.save(save_path, tforms, allow_pickle=True)


def load_and_register_tile(data_path, tile_coors=(0,0,0), suffix='proj', filter_r=(2,4)):
    processed_path = Path(PARAMETERS['data_root']['processed'])
    tforms_path = processed_path / data_path / 'tforms.npy'
    tforms = np.load(tforms_path, allow_pickle=True)

    stack = load_processed_tile(data_path, tile_coors, suffix=suffix)

    stack = align_channels_and_rounds(stack, tforms)
    bad_pixels = np.any(np.isnan(stack), axis=(2,3))
    stack[np.isnan(stack)] = 0

    stack = np.moveaxis(stack, 2, 3)
    stack = filter_stack(stack, r1=filter_r[0], r2=filter_r[1])
    return stack, bad_pixels


def run_omp_on_tile(data_path, tile_coors, save_stack=False):
    processed_path = Path(PARAMETERS['data_root']['processed'])
    ops_path = processed_path / data_path / 'ops.npy'
    ops = np.load(ops_path, allow_pickle=True).item()

    stack, bad_pixels = load_and_register_tile(data_path, tile_coors, suffix=ops['projection'])

    if save_stack:
        save_dir = processed_path / data_path / 'reg'
        save_dir.mkdir(parents=True, exist_ok=True)
        stack_path = save_dir / f'tile_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.tif'
        write_stack(stack.copy(), stack_path, bigtiff=True)

    omp_stat = np.load(processed_path / data_path / 'gene_dict.npz', allow_pickle=True)
    g, b, r = run_omp(
        stack,
        omp_stat['gene_dict'],
        tol=ops['omp_threshold'],
        weighted=True,
        refit_background=True,
        alpha=200.,
        norm_shift=omp_stat['norm_shift'],
        max_comp=12
    )

    for igene in range(g.shape[2]):
        g[bad_pixels, igene] = 0

    spot_image_path = processed_path / data_path / 'spot_sign_image.npy'
    if spot_image_path.exists():
        spot_sign_image = np.load(spot_image_path)
    else:
        print('No spot sign image for this dataset - using default.')
        spot_sign_image = np.load(Path(__file__).parent.parent / 'call/spot_signimage.npy')
        
    spot_sign_threshold = 0.15
    spot_sign_image[np.abs(spot_sign_image) < spot_sign_threshold] = 0

    gene_spots = find_gene_spots(g, spot_sign_image, 
        rho=ops['spot_rho'], omp_score_threshold=ops['spot_threshold'])

    for df, gene in zip(gene_spots, omp_stat['gene_names']):
        df['gene'] = gene
    save_dir = processed_path / data_path / 'spots'
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(gene_spots).to_pickle(
        save_dir / f'gene_spots_{tile_coors[0]}_{tile_coors[1]}_{tile_coors[2]}.pkl'
    )


def register_adjacent_tiles(data_path, ref_coors=(1,0,0), reg_fraction=0.1,
                            ref_ch=0, ref_round=0):
    """Estimate shift between adjacent imaging tiles using phase correlation.

    Args:
        data_path (str): path to image stacks.
        ref_coors (tuple, optional): coordinates of the reference tile to use for
            registration. Must not be along the bottom or right edge of image. Defaults to (1,0,0).
        reg_fraction (float, optional): overlap fraction used for registration. Defaults to 0.1.
        ref_ch (int, optional): reference channel used for registration. Defaults to 0.
        ref_round (int, optional): reference round used for registration. Defaults to 0.

    Returns:
        numpy.array: `shift_right`, X and Y shifts between different columns
        numpy.array: `shift_down`, X and Y shifts between different rows
        numpy.array: shape of the tile

    """
    tile_ref = load_processed_tile(data_path, ref_coors)
    down_coors = (ref_coors[0], ref_coors[1], ref_coors[2]+1)
    tile_down = load_processed_tile(data_path, down_coors)
    right_coors = (ref_coors[0], ref_coors[1]+1, ref_coors[2])
    tile_right = load_processed_tile(data_path, right_coors)

    ypix = tile_ref.shape[0]
    xpix = tile_ref.shape[1]
    reg_pix_x = int(xpix * reg_fraction)
    reg_pix_y = int(ypix * reg_fraction)

    shift_right = phase_cross_correlation(
        tile_ref[:, -reg_pix_x:, ref_ch, ref_round],
        tile_right[:, :reg_pix_x, ref_ch, ref_round],
        upsample_factor=5
    )[0] + [0, xpix-reg_pix_x]

    shift_down = phase_cross_correlation(
        tile_ref[:reg_pix_y, :, ref_ch, ref_round],
        tile_down[-reg_pix_y:, :, ref_ch, ref_round],
        upsample_factor=5
    )[0] - [ypix-reg_pix_y, 0]

    return shift_right, shift_down, (ypix, xpix)


def calculate_tile_positions(shift_right, shift_down, tile_shape, ntiles):
    """Calculate position of each tile based on the provided shifts.

    Args:
        shift_right (numpy.array): X and Y shifts between different columns
        shift_down (numpy.array): X and Y shifts between different rows
        tile_shape (numpy.array): shape of each tile
        ntiles (numpy.array): number of tile rows and columns

    Returns:
        numpy.ndarray: `tile_origins`, ntiles[0] x ntiles[1] x 2 matrix of tile origin coordinates
        numpy.ndarray: `tile_centers`, ntiles[0] x ntiles[1] x 2 matrix of tile center coordinates

    """
    tile_centers = np.empty((ntiles[0], ntiles[1], 2))
    tile_origins = np.empty((ntiles[0], ntiles[1], 2))

    center_offset = np.array([tile_shape[0]/2, tile_shape[1]/2])
    for ix in range(ntiles[0]):
        for iy in range( ntiles[1]):
            tile_origins[ix, iy, :] = iy * shift_down + ix * shift_right
    tile_origins = tile_origins - np.min(tile_origins, axis=(0,1))[np.newaxis, np.newaxis, :]
    tile_centers = tile_origins + center_offset[np.newaxis, np.newaxis, :]
    return tile_origins, tile_centers


def stitch_tiles(data_path, prefix, tile_origins, tile_shape, roi=1, suffix='proj', ich=0):
    """Load and stitch tile images using provided tile origin locations.

    Args:
        data_path (str): path to image stacks.
        prefix (str): prefix specifying which images to load, e.g. 'round_01_1'
        tile_origins (numpy.ndarray): matrix of tile origin coordinates generated by `calculate_tile_positions`
        tile_shape (numpy.array): shape of each tile
        roi (int, optional): id of ROI to load. Defaults to 1.
        suffix (str, optional): filename suffix. Defaults to 'proj'.
        ich (int, optional): index of the channel to stitch. Defaults to 0.

    Returns:
        numpy.ndarray: stitched image.
        
    """
    processed_path = Path(PARAMETERS['data_root']['processed'])
    ntiles = tile_origins.shape[:2]
    tile_origins = tile_origins.astype(int)
    max_origin = np.max(tile_origins, axis=(0,1))
    stitched_stack = np.zeros(max_origin + tile_shape)
    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            fname = f'{prefix}_MMStack_{roi}-Pos{str(ix).zfill(3)}_{str(iy).zfill(3)}_{suffix}.tif'
            stack = load_stack(processed_path / data_path / prefix / fname)
            stitched_stack[tile_origins[ix, iy, 0]:tile_origins[ix, iy, 0]+tile_shape[0],
                           tile_origins[ix, iy, 1]:tile_origins[ix, iy, 1]+tile_shape[1]] = stack[:,:,ich]
    return stitched_stack


def merge_roi_spots(data_path, shift_right, shift_down, tile_shape, ntiles, iroi=1):
    """Load and combine spot locations across all tiles for an ROI.

    Args:
        data_path (str): path to pickle files containing spot locations for each tile.
        shift_right (numpy.array): X and Y shifts between different columns
        shift_down (numpy.array): X and Y shifts between different rows
        tile_shape (numpy.array): shape of each tile
        ntiles (numpy.array): number of tile rows and columns
        iroi (int, optional): ID of ROI to load. Defaults to 1.

    Returns:
        pandas.DataFrame: table containing spot locations across all tiles.
    """
    processed_path = Path(PARAMETERS['data_root']['processed'])
    all_spots = []
    tile_origins, tile_centers = calculate_tile_positions(shift_right, shift_down, tile_shape, ntiles)

    for ix in range(ntiles[0]):
        for iy in range(ntiles[1]):
            spots = pd.read_pickle(processed_path / data_path / 'spots' / f'gene_spots_{iroi}_{ix}_{iy}.pkl')
            spots['x'] = spots['x'] + tile_origins[ix, iy, 1]
            spots['y'] = spots['y'] + tile_origins[ix, iy, 0]

            spot_dist = (spots['x'].to_numpy()[:, np.newaxis, np.newaxis] - tile_centers[np.newaxis, :, :, 1]) ** 2 + \
                    (spots['y'].to_numpy()[:, np.newaxis, np.newaxis] - tile_centers[np.newaxis, :, :, 0]) ** 2
            home_tile_dist = (spot_dist[:, ix, iy]).copy()
            spot_dist[:, ix, iy] = np.inf
            min_spot_dist = np.min(spot_dist, axis=(1,2))
            keep_spots = home_tile_dist < min_spot_dist
            all_spots.append(spots[keep_spots])

    spots = pd.concat(all_spots, ignore_index=True)
    return spots