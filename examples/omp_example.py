# Initial registration
#%%
import pandas as pd

import iss_preprocess as iss
import numpy as np
from pathlib import Path
from skimage.io import ImageCollection

#%%
data_dir = Path('/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/brain/slide_006/section_02/sequencing_cycles')
fnames = [
    'section_02_cycle_01.czi',
    'section_02_cycle_02.czi',
    'section_02_cycle_03.czi',
    'section_02_cycle_04_2trk.czi',
    'section_02_cycle_05_2trk.czi',
    'section_02_cycle_06_2trk.czi',
    'section_02_cycle_07_2trk.czi',
    'section_02_cycle_08_2trk.czi'
]
# load stacks that contain both tracks
stacks = []
for i, fname in enumerate(fnames):
    tiles, metadata = iss.io.get_tiles(data_dir / fname)
    tiles = iss.image.correct_offset(tiles, method='metadata', metadata=metadata)
    im, tile_pos = iss.reg.register_tiles(tiles, reg_pix=200)
    im = iss.io.reorder_channels(im, metadata)
    stacks.append(np.mean(im, axis=3))

# register sequencing rounds
corrected_stacks = iss.image.correct_levels(stacks, stacks[0][:,:,0], method='histogram')
corrected_stacks = iss.reg.register_rounds(corrected_stacks, ch_to_align=0)
# save output
for i in range(corrected_stacks.shape[0]):
    iss.io.write_stack(corrected_stacks[i,:,:,:].squeeze(), f'/camp/home/znamenp/home/users/znamenp/tmp/stack_mean_{i}.tif')

#%%
# Load preregistered stack
coll = ImageCollection('/camp/home/znamenp/home/users/znamenp/tmp/stack_mean_*.tif')
stack = coll.concatenate()
stack = np.moveaxis(stack, 0, 2)
stack = stack.reshape([stack.shape[0], stack.shape[1], -1, 4])

stack = iss.reg.register_rounds([ stack[:1024,:1024,i,:].squeeze() for i in range(8) ], ch_to_align=0)
stack = np.moveaxis(stack, 0, 2)

#%%
# Run OMP on the substack
spots = iss.segment.detect_spots(stack[:,:,0,:].squeeze(), method='trackpy', threshold=20)
rois = iss.call.extract_spots(spots, stack[:,:,:7,:])

codebook = pd.read_csv('../iss_preprocess/call/codebook_YS220.csv', header=0, names=['gii', 'seq', 'gene'])
gene_dict, unique_genes = iss.call.make_gene_templates(rois, codebook)
g, b, r = iss.call.run_omp(stack[:,:,:7,:], gene_dict)

#%%
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from scipy.signal import medfilt2d
from itertools import cycle

markers = cycle('ov^<>spPXD*')
plt.figure(figsize=(15,15))
ax = plt.subplot(1,1,1)
h = []
for gene_idx, gene in enumerate(unique_genes):
    spots_array = blob_log(
        medfilt2d(g[:,:,gene_idx], kernel_size=3),
        max_sigma=4.,
        min_sigma=.5,
        num_sigma=10,
        log_scale=True,
        overlap=0.9,
    )
    gene_spots = pd.DataFrame(spots_array, columns=['y', 'x', 'size'])
    gene_spots = gene_spots[gene_spots['size']>=1.2]
    h.append(
        plt.plot(gene_spots['x'], gene_spots['y'], next(markers))
    )
ax.set_aspect('equal', 'box')
ax.invert_yaxis()
plt.legend(unique_genes, loc='right', ncol=2, bbox_to_anchor=(0.,0.,1.25,1.))