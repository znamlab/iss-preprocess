# Initial registration
#%%
import pandas as pd
import iss_preprocess as iss
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from skimage.io import ImageCollection
from itertools import compress, cycle

#%%
ops = {
    'tile_reg_pix': 200,            # number of border pixels to use for stitching
    'ch_to_align': 0,               # channel to use for registration between rounds
    'spot_threshold': 20,           # threshold for initial spot detection
    'omp_tol': 0.1,                 # tolerance threshold for OMP algorithm
    'include_cycles': np.arange(7)  # cycles to include for basecalling and OMP
}

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
    im, tile_pos = iss.reg.register_tiles(tiles, reg_pix=ops['tile_reg_pix'])
    im = iss.io.reorder_channels(im, metadata)
    stacks.append(np.mean(im, axis=3))

# register sequencing rounds
corrected_stacks = iss.image.correct_levels(stacks, stacks[0][:,:,0], method='histogram')
corrected_stacks = iss.reg.register_rounds(corrected_stacks, ch_to_align=ops['ch_to_align'])
# save output
for i in range(corrected_stacks.shape[0]):
    iss.io.write_stack(
        corrected_stacks[i,:,:,:].squeeze(),
        f'/camp/home/znamenp/home/users/znamenp/tmp/stack_mean_{i}.tif'
    )

#%%
# Load preregistered stack
coll = ImageCollection('/camp/home/znamenp/home/users/znamenp/tmp/stack_mean_*.tif')
stack = coll.concatenate()
stack = np.moveaxis(stack, 0, 2)
stack = stack.reshape([stack.shape[0], stack.shape[1], -1, 4])
# do fine local registration
stack = iss.reg.register_rounds_fine(stack)

#%%
# Run OMP on the substack
spots = iss.segment.detect_spots(
    stack[:, :, 0, :].squeeze(),
    method='trackpy',
    threshold=ops['spot_threshold']
)
rois = iss.call.extract_spots(spots, stack[:, :, ops['include_cycles'], :])
# temporarily needed to avoid ROIs that have 0s for all channels in some rounds
valid_rois = np.mean(iss.call.rois_to_array(rois).reshape((-1, len(rois))), axis=0) > 0
codebook = pd.read_csv(
    '../iss_preprocess/call/codebook_YS220.csv',
    header=0,
    names=['gii', 'seq', 'gene']
)
gene_dict, unique_genes = iss.call.make_gene_templates(list(compress(rois, valid_rois)), codebook)
# trimming stack to avoid pixels that are always zero
# OMP functions should check for this in the future
g, b, r = iss.call.run_omp(
    stack[:7000, :6900, ops['include_cycles'], :],
    gene_dict,
    tol=ops['omp_tol']
)

#%%
# Detect and plot genes
markers = cycle('ov^<>spPXD*')
plt.figure(figsize=(15,15))
ax = plt.subplot(1,1,1)
h = []
for gene_idx, gene in enumerate(unique_genes):
    gene_spots = iss.segment.detect_gene_spots(g[:,:,gene_idx])
    h.append(
        plt.plot(gene_spots['x'], gene_spots['y'], next(markers))
    )
ax.set_aspect('equal', 'box')
ax.invert_yaxis()
plt.legend(unique_genes, loc='right', ncol=2, bbox_to_anchor=(0.,0.,1.25,1.))