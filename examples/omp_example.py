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
    'tile_reg_fraction': 0.1,            # fraction of border pixels to use for stitching
    'normalization': 'phase',       # phase correlation normalization
    'max_shift': 20,     # max allowed shift value before defaulting to 0
    'ch_to_align': 0,               # channel to use for registration between rounds
    'spot_threshold': 20,           # threshold for initial spot detection
    'omp_tol': 0.1,                 # tolerance threshold for OMP algorithm
    'include_cycles': np.arange(7), # cycles to include for basecalling and OMP
    'fine_registration_tile_size': 512
}

data_dir = Path('/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/brain/slide_006/section_02/sequencing_cycles')
fnames = [
    'section_02_cycle_01.czi',
    'section_02_cycle_02.czi',
    'section_02_cycle_03_03.czi',
    'section_02_cycle_04_2trk.czi',
    'section_02_cycle_05_2trk.czi',
    'section_02_cycle_06_2trk.czi',
    'section_02_cycle_07_2trk.czi',
    'section_02_cycle_08_2trk.czi'
]

hyb_fname = '/camp/home/znamenp/home/shared/projects/rabies_BARseq/BRAC6246.1b/brain/slide_006/section_02/section_02_Hyb_probes.czi'
dapi_fname = '/camp/home/znamenp/home/shared/projects/rabies_BARseq/BRAC6246.1b/brain/slide_006/section_02/DAPI_cycle_11.czi'
savepath = '/camp/home/znamenp/home/users/znamenp/tmp/20220210/'

# load stacks that contain both tracks
stacks = []
for fname in fnames:
    stacks.append(iss.io.load_image(data_dir / fname), ops)

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
stack = iss.reg.register_rounds_fine(
    stack,
    tile_size=ops['fine_registration_tile_size']
)

#%%
print(f'loading and stitching {hyb_fname}...')
stack_hyb = iss.io.load_image(hyb_fname, ops)
# hybridisation image wasn't the same size as the sequencing images so we have to crop
hyb_registered = iss.reg.register_rounds(
    [stack[:,:,0,:].squeeze(), stack_hyb[200:-3500,1000:-900,:]],
    ch_to_align=ops['ch_to_align'],
    method='custom'
)

hyb_registered = np.moveaxis(hyb_registered, 0, 2)
hyb_registered = iss.reg.register_rounds_fine(
    hyb_registered,
    tile_size=ops['fine_registration_tile_size']
)

hyb_registered = iss.image.correct_levels(
    [hyb_registered[:,:,1,:].squeeze(), ],
    hyb_registered[:,:,1,0],
    method='histogram'
)
iss.io.write_stack(
    hyb_registered[0],
    f'{savepath}stack_hyb.tif'
)

print(f'loading and stitching {dapi_fname}...')
stack_dapi = iss.io.load_image(dapi_fname, ops)

dapi_registered = iss.reg.register_rounds(
    [stack[:,:,0,0][:,:,np.newaxis], stack_dapi[1450:-400,3050:-600,:]],
    ch_to_align=ops['ch_to_align'],
    method='custom'
)
dapi_registered = np.moveaxis(dapi_registered, 0, 2)
dapi_registered = iss.reg.register_rounds_fine(dapi_registered, tile_size=512)
iss.io.write_stack(
    dapi_registered[:,:,1,:],
    f'{savepath}stack_dapi.tif'
)

#%%
# Run OMP on the substack
spots = iss.segment.detect_spots(
    stack[:, :, 0, :].squeeze(),
    method='trackpy',
    threshold=ops['spot_threshold']
)
rois = iss.call.extract_spots(spots, stack[:, :, ops['include_cycles'], :])
# temporarily needed to avoid ROIs that have 0s for all channels in some rounds
codebook = pd.read_csv(
    '../iss_preprocess/call/codebook_YS220.csv',
    header=0,
    names=['gii', 'seq', 'gene']
)
gene_dict, gene_names = iss.call.make_gene_templates(rois, codebook)
# trimming stack to avoid pixels that are always zero
# OMP functions should check for this in the future
g, b, r = iss.call.run_omp(
    stack[:, :, ops['include_cycles'], :],
    gene_dict,
    tol=ops['omp_tol']
)

#%%
# Visualize the gene dictionary
plt.figure(figsize=(20,40))
for igene, gene in enumerate(gene_names):
    plt.subplot(10,5,igene+1)
    plt.imshow(np.reshape(gene_dict[:,igene], (7, 4)), cmap='gray')
    plt.title(gene)
    plt.colorbar()
    plt.xticks(np.arange(4), iss.call.BASES)

#%%
# Detect and plot genes
h = []
# filter pixels based on mean image values
s = np.mean(np.mean(stack, axis=3), axis=2)
g_filt = g.copy()
g_filt[s<250] = 0
rolony_locations = []
for gene_idx, gene in enumerate(gene_names):
    gene_spots = iss.segment.detect_gene_spots(g_filt[:,:,gene_idx])
    rolony_locations.append(gene_spots)

plt.figure(figsize=(20,40))
h = []
colors = cycle([ 'deepskyblue', 'aquamarine', 'orangered', 'violet', 'forestgreen', 'darkorange'])
markers = cycle('ov^<>spPXD*')

for gene_idx, gene in enumerate(gene_names):
    ax = plt.subplot(10,5,gene_idx+1)
    plt.plot(
        rolony_locations[gene_idx]['x'],
        rolony_locations[gene_idx]['y'],
        next(markers),
        c=next(colors),
        markersize=2
    )
    plt.title(gene)
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()

plt.figure(figsize=(30,30))
ax = plt.subplot(1,1,1)
h = []
for gene_idx, gene in enumerate(gene_names):
    h.append(
        plt.plot(
            rolony_locations[gene_idx]['x'],
            rolony_locations[gene_idx]['y'],
            next(markers),
            c=next(colors),
            markersize=4
        )
    )
ax.set_aspect('equal', 'box')
ax.invert_yaxis()
plt.legend(gene_names, loc='right', ncol=2, bbox_to_anchor=(0.,0.,1.15,1.))

#%%
hyb_spots = iss.segment.detect_spots(
    hyb_registered[0][:,:,:3].squeeze(),
    method='trackpy',
    threshold=80
)
spot_channels = iss.call.call_hyb_spots(hyb_spots, hyb_registered[0][:,:,:3], nprobes=3, vis=True)
fname = f'{savepath}genes.npz'
with np.load(fname, allow_pickle=True) as data:
    rolony_locations = data['all_genes'].tolist()
    gene_names = data['unique_genes'].tolist()

hyb_genes = [ 'Gad1', 'Slc17a7', 'Slc30a3' ]
for igene, gene in enumerate(hyb_genes):
    rolony_locations.append(hyb_spots[spot_channels == igene])
    gene_names.append(gene)

#%%
# cellpose segmentation
masks = iss.segment.cellpose_segmentation(
    '/camp/home/znamenp/home/users/znamenp/tmp/stack_dapi.tif',
    channels=None,
    flow_threshold=2,
    vis=False,
    dilate_pix=0,
    rescale=0.55,
    model_type='nuclei'
)
np.save(f'{savepath}masks.npy', masks)

#%%
gene_df = iss.segment.count_rolonies(masks, rolony_locations, gene_names)
gene_df.to_pickle(f'{savepath}genes.pkl')
