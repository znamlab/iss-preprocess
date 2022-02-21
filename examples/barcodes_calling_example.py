# Initial registration
import iss_preprocess as iss
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from itertools import cycle, compress

ops = {
    'tile_reg_fraction': 0.1,  # fraction of border pixels to use for stitching
    'normalization': 'phase',  # phase correlation normalization
    'max_shift': 20,  # max allowed shift value before defaulting to 0
    'ch_to_align': -1,  # channel to use for registration between rounds
    'spot_threshold': 80,  # threshold for initial spot detection
    'omp_tol': 0.05,  # tolerance threshold for OMP algorithm
    'include_cycles': np.arange(7)  # cycles to include for basecalling and OMP
}

def load_image(fname):
    tiles, metadata = iss.io.get_tiles(fname)
    tiles = iss.image.correct_offset(tiles, method='metadata', metadata=metadata)
    im, tile_pos = iss.reg.register_tiles(
        tiles,
        reg_fraction=ops['tile_reg_fraction'],
        method='custom',
        max_shift=ops['max_shift'],
        ch_to_align=-ops['ch_to_align']
    )
    im = iss.io.reorder_channels(im, metadata)
    if im.ndim>3:
        im = np.mean(im, axis=3)
    return im

# files from the pilot run
fnames = [
    '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round1_section1.czi',
    '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round2_section1.czi',
    '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round3_section1.czi',
    '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round4_section1.czi']

fnames_tracks = [
    ['/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round5_section1_trk1.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round6_section1_trk1.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round7_section1_trk1.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round8_section1_trk1.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round9_section1_trk1.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round10_section1_trk1.czi'],
    ['/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round5_section1_trk2.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round6_section1_trk2.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round7_section1_trk2.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round8_section1_trk2.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round9_section1_trk2.czi',
     '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round10_section1_trk2.czi']
]

savepath = '/camp/home/znamenp/home/users/znamenp/tmp/slide3/'

#%%
# load stacks that contain both tracks
stacks = []
for i, fname in enumerate(fnames):
    print(f'loading and stitching {fname}...')
    stacks.append(load_image(fname))
# load stacks with individual tracks
tracks = []
for fnames_track in fnames_tracks:
    track = []
    for fname in fnames_track:
        print(f'loading and stitching {fname}...')
        track.append(load_image(fname))
    tracks.append(track)
# register tracks to each other
tracks = iss.reg.register_tracks(tracks[0], tracks[1], chs_to_align=(0,0))

stacks.extend(tracks)
corrected_stacks = iss.image.correct_levels(stacks, stacks[0][:, :, 0], method='histogram')
registered_stacks = iss.reg.register_rounds(
    corrected_stacks,
    ch_to_align=ops['ch_to_align'],
    method='custom'
)

registered_stacks = np.moveaxis(registered_stacks, 0, 2)
# save output
for i in range(registered_stacks.shape[2]):
    iss.io.write_stack(
        registered_stacks[:, :, i, :].squeeze(),
        f'{savepath}stack_mean_{i}.tif'
    )

#%%
from skimage.io import ImageCollection

coll = ImageCollection(f'{savepath}stack_mean_*.tif')
registered_stacks = coll.concatenate()
registered_stacks = np.moveaxis(registered_stacks, 0, 2)
registered_stacks = registered_stacks.reshape([registered_stacks.shape[0], registered_stacks.shape[1], -1, 4])

substack = np.reshape(registered_stacks, (registered_stacks.shape[0], registered_stacks.shape[1], -1))
substack = np.moveaxis(substack, 2, 0)
# run only small area of stack with the majority of labelled neurons
substack = substack[:, 1700:3250, 700:2250]

cmap = iss.segment.correlation_map(substack)
# set pixels outside the brain to 0
intensity_threshold = 100
cmap[substack.mean(axis=0) < intensity_threshold] = 0

rois = iss.segment.detect_rois(
    substack,
    cmap.copy(),
    min_size=4,
    max_size=500,
    threshold=0.5,
    nsteps=500
)
np.savez(f'{savepath}rois.npz', rois=rois, cmap=cmap)

#%%
# keep only the first 300 rois
rois = rois[:300]
keep_rois = iss.segment.find_overlappers(rois, max_overlap=0.5)
substack = registered_stacks[1700:3250, 700:2250,:,:]
# extract fluorescence
for roi in rois:
    roi.trace = substack[roi.xpix,roi.ypix,:,:].mean(axis=0)

rois = list(compress(rois, keep_rois))
bases, base_means, x = iss.call.basecall_rois(rois, separate_rounds=False)
# plot basecalling results
plt.figure(figsize=(20, 20))
x = np.moveaxis(x, 2, 0)
x = np.reshape(x, (-1, 4))
for xch in range(x.shape[1]):
    for ych in range(x.shape[1]):
        plt.subplot(x.shape[1], x.shape[1], xch * x.shape[1] + ych + 1)
        plt.scatter(x[:, xch], x[:, ych], c=bases, s=5)
# save sequences
seqs = []
for seq in bases:
    seqs.append(''.join(iss.call.BASES[seq]))
np.savetxt(f'{savepath}barcodes.csv', np.array(seqs), fmt='%s')

#%%
# plot some barcodes
plt.rcParams['pdf.fonttype'] = 'truetype'

colors = [[0,1,0], [1,0,1], [0,1,1], [1,0,0]]
im = iss.vis.to_rgb(registered_stacks[1700:3250, 700:2250,0,:], colors, vmax=7000, vmin=np.array([1300,]))
colors = cycle(['deepskyblue', 'aquamarine', 'orangered', 'violet', 'forestgreen', 'darkorange'])
plt.figure(figsize=(3,3))
ax = plt.subplot(1,1,1)
plt.imshow(np.swapaxes(im, 0, 1))

sequences = []
h = []
(unique, counts) = np.unique(bases, return_counts=True, axis=0)
# plot sequences present at least 3 times
for seq in unique[counts > 2,:]:
    masks = np.zeros(rois[0].shape)
    this_seq = list(compress(rois, np.all(np.equal(bases, seq), axis=1)))
    sequences.append(''.join(iss.call.BASES[seq]))
    this_color = next(colors)
    for roi in this_seq:
        coor = center_of_mass(roi.mask)
        h_, = plt.plot(coor[0], coor[1], 'o', markersize=5, color=this_color, linewidth=1, fillstyle='none',
                       label=''.join(iss.call.BASES[seq]))
    h.append(h_)

ax.invert_yaxis()
ax.legend(handles=h, loc='right', bbox_to_anchor=(0., 0., 1.7, 1.))
# 0.83 microns per pixel
plt.plot([1200, 1300+100/0.83], [100,100], color='white', linewidth=8)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig(f'{savepath}barcodes.pdf', dpi=250)
