import iss_preprocess as iss
import numpy as np
import skimage

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

# load stacks that contain both tracks
stacks = []
for i, fname in enumerate(fnames):
    tiles, metadata = iss.io.get_tiles(fname)
    tiles = iss.image.correct_offset(tiles, method='metadata', metadata=metadata)
    im, tile_pos = iss.reg.register_tiles(tiles, reg_pix=tiles.iloc[0].data.shape[0]*0.1)
    stacks.append(np.max(im, axis=3))
# load stacks with individual tracks
tracks = []
for fnames_track in fnames_tracks:
    track = []
    for fname in fnames_track:
        tiles, metadata = iss.io.get_tiles(fname)
        tiles = iss.image.correct_offset(tiles, method='metadata', metadata=metadata)
        im, _ = iss.reg.register_tiles(tiles, reg_pix=tiles.iloc[0].data.shape[0]*0.1)
        track.append(np.max(im, axis=3))
    tracks.append(track)
# register tracks to each other
tracks = iss.reg.register_tracks(tracks[0], tracks[1], chs_to_align=(0,0))

stacks.extend(tracks)
# match histograms using the first cycle as template
corrected_stacks = []
for stack in stacks:
    corrected_stack = np.empty(stack.shape)
    nchannels = stack.shape[2]
    for channel in range(nchannels):
        corrected_stack[:,:,channel] = skimage.exposure.match_histograms(stack[:,:,channel], stacks[0][:,:,0])
    corrected_stacks.append(corrected_stack)
# register sequencing rounds
corrected_stacks = iss.reg.register_rounds(corrected_stacks, ch_to_align=0)
# save output
for i in range(corrected_stacks.shape[0]):
    iss.io.write_stack(corrected_stacks[i,:,:,:].squeeze(), f'/camp/home/znamenp/home/users/znamenp/tmp/stack{i}.tif')
