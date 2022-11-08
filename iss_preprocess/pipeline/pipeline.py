import numpy as np
from ..image import correct_offset, fstack_channels
from ..reg import register_tiles, estimate_correction, apply_corrections, estimate_rotation_translation
from ..io import get_tiles, get_tile_ome, reorder_channels, write_stack


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


def align_channels_and_rounds(stack):
    nchannels, nrounds = stack.shape[2:]
    angles_channels, shifts_channels = align_within_channels(stack)
    std_stack, mean_stack = get_channel_reference_images(stack, angles_channels, shifts_channels)
    scales, angles, shifts = estimate_correction(std_stack, ch_to_align=0)
    reg_stack = np.zeros((stack.shape))

    for ich in range(nchannels):
        reg_stack[:,:,ich,:] = apply_corrections(stack[:,:,ich,:], np.ones((nrounds)), angles_channels[ich], shifts_channels[ich])
    for iround in range(nrounds):
        reg_stack[:,:,:,iround] = apply_corrections(reg_stack[:,:,:,iround], scales, angles, shifts)

    return reg_stack


def project_tile(fnames):
    for fname in fnames:
        print(f'loading {fname}')
    im = get_tile_ome(fname + '.ome.tif', fname + '_metadata.txt')
    print('computing projection')
    im_proj = fstack_channels(im, sth=10)
    #im_proj = np.max(im, axis=3)
    write_stack(im_proj, fname + '_proj.tif', bigtiff=True)


def align_within_channels(stack):
    # align rounds to each other for each channel
    nchannels, nrounds = stack.shape[2:]
    ref_round = 0
    angles_channels = []
    shifts_channels = []
    for ref_ch in range(nchannels):
        angles = []
        shifts = []
        for iround in range(nrounds):
            if ref_round != iround:
                angle, shift = estimate_rotation_translation(
                    stack[:,:,ref_ch,ref_round], stack[:,:,ref_ch,iround],
                    angle_range=1.,
                    niter=3,
                    nangles=15,
                    scale=1.,
                    min_shift=2
                )
            else:
                angle, shift = 0., [0., 0.]
            angles.append(angle)
            shifts.append(shift)
            print(f'angle: {angle}, shift: {shift}')
        angles_channels.append(angles)
        shifts_channels.append(shifts)
    return angles_channels, shifts_channels


def get_channel_reference_images(stack, angles_channels, shifts_channels):
    nchannels, nrounds = stack.shape[2:]

    # get a good reference image for each channel
    std_stack = np.zeros((stack.shape[:3]))
    mean_stack = np.zeros((stack.shape[:3]))

    for ich in range(nchannels):
        std_stack[:,:,ich] = np.std(
            apply_corrections(stack[:,:,ich,:], np.ones((nrounds)), angles_channels[ich], shifts_channels[ich]),
            axis=2
        )
        mean_stack[:,:,ich] = np.mean(
            apply_corrections(stack[:,:,ich,:], np.ones((nrounds)), angles_channels[ich], shifts_channels[ich]),
            axis=2
        )
    return std_stack, mean_stack