import numpy as np
from pystackreg import StackReg

from skimage.registration import phase_cross_correlation
from skimage.filters import window

def register_rounds_fine(stack, tile_size=1024, ch_to_align=0, padding=100, max_shift=20):
    """
    Do fine registration across rounds by breaking up the stack into smaller substacks.

    Args:
        stack (numpy.ndarray): X x Y x R x C image stack.
        tile_size (int): size of substacks for local registration.
        ch_to_align (int): channel to use for phase correlation.
        padding (int): used to pad the stack to avoid exceeding array bounds.
        max_shift: maximum shift permitted for substacks. If exceeded, shift is set to 0.
            Mostly useful for blank tiles.

    Returns:
         numpy.ndarray: X x Y x R x C image stack after registration.

    """
    xtiles = np.ceil(stack.shape[0] / tile_size).astype(int)
    ytiles = np.ceil(stack.shape[1] / tile_size).astype(int)

    padded_stack = np.pad(stack, [[padding,padding],[padding,padding],[0,0],[0,0]])
    registered_stack = np.empty(stack.shape)
    for ix in range(xtiles):
        for iy in range(ytiles):
            xstart = np.max([ix * tile_size, 0])
            xend = np.min([(ix + 1) * tile_size, stack.shape[0]])
            ystart = np.max([iy * tile_size, 0])
            yend = np.min([(iy + 1) * tile_size, stack.shape[1]])
            reference_tile = stack[xstart:xend, ystart:yend, 0, ch_to_align]
            for ir in range(stack.shape[2]):
                s = phase_cross_correlation(
                    reference_tile,
                    stack[xstart:xend, ystart:yend, ir, ch_to_align]
                )[0]
                if abs(s[0])>max_shift:
                    s[0] = 0
                if abs(s[1])>max_shift:
                    s[1] = 0
                xshift = int(padding - s[0])
                yshift = int(padding - s[1])
                registered_stack[xstart:xend, ystart:yend, ir, :] = \
                    padded_stack[xstart+xshift:xend+xshift, ystart+yshift:yend+yshift, ir, :]
    return registered_stack


def register_rounds(stacks, ch_to_align=0, threshold=None, filter_window=None):
    """
    Register sequencing rounds.

    Args:
        stacks (list): list of X x Y x C ndarrays with individual rounds.
        ch_to_align(int): channel to use for registration.
        filter_window (str): whether to window the input images before registration.
            Example windows are 'cosine', 'blackman', and 'flattop'.
    """
    maxx = 0
    maxy = 0
    for stack in stacks:
        maxx = np.max((maxx, stack.shape[0]))
        maxy = np.max((maxy, stack.shape[1]))
    nchannels = stacks[0].shape[2]
    nrounds = len(stacks)
    for i, stack in enumerate(stacks):
        padx = maxx - stack.shape[0]
        pady = maxy - stack.shape[1]
        stacks[i] = np.pad(stack, ((0, padx), (0, pady), (0,0)), 'constant')

    stacks = np.stack(stacks, axis=0)
    if threshold:
        stacks[stacks<threshold] = 0

    sr = StackReg(StackReg.RIGID_BODY)
    if filter_window:
        w = window(filter_window, stacks.shape[1:3])[np.newaxis, :, :]
        sr.register_stack(stacks[:,:,:,ch_to_align].squeeze() * w, reference='previous')
    else:
        sr.register_stack(stacks[:,:,:,ch_to_align].squeeze(), reference='previous')

    for channel in range(nchannels):
        stacks[:,:,:,channel] = sr.transform_stack(stacks[:,:,:,channel].squeeze())

    return stacks


def register_tracks(track1, track2, chs_to_align=(0,0), threshold=None, filter_window=None):
    """
    Register imaging tracks.

    Args:
        track1 (list): list of X x Y x C ndarrays for track 1 on individual rounds.
        track2 (list): list of X x Y x C ndarrays for track 2 on individual rounds.
        chs_to_align (tuple): channels to use for registration.
        filter_window (str): whether to window the input images before registration.
            Example windows are 'cosine', 'blackman', and 'flattop'.

    """
    out = []

    for stacks in zip(track1, track2):
        maxx = 0
        maxy = 0
        for stack in stacks:
            maxx = np.max((maxx, stack.shape[0]))
            maxy = np.max((maxy, stack.shape[1]))
        padded_stacks = []

        for stack in stacks:
            padx = maxx - stack.shape[0]
            pady = maxy - stack.shape[1]
            padded_stacks.append(np.pad(stack, ((0, padx), (0, pady), (0,0)), 'constant'))

        sr = StackReg(StackReg.TRANSLATION)
        if threshold:
            padded_stacks[padded_stacks<threshold] = 0
        if filter_window:
            w = window(filter_window, stacks.shape[:2])
            sr.register(
                padded_stacks[0][:,:,chs_to_align[0]].squeeze() * w,
                padded_stacks[1][:,:,chs_to_align[1]].squeeze() * w
            )
        else:
            sr.register(
                padded_stacks[0][:,:,chs_to_align[0]].squeeze(),
                padded_stacks[1][:,:,chs_to_align[1]].squeeze()
            )

        nchannels = padded_stacks[1].shape[2]
        for channel in range(nchannels):
            padded_stacks[1][:,:,channel] = sr.transform(
                padded_stacks[1][:,:,channel]
            )
        # stack along the channels axis
        stacks = np.concatenate((padded_stacks[0], padded_stacks[1]), axis=2)
        out.append(stacks)

    return out
