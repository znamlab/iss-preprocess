import numpy as np
from pystackreg import StackReg

def register_rounds(stacks, ch_to_align=0):
    """
    Register sequencing rounds.

    Args:
        stacks (list): list of X x Y x C ndarrays with individual rounds.
        ch_to_align(int): channel to use for registration.

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

    sr = StackReg(StackReg.RIGID_BODY)
    sr.register_stack(stacks[:,:,:,ch_to_align].squeeze(), reference='first')

    for channel in range(nchannels):
        stacks[:,:,:,channel] = sr.transform_stack(stacks[:,:,:,channel].squeeze())

    return stacks


def register_tracks(track1, track2, chs_to_align=(0,0)):
    """
    Register imaging tracks.

    Args:
        track1 (list): list of X x Y x C ndarrays for track 1 on individual rounds.
        track2 (list): list of X x Y x C ndarrays for track 2 on individual rounds.
        chs_to_align (tuple): channels to use for registration.

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
