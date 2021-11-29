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
    sr.register_stack(stacks[:,:,:,ch_to_align].squeeze(), referen='first')

    for channel in range(nchannels):
        stacks[:,:,:,channel] = sr.transform_stack(stacks[:,:,:,channel].squeeze())

    return stacks
