import numpy as np
import scipy.fft
import scipy.ndimage
from pystackreg import StackReg
from numba import jit, prange
from skimage.filters import window, difference_of_gaussians
from skimage.transform import rotate, rescale
from . import phase_corr


def apply_corrections(im, scales, angles, shifts, ch_to_align=0):
    nchannels = im.shape[2]
    im_reg = np.zeros(im.shape)
    for channel, scale, angle, shift in zip(range(nchannels), scales, angles, shifts):
        target_rescaled = rotate(rescale(im[:,:,channel], scale, preserve_range=True), angle)
        target_rescaled = pad_to_size(target_rescaled, im.shape[0:2])
        #shift, _ = phase_corr(im[:,:,ch_to_align], target_rescaled)
        im_reg[:,:,channel] = scipy.ndimage.shift(target_rescaled, shift)

    return im_reg


def estimate_correction(im, ch_to_align=0):
    """

    Args:
        im:
        ch_to_align:

    Returns:

    """
    nchannels = im.shape[2]
    scale_range = 0.05
    scales, angles, shifts = [], [], []
    for channel in range(nchannels):
        if channel != ch_to_align:
            scale, angle, shift = estimate_scale_rotation_translation(
                im[:,:,ch_to_align],
                im[:,:,channel],
                niter=5,
                nangles=3,
                verbose=True,
                scale_range=scale_range,
                angle_range=1.,
            )
        else:
            scale, angle, shift = 1.0, 0., (0, 0)
        scales.append(scale)
        angles.append(angle)
        shifts.append(shift)
    return scales, angles, shifts


def pad_to_size(array, size):
    if size[0] > array.shape[0]:
        pad_x = size[0] - array.shape[0]
        array = np.pad(array,
                       ((int(pad_x / 2), pad_x - int(pad_x / 2)),
                        (0, 0))
                       )
    if size[0] < array.shape[0]:
        trim_x = array.shape[0] - size[0]
        array = array[int(trim_x / 2): -(trim_x - int(trim_x / 2)), :]
    if size[1] > array.shape[1]:
        pad_y = size[1] - array.shape[1]
        array = np.pad(array,
                       ((0, 0),
                        (int(pad_y / 2), pad_y - int(pad_y / 2)))
                       )
    if size[1] < array.shape[1]:
        trim_y = array.shape[1] - size[1]
        array = array[:, int(trim_y / 2): -(trim_y - int(trim_y / 2))]
    return array


def estimate_scale_rotation_translation(reference, target, angle_range=5., scale_range=0.01, nscales=15,
                                        niter=3, nangles=15, verbose=False, min_shift=None):
    """
    Estimate rotation and translation that maximizes phase correlation between the target and the
    reference image. Search for the best angle is performed iteratively by decreasing the gradually
    searching over small and smaller angle range.

    Args:
        reference (numpy.ndarray): X x Y reference image
        target (numpy.ndarray): X x Y target image
        angle_range (float): initial range of angles in degrees to search over
        niter (int): number of iterations to refine rotation angle
        nangles (int): number of angles to try on each iteration
        verbose (bool): whether to print progress of registration
        scale (float): how to rescale image for finding the optimal rotation

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts

    """
    best_angle = 0
    best_scale = 1
    reference_fft = scipy.fft.fft2(reference)

    for i in range(niter):
        scales = np.linspace(-scale_range, scale_range, nscales) + best_scale
        max_cc = np.empty(scales.shape)
        best_angles = np.empty(scales.shape)
        for iscale in range(nscales):
            target_rescaled = rescale(target, scales[iscale])
            target_rescaled = pad_to_size(target_rescaled, reference_fft.shape)
            best_angles[iscale], max_cc[iscale] = estimate_rotation_angle(reference_fft, target_rescaled, angle_range,
                                                                          best_angle, nangles, min_shift=None)
        best_scale_index = np.argmax(max_cc)
        best_scale = scales[best_scale_index]
        best_angle = best_angles[best_scale_index]
        angle_range = angle_range / 5
        scale_range = scale_range / 5
        if verbose:
            print(f'Best scale: {best_scale}. Best angle: {best_angle}')
    target_rescaled = rescale(target, best_scale)
    target_rescaled = pad_to_size(target_rescaled, reference_fft.shape)
    shift, _ = phase_corr(reference_fft, rotate(target_rescaled, best_angle), fft_ref=False)
    return best_scale, best_angle, shift


@jit(parallel=True, forceobj=True)
def estimate_rotation_angle(reference_fft, target, angle_range, best_angle, nangles, min_shift=None):
    angles = np.linspace(-angle_range, angle_range, nangles) + best_angle
    max_cc = np.empty(angles.shape)
    shifts = np.empty((nangles, 2))
    for iangle in prange(nangles):
        shifts[iangle, :], cc = phase_corr(
            reference_fft,
            rotate(target, angles[iangle]),
            fft_ref=False,
            min_shift=min_shift
        )
        max_cc[iangle] = np.max(cc)
    best_angle_index = np.argmax(max_cc)
    best_angle = angles[best_angle_index]
    return best_angle, max_cc[best_angle_index]


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

    padded_stack = np.pad(stack, [[padding, padding], [padding, padding], [0, 0], [0, 0]])
    registered_stack = np.empty(stack.shape)
    for ix in range(xtiles):
        for iy in range(ytiles):
            xstart = np.max([ix * tile_size, 0])
            xend = np.min([(ix + 1) * tile_size, stack.shape[0]])
            ystart = np.max([iy * tile_size, 0])
            yend = np.min([(iy + 1) * tile_size, stack.shape[1]])
            if ch_to_align >= 0:
                reference_tile = stack[xstart:xend, ystart:yend, 0, ch_to_align]
            else:
                reference_tile = np.sum(stack[xstart:xend, ystart:yend, 0, :], axis=2)
            for ir in range(stack.shape[2]):
                if ch_to_align >= 0:
                    s, _ = phase_corr(
                        reference_tile,
                        stack[xstart:xend, ystart:yend, ir, ch_to_align],
                        max_shift=max_shift
                    )
                else:
                    s, _ = phase_corr(
                        reference_tile,
                        np.sum(stack[xstart:xend, ystart:yend, ir, :], axis=2),
                        max_shift=max_shift
                    )
                xshift = int(padding - s[0])
                yshift = int(padding - s[1])
                registered_stack[xstart:xend, ystart:yend, ir, :] = \
                    padded_stack[xstart + xshift:xend + xshift, ystart + yshift:yend + yshift, ir, :]
    return registered_stack


@jit(parallel=True, forceobj=True)
def estimate_rotation_translation(reference, target, angle_range=5., niter=3, nangles=20, scale=1., min_shift=None):
    """
    Estimate rotation and translation that maximizes phase correlation between the target and the
    reference image. Search for the best angle is performed iteratively by decreasing the gradually
    searching over small and smaller angle range.

    Args:
        reference (numpy.ndarray): X x Y reference image
        target (numpy.ndarray): X x Y target image
        angle_range (float): initial range of angles in degrees to search over
        niter (int): number of iterations to refine rotation angle
        nangles (int): number of angles to try on each iteration
        verbose (bool): whether to print progress of registration
        scale (float): how to rescale image for finding the optimal rotation

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts

    """
    best_angle = 0
    if scale != 1:
        target_rescaled = rescale(target, scale)
        reference_fft = scipy.fft.fft2(rescale(reference, scale))
    else:
        target_rescaled = target
        reference_fft = scipy.fft.fft2(reference)
    for i in range(niter):
        best_angle, max_cc = estimate_rotation_angle(
            reference_fft, target_rescaled, angle_range, best_angle, nangles, min_shift=min_shift
        )
        angle_range = angle_range / 5
    shift, _ = phase_corr(reference, rotate(target, best_angle))
    return best_angle, shift


def register_rounds(stacks, ch_to_align=0, filter_window=None, dog_filter=None, method='pystackreg'):
    """
    Register sequencing rounds.

    Args:
        stacks (list): list of X x Y x C ndarrays with individual rounds.
        ch_to_align(int): channel to use for registration.
        filter_window (str): whether to window the input images before registration.
            Example windows are 'cosine', 'blackman', and 'flattop'.
        dog_filter (tuple): whether to filter images with a difference-of-gaussians
            filter before registering. A tuple of `low_sigma` and `high_sigma`
            defining the range of spatial scales to bandpass.
        method (str): 'pystackreg' to use StackReg registration, or custom method.

    """
    maxx = 0
    maxy = 0
    for stack in stacks:
        maxx = np.max((maxx, stack.shape[0]))
        maxy = np.max((maxy, stack.shape[1]))
    nchannels = stacks[0].shape[2]
    for i, stack in enumerate(stacks):
        padx = maxx - stack.shape[0]
        pady = maxy - stack.shape[1]
        stacks[i] = np.pad(
            stack,
            ((int(padx / 2), padx - int(padx / 2)), (int(pady / 2), pady - int(pady / 2)), (0, 0)),
            'constant'
        )

    stacks = np.stack(stacks, axis=0)

    stack_for_registration = stacks[:, :, :, ch_to_align].squeeze()
    if filter_window:
        w = window(filter_window, stacks.shape[1:3])[np.newaxis, :, :]
        stack_for_registration = stack_for_registration * w
    if dog_filter:
        stack_for_registration = difference_of_gaussians(
            stack_for_registration,
            low_sigma=dog_filter[0],
            high_sigma=dog_filter[1],
            channel_axis=0
        )
    if method == 'pystackreg':
        sr = StackReg(StackReg.RIGID_BODY)
        sr.register_stack(stack_for_registration, reference='previous')

        for channel in range(nchannels):
            stacks[:, :, :, channel] = sr.transform_stack(stacks[:, :, :, channel].squeeze())
    else:
        for iround in range(stacks.shape[0]):
            if iround > 0:
                angle, shift = estimate_rotation_translation(
                    stack_for_registration[0, :, :],
                    stack_for_registration[iround, :, :],
                    scale=0.33
                )
                for channel in range(nchannels):
                    stacks[iround, :, :, channel] = scipy.ndimage.shift(rotate(stacks[iround, :, :, channel], angle),
                                                                        shift)

    return stacks


def register_tracks(track1, track2, chs_to_align=(0, 0), threshold=None, filter_window=None):
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
            padded_stacks.append(np.pad(stack, ((0, padx), (0, pady), (0, 0)), 'constant'))

        sr = StackReg(StackReg.TRANSLATION)
        if threshold:
            padded_stacks[padded_stacks < threshold] = 0
        if filter_window:
            w = window(filter_window, stacks.shape[:2])
            sr.register(
                padded_stacks[0][:, :, chs_to_align[0]].squeeze() * w,
                padded_stacks[1][:, :, chs_to_align[1]].squeeze() * w
            )
        else:
            sr.register(
                padded_stacks[0][:, :, chs_to_align[0]].squeeze(),
                padded_stacks[1][:, :, chs_to_align[1]].squeeze()
            )

        nchannels = padded_stacks[1].shape[2]
        for channel in range(nchannels):
            padded_stacks[1][:, :, channel] = sr.transform(
                padded_stacks[1][:, :, channel]
            )
        # stack along the channels axis
        stacks = np.concatenate((padded_stacks[0], padded_stacks[1]), axis=2)
        out.append(stacks)

    return out
