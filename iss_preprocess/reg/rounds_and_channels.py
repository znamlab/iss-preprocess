import numpy as np
import scipy.fft
import scipy.ndimage
from numba import jit
from skimage.transform import SimilarityTransform, warp
from skimage.registration import phase_cross_correlation
from . import phase_corr, make_transform, transform_image


def register_channels_and_rounds(stack, ref_ch=0, ref_round=0, max_shift=None):
    """
    Estimate transformation matrices for alignment across channels and rounds.

    Args:
        stack: X x Y x Nchannels x Nrounds images stack
        ref_ch (int): channel to align to
        ref_round (int): round to align to

    Returns:
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        scales_between_channels (np.array): Nchannels array of scales
        angles_between_channels (np.array): Nchannels array of angles
        shifts_between_channels (np.array): Nchannels x 2 array of shifts

    """
    # first register images across rounds within each channel
    angles_within_channels, shifts_within_channels = align_within_channels(
        stack, upsample=False, ref_round=ref_round, max_shift=max_shift
    )
    # use these to computer a reference image for each channel
    std_stack, mean_stack = get_channel_reference_images(
        stack, angles_within_channels, shifts_within_channels
    )
    (
        scales_between_channels,
        angles_between_channels,
        shifts_between_channels,
    ) = estimate_correction(std_stack, ch_to_align=ref_ch, upsample=5)

    return (
        angles_within_channels,
        shifts_within_channels,
        scales_between_channels,
        angles_between_channels,
        shifts_between_channels,
    )


def generate_channel_round_transforms(
    angles_within_channels,
    shifts_within_channels,
    scales_between_channels,
    angles_between_channels,
    shifts_between_channels,
    stack_shape,
    align_channels=True,
    ref_ch=0,
):
    """Generate transformation matrices for each channel and round.

    Args:
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        scales_between_channels (np.array): Nchannels array of scales
        angles_between_channels (np.array): Nchannels array of angles
        shifts_between_channels (np.array): Nchannels x 2 array of shifts
        stack_shape (tuple): shape of the stack
        align_channels (bool): whether to register channels to each other

    Returns:
        np.array: Nch x Nrounds array of transformation matrices (each 3x3)

    """
    nrounds = len(angles_within_channels[0])
    nchannels = len(angles_within_channels)
    tforms = np.empty((nchannels, nrounds), dtype=object)

    for ich in range(nchannels):
        if align_channels and ich != ref_ch:
            use_ch = ich
        else:
            use_ch = ref_ch
        for iround in range(nrounds):
            tforms[ich, iround] = make_transform(
                scales_between_channels[use_ch],
                angles_between_channels[use_ch],
                shifts_between_channels[use_ch],
                stack_shape,
            ) @ make_transform(
                1.0,
                angles_within_channels[use_ch][iround],
                shifts_within_channels[use_ch][iround],
                stack_shape,
            )
    return tforms


def align_channels_and_rounds(stack, tforms):
    """
    Apply the provided transformations to align images across channels and rounds.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        tforms (np.array): Nch x Nrounds array of transformation matrices (each 3x3)

    Returns:
        np.array: Aligned stack with NaN for missing pixels. Same shape as input stack

    """
    nchannels, nrounds = stack.shape[2:]
    reg_stack = np.empty((stack.shape))

    for ich in range(nchannels):
        for iround in range(nrounds):
            tform = SimilarityTransform(matrix=tforms[ich, iround])
            reg_stack[:, :, ich, iround] = warp(
                stack[:, :, ich, iround],
                tform.inverse,
                preserve_range=True,
                cval=np.nan,
            )
    return reg_stack


def align_within_channels(
    stack,
    upsample=False,
    ref_round=0,
    angle_range=1.0,
    niter=3,
    nangles=15,
    min_shift=2,
    max_shift=None,
):
    """Align images within each channel.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        upsample (bool, or int): whether to use subpixel registration, and if so, how much
            to upsample
        ref_round (int): round to align to
        angle_range (float): range of angles to search for each round
        niter (int): number of iterations to run
        nangles (int): number of angles to search for each iteration
        min_shift (int): minimum shift. Necessary to avoid spurious cross-correlations
            for images acquired from the same camera

    Returns:
        angles (np.array): Nchannels x Nrounds array of angles
        shifts (np.array): Nchannels x Nrounds x 2 array of shifts

    """
    # align rounds to each other for each channel
    nchannels, nrounds = stack.shape[2:]
    angles_channels = []
    shifts_channels = []
    for ref_ch in range(nchannels):
        print(f"optimizing angles and shifts for channel {ref_ch}", flush=True)
        angles = []
        shifts = []
        for iround in range(nrounds):
            if ref_round != iround:
                angle, shift = estimate_rotation_translation(
                    stack[:, :, ref_ch, ref_round],
                    stack[:, :, ref_ch, iround],
                    angle_range=angle_range,
                    niter=niter,
                    nangles=nangles,
                    min_shift=min_shift,
                    upsample=upsample,
                    max_shift=max_shift,
                )
            else:
                angle, shift = 0.0, [0.0, 0.0]
            angles.append(angle)
            shifts.append(shift)
            print(f"angle: {angle}, shift: {shift}", flush=True)
        angles_channels.append(angles)
        shifts_channels.append(shifts)
    return angles_channels, shifts_channels


def estimate_shifts_and_angles_for_tile(
    stack, scales, ref_ch=0, threshold_quantile=0.9, max_shift=None
):
    """Estimate shifts and angles. Registration is carried out on thresholded images
    using the provided quantile threshold.

    Args:
        stack (np.array): X x Y x Nchannels images stack
        scales (np.array): Nchannels array of scales
        ref_ch (int): reference channel
        threshold_quantile (float): quantile to use for thresholding
        max_shift (int): maximum shift to avoid spurious cross-correlations

    Returns:
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts

    """

    nch = stack.shape[2]
    angles = []
    shifts = []
    ref_thresh = np.quantile(stack[:, :, ref_ch], threshold_quantile)
    for ich in range(nch):
        if ref_ch != ich:
            thresh = np.quantile(stack[:, :, ich], threshold_quantile)
            angle, shift = estimate_rotation_translation(
                stack[:, :, ref_ch] > ref_thresh,
                transform_image(stack[:, :, ich] > thresh, scale=scales[ich]),
                angle_range=2.0,
                niter=3,
                nangles=21,
                max_shift=max_shift,
            )
        else:
            angle, shift = 0.0, [0.0, 0.0]
        angles.append(angle)
        shifts.append(shift)
        print(f"Channel {ich} angle: {angle}, shift: {shift}", flush=True)
    return angles, shifts


def estimate_shifts_for_tile(
    stack,
    angles_within_channels,
    scales_between_channels,
    angles_between_channels,
    ref_ch=0,
    ref_round=0,
):
    """Use precomputed rotations and scale factors to re-estimate shifts for every round and between channels.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        scales_between_channels (np.array): Nchannels x Nchannels array of scale factors
        angles_between_channels (np.array): Nchannels x Nchannels array of angles
        ref_ch (int): reference channel
        ref_round (int): reference round

    Returns:
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        shifts_between_channels (np.array): Nchannels x Nchannels x 2 array of shifts

    """
    nchannels, nrounds = stack.shape[2:]
    shifts_within_channels = []
    for ich in range(nchannels):
        shifts = []
        reference_fft = scipy.fft.fft2(stack[:, :, ich, ref_round])
        for iround in range(nrounds):
            if iround != ref_round:
                shift = phase_corr(
                    reference_fft,
                    transform_image(
                        stack[:, :, ich, iround],
                        angle=angles_within_channels[ich][iround],
                    ),
                    fft_ref=False,
                    min_shift=2,
                )[0]
            else:
                shift = [0.0, 0.0]
            shifts.append(shift)
        shifts_within_channels.append(shifts)
    std_stack, _ = get_channel_reference_images(
        stack, angles_within_channels, shifts_within_channels
    )
    shifts_between_channels = []
    for ich in range(nchannels):
        shifts_between_channels.append(
            phase_cross_correlation(
                std_stack[:, :, ref_ch],
                transform_image(
                    std_stack[:, :, ich],
                    scale=scales_between_channels[ich],
                    angle=angles_between_channels[ich],
                ),
                upsample_factor=5,
            )[0]
        )
    tforms = generate_channel_round_transforms(
        angles_within_channels,
        shifts_within_channels,
        scales_between_channels,
        angles_between_channels,
        shifts_between_channels,
        stack.shape[:2],
    )
    return tforms, shifts_within_channels, shifts_between_channels


def get_channel_reference_images(stack, angles_channels, shifts_channels):
    """Get reference images for each channel from STD or mean projection after registration.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        angles_channels (np.array): Nchannels x Nrounds array of angles
        shifts_channels (np.array): Nchannels x Nrounds x 2 array of shifts

    Returns:
        std_stack (np.array): X x Y x Nchannels std projections
        mean_stack (np.array): X x Y x Nchannels mean projections

    """
    nchannels, nrounds = stack.shape[2:]

    # get a good reference image for each channel
    std_stack = np.zeros((stack.shape[:3]))
    mean_stack = np.zeros((stack.shape[:3]))

    for ich in range(nchannels):
        std_stack[:, :, ich] = np.std(
            apply_corrections(
                stack[:, :, ich, :],
                np.ones((nrounds)),
                angles_channels[ich],
                shifts_channels[ich],
            ),
            axis=2,
        )
        mean_stack[:, :, ich] = np.mean(
            apply_corrections(
                stack[:, :, ich, :],
                np.ones((nrounds)),
                angles_channels[ich],
                shifts_channels[ich],
            ),
            axis=2,
        )
    return std_stack, mean_stack


def apply_corrections(im, scales, angles, shifts, cval=0.0):
    """Apply scale, rotation and shift corrections to a multichannel image.

    Args:
        im (np.array): X x Y x Nchannels image
        scales (np.array): Nchannels array of scale factors
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts
        cval (float): value to fill empty pixels

    Returns:
        im_reg (np.array): X x Y x Nchannels registered image

    """
    nchannels = im.shape[2]
    im_reg = np.zeros(im.shape)
    for channel, scale, angle, shift in zip(range(nchannels), scales, angles, shifts):
        im_reg[:, :, channel] = transform_image(
            im[:, :, channel], scale=scale, angle=angle, shift=shift, cval=cval
        )

    return im_reg


def estimate_correction(
    im,
    ch_to_align=0,
    upsample=False,
    scale_range=0.05,
    nangles=3,
    niter=5,
    angle_range=1.0,
    max_shift=None,
):
    """
    Estimate scale, rotation and translation corrections for each channel of a multichannel image.

    Args:
        im (np.array): X x Y x Nchannels image
        ch_to_align (int): channel to align to
        upsample (bool, or int): whether to upsample the image, and if so but what factor
        scale_range (float): range of scale factors to search through
        nangles (int): number of angles to search through
        niter (int): number of iterations to run
        angle_range (float): range of angles to search through
        max_shift (int): maximum shift to avoid spurious cross-correlations

    Returns:
        scales (np.array): Nchannels array of scale factors
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts

    """
    nchannels = im.shape[2]
    scales, angles, shifts = [], [], []
    for channel in range(nchannels):
        print(f"optimizing rotation and scale for channel {channel}", flush=True)
        if channel != ch_to_align:
            scale, angle, shift = estimate_scale_rotation_translation(
                im[:, :, ch_to_align],
                im[:, :, channel],
                niter=niter,
                nangles=nangles,
                verbose=True,
                scale_range=scale_range,
                angle_range=angle_range,
                upsample=upsample,
                max_shift=max_shift
            )
        else:
            scale, angle, shift = 1.0, 0.0, (0, 0)
        scales.append(scale)
        angles.append(angle)
        shifts.append(shift)
    return scales, angles, shifts


def estimate_scale_rotation_translation(
    reference,
    target,
    angle_range=5.0,
    scale_range=0.01,
    nscales=15,
    niter=3,
    nangles=15,
    verbose=False,
    upsample=False,
    max_shift=None,
):
    """
    Estimate rotation and translation that maximizes phase correlation between the target and the
    reference image. Search for the best angle is performed iteratively by gradually
    searching over small and smaller angle range.

    Args:
        reference (numpy.ndarray): X x Y reference image
        target (numpy.ndarray): X x Y target image
        angle_range (float): initial range of angles in degrees to search over
        niter (int): number of iterations to refine rotation angle
        nangles (int): number of angles to try on each iteration
        verbose (bool): whether to print progress of registration
        scale (float): how to rescale image for finding the optimal rotation
        upsample (bool, or int): whether to upsample the image, and if so but what factor
        max_shift (int): maximum shift to avoid spurious cross-correlations

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
            target_rescaled = transform_image(target, scale=scales[iscale])
            best_angles[iscale], max_cc[iscale] = estimate_rotation_angle(
                reference_fft, target_rescaled, angle_range, best_angle, nangles, max_shift=max_shift
            )
        best_scale_index = np.argmax(max_cc)
        best_scale = scales[best_scale_index]
        best_angle = best_angles[best_scale_index]
        angle_range = angle_range / 5
        scale_range = scale_range / 5
        if verbose:
            print(f"Best scale: {best_scale}. Best angle: {best_angle}", flush=True)
    if not upsample:
        shift, _ = phase_corr(
            reference_fft,
            transform_image(target, scale=best_scale, angle=best_angle),
            fft_ref=False,
            max_shift=max_shift,
        )
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, scale=best_scale, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    return best_scale, best_angle, shift


# Numba makes no difference here
def estimate_rotation_angle(
    reference_fft,
    target,
    angle_range,
    best_angle,
    nangles,
    min_shift=None,
    max_shift=None,
    debug=False,
):
    """
    Estimate rotation angle that maximizes phase correlation between the target and the
    reference image.

    Args:
        reference_fft (numpy.ndarray): X x Y reference image in Fourier domain
        target (numpy.ndarray): X x Y target image
        angle_range (float): range of angles in degrees to search over
        best_angle (float): initial angle in degrees
        nangles (int): number of angles to try
        min_shift (int): minimum shift. Necessary to avoid spurious cross-correlations
            for images acquired from the same camera
        max_shift (int): maximum shift.
        debug (bool): whether to return full cross-correlation array

    Returns:
        best_angle (float) in degrees
        max_cc (float) maximum cross correlation
        cc (np.array) cross correlation array, if debug=True

    """
    angles = np.linspace(-angle_range, angle_range, nangles) + best_angle
    max_cc = np.empty(angles.shape)
    shifts = np.empty((nangles, 2))
    if debug:
        all_cc = np.empty((nangles, *reference_fft.shape), dtype=np.float64)
    for iangle in range(nangles):
        shifts[iangle, :], cc = phase_corr(
            reference_fft,
            transform_image(target, angle=angles[iangle]),
            fft_ref=False,
            min_shift=min_shift,
            max_shift=max_shift,
        )
        max_cc[iangle] = np.max(cc)
        if debug:
            all_cc[iangle] = cc
    best_angle_index = np.argmax(max_cc)
    best_angle = angles[best_angle_index]
    if debug:
        return best_angle, max_cc[best_angle_index], all_cc
    return best_angle, max_cc[best_angle_index]


# TODO: check if this numba is useful here.
@jit(parallel=True, forceobj=True)
def estimate_rotation_translation(
    reference,
    target,
    angle_range=5.0,
    niter=3,
    nangles=20,
    min_shift=None,
    upsample=None,
    max_shift=None,
    iter_range_factor=5.0,
):
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
        iter_range_factor (float): how much to shrink the angle range for each
            iteration. Default: 5.

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts

    """
    best_angle = 0
    reference_fft = scipy.fft.fft2(reference)
    for i in range(niter):
        best_angle, _ = estimate_rotation_angle(
            reference_fft,
            target,
            angle_range,
            best_angle,
            nangles,
            min_shift=min_shift,
            max_shift=max_shift,
        )
        angle_range = angle_range / iter_range_factor
    if not upsample:
        shift, _ = phase_corr(
            reference,
            transform_image(target, angle=best_angle),
            min_shift=min_shift,
            max_shift=max_shift,
        )
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    return best_angle, shift
