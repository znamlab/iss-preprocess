import numpy as np
import scipy.fft
import scipy.ndimage
from scipy.ndimage import median_filter
from skimage.morphology import disk
import multiprocessing
from numba import jit
from skimage.transform import SimilarityTransform, warp
from skimage.registration import phase_cross_correlation
from . import phase_corr, make_transform, transform_image


def register_channels_and_rounds(
    stack,
    ref_ch=0,
    ref_round=0,
    median_filter=None,
    min_shift=None,
    max_shift=None,
    debug=False,
):
    """
    Estimate transformation matrices for alignment across channels and rounds.

    Args:
        stack: X x Y x Nchannels x Nrounds images stack
        ref_ch (int): channel to align to
        ref_round (int): round to align to
        median_filter (int): size of median filter to apply to the stack.
        min_shift (int): minimum shift. Necessary to avoid spurious cross-correlations
            for images acquired from the same camera
        max_shift (int): maximum shift. Necessary to avoid spurious cross-correlations
        debug (bool): whether to return debug info, default: False

    Returns:
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        scales_between_channels (np.array): Nchannels array of scales
        angles_between_channels (np.array): Nchannels array of angles
        shifts_between_channels (np.array): Nchannels x 2 array of shifts
    """

    # first register images across rounds within each channel
    out_within = align_within_channels(
        stack,
        upsample=False,
        ref_round=ref_round,
        median_filter_size=median_filter,
        min_shift=min_shift,
        max_shift=max_shift,
        debug=debug,
    )
    if debug:
        angles_within_channels, shifts_within_channels, db_info = out_within
        debug_info = {"align_within_channels": db_info}
    else:
        angles_within_channels, shifts_within_channels = out_within
    # use these to compute a reference image for each channel
    std_stack, mean_stack = get_channel_reference_images(
        stack, angles_within_channels, shifts_within_channels
    )
    out_across = list(
        estimate_correction(
            std_stack,
            ch_to_align=ref_ch,
            upsample=5,
            max_shift=max_shift,
            median_filter_size=median_filter,
            debug=debug,
        )
    )
    if debug:
        debug_info["estimate_correction"] = out_across.pop(-1)
    output = [angles_within_channels, shifts_within_channels] + out_across

    if debug:
        output.append(debug_info)
    return tuple(output)


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
    ref_round=0,
    angle_range=1.0,
    niter=3,
    nangles=15,
    upsample=False,
    median_filter_size=None,
    min_shift=None,
    max_shift=None,
    debug=False,
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
        upsample (bool, or int): whether to upsample the image, and if so by what factor
        median_filter_size (int): size of median filter to apply to the stack.
        min_shift (int): minimum shift. Necessary to avoid spurious cross-correlations
            for images acquired from the same camera
        max_shift (int): maximum shift. Necessary to avoid spurious cross-correlations
        debug (bool): whether to return debug info, default: False

    Returns:
        angles (np.array): Nchannels x Nrounds array of angles
        shifts (np.array): Nchannels x Nrounds x 2 array of shifts
        debug_info (dict): dictionary with debug info, only if debug=True

    """
    nchannels, nrounds = stack.shape[2:]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        stack = median_filter(stack, footprint=disk(median_filter_size), axes=(0, 1))

    # Prepare tasks for each combination of channel and round
    pool_args = [
        (
            ref_ch,
            iround,
            stack[:, :, ref_ch, ref_round],
            stack[:, :, ref_ch, iround],
            ref_round,
            angle_range,
            niter,
            nangles,
            upsample,
            min_shift,
            max_shift,
            debug,
        )
        for ref_ch in range(nchannels)
        for iround in range(nrounds)
    ]

    # TODO: Process tasks in parallel, each process uses ~3Gb RAM so limit to amount available
    with multiprocessing.Pool(15) as pool:
        results = pool.map(_process_single_rotation_translation, pool_args)

    # Organize results
    angles_channels = [[None] * nrounds for _ in range(nchannels)]
    shifts_channels = [[None] * nrounds for _ in range(nchannels)]
    if debug:
        debug_info = {}

    for res in results:
        if debug:
            ref_ch, iround, angle, shift, db_info = res
            debug_info[(ref_ch, iround)] = db_info
        else:
            ref_ch, iround, angle, shift = res
        angles_channels[ref_ch][iround] = angle
        shifts_channels[ref_ch][iround] = shift
    if debug:
        return angles_channels, shifts_channels, debug_info
    return angles_channels, shifts_channels


def _process_single_rotation_translation(args):
    (
        ref_ch,
        iround,
        reference,
        target,
        ref_round,
        angle_range,
        niter,
        nangles,
        upsample,
        min_shift,
        max_shift,
        debug,
    ) = args

    print(f"Processing channel {ref_ch}, round {iround}", flush=True)

    if ref_round != iround:
        out = estimate_rotation_translation(
            reference,
            target,
            angle_range=angle_range,
            niter=niter,
            nangles=nangles,
            upsample=upsample,
            min_shift=min_shift,
            max_shift=max_shift,
            debug=debug,
        )
        return ref_ch, iround, *out
    elif debug:
        return ref_ch, iround, 0.0, [0.0, 0.0], {}
    return ref_ch, iround, 0.0, [0.0, 0.0]


def estimate_shifts_and_angles_for_tile(
    stack, scales, ref_ch=0, binarise_quantile=0.9, max_shift=None, debug=False
):
    """Estimate shifts and angles. Registration is carried out on thresholded images
    using the provided quantile threshold.

    Args:
        stack (np.array): X x Y x Nchannels images stack
        scales (np.array): Nchannels array of scales
        ref_ch (int): reference channel
        binarise_quantile (float): quantile to use for thresholding
        max_shift (int): maximum shift to avoid spurious cross-correlations
        debug (bool): whether to return debug info, default: False

    Returns:
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts
        debug_info (dict): dictionary with debug info, only if debug=True

    """
    nch = stack.shape[2]
    angles = []
    shifts = []
    if binarise_quantile is not None:
        for ich in range(nch):
            ref_thresh = np.quantile(stack[:, :, ich], binarise_quantile)
            stack[:, :, ich] = stack[:, :, ich] > ref_thresh
    if debug:
        debug_info = {}
    for ich in range(nch):
        if ref_ch != ich:
            out = estimate_rotation_translation(
                stack[:, :, ref_ch],
                transform_image(stack[:, :, ich], scale=scales[ich]),
                angle_range=2.0,
                niter=3,
                nangles=21,
                max_shift=max_shift,
                debug=debug,
            )
            if debug:
                angle, shift, db_info = out
                debug_info[ich] = db_info
            else:
                angle, shift = out
        else:
            angle, shift = 0.0, [0.0, 0.0]
        angles.append(angle)
        shifts.append(shift)
        print(f"Channel {ich} angle: {angle}, shift: {shift}", flush=True)
    if debug:
        return angles, shifts, debug_info
    return angles, shifts


def estimate_shifts_for_tile(
    stack,
    angles_within_channels,
    scales_between_channels,
    angles_between_channels,
    ref_ch=0,
    ref_round=0,
    max_shift=None,
    min_shift=None,
    median_filter_size=None,
):
    """Use precomputed rotations and scale factors to re-estimate shifts for every round and between channels.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        scales_between_channels (np.array): Nchannels x Nchannels array of scale factors
        angles_between_channels (np.array): Nchannels x Nchannels array of angles
        ref_ch (int): reference channel
        ref_round (int): reference round
        max_shift (int): maximum shift to avoid spurious cross-correlations
        min_shift (int): minimum shift to avoid spurious cross-correlations
        median_filter_size (int): size of median filter to apply to the stack.

    Returns:
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        shifts_between_channels (np.array): Nchannels x Nchannels x 2 array of shifts

    """
    nchannels, nrounds = stack.shape[2:]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        stack = median_filter(stack, footprint=disk(median_filter_size), axes=(0, 1))

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
                    min_shift=min_shift,
                    max_shift=max_shift,
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
        # TODO this always uses upsample. Is that what we want?
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
    max_shift=None,
    median_filter_size=None,
    scale_range=0.05,
    nangles=3,
    niter=5,
    angle_range=1.0,
    debug=False,
):
    """
    Estimate scale, rotation and translation corrections for each channel of a multichannel image.

    Args:
        im (np.array): X x Y x Nchannels image
        ch_to_align (int): channel to align to
        upsample (bool, or int): whether to upsample the image, and if so by what factor
        max_shift (int): maximum shift to avoid spurious cross-correlations
        median_filter_size (int): size of median filter to apply to the stack.
        scale_range (float): range of scale factors to search through
        nangles (int): number of angles to search through
        niter (int): number of iterations to run
        angle_range (float): range of angles to search through
        debug (bool): whether to return debug info, default: False

    Returns:
        scales (np.array): Nchannels array of scale factors
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts

    """
    nchannels = im.shape[2]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        im = median_filter(im, footprint=disk(median_filter_size), axes=(0, 1))
    # Prepare tasks for each channel
    pool_args = [
        (
            channel,
            im[:, :, ch_to_align],
            im[:, :, channel],
            ch_to_align,
            upsample,
            max_shift,
            niter,
            nangles,
            scale_range,
            angle_range,
            debug,
        )
        for channel in range(nchannels)
    ]

    # Process tasks in parallel
    with multiprocessing.Pool(15) as pool:
        results = pool.map(_process_single_scale_rotation_translation, pool_args)

    # Organize results
    if debug:
        scales, angles, shifts, debug_info = zip(*results)
        return list(scales), list(angles), list(shifts), list(debug_info)
    scales, angles, shifts = zip(*results)
    return list(scales), list(angles), list(shifts)


def _process_single_scale_rotation_translation(args):
    (
        channel,
        reference,
        target,
        ch_to_align,
        upsample,
        max_shift,
        niter,
        nangles,
        scale_range,
        angle_range,
        debug,
    ) = args
    print(f"Processing channel {channel}", flush=True)

    if channel != ch_to_align:
        return estimate_scale_rotation_translation(
            reference,
            target,
            angle_range=angle_range,
            scale_range=scale_range,
            niter=niter,
            nangles=nangles,
            verbose=True,
            upsample=upsample,
            max_shift=max_shift,
            debug=debug,
        )
    elif debug:
        return 1.0, 0.0, (0, 0), {}
    return 1.0, 0.0, (0, 0)


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
    debug=False,
):
    """
    Estimate rotation and translation that maximizes phase correlation between the target and the
    reference image. Search for the best angle is performed iteratively by gradually
    searching over small and smaller angle range.

    Args:
        reference (numpy.ndarray): X x Y reference image
        target (numpy.ndarray): X x Y target image
        angle_range (float): initial range of angles in degrees to search over
        scale_range (float): initial range of scales to search over
        nscales (int): number of scales to try
        niter (int): number of iterations to refine rotation angle
        nangles (int): number of angles to try on each iteration
        verbose (bool): whether to print progress of registration
        upsample (bool, or int): whether to upsample the image, and if so but what factor
        max_shift (int): maximum shift to avoid spurious cross-correlations
        debug (bool): whether to return debug info. Default: False

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts
        debug_info (dict): dictionary with debug info, only if debug=True

    """
    best_angle = 0
    best_scale = 1
    reference_fft = scipy.fft.fft2(reference)
    if debug:
        debug_info = {}
    for i in range(niter):
        scales = np.linspace(-scale_range, scale_range, nscales) + best_scale
        max_cc = np.empty(scales.shape)
        best_angles = np.empty(scales.shape)
        for iscale in range(nscales):
            target_rescaled = transform_image(target, scale=scales[iscale])
            out = estimate_rotation_angle(
                reference_fft,
                target_rescaled,
                angle_range,
                best_angle,
                nangles,
                max_shift=max_shift,
                debug=debug,
            )
            if debug:
                best_angles[iscale], max_cc[iscale], db_info = out
                debug_info[f"estimate_angle_iter{i}_scale_iter{iscale}"] = db_info
            else:
                best_angles[iscale], max_cc[iscale] = out
        best_scale_index = np.argmax(max_cc)
        best_scale = scales[best_scale_index]
        best_angle = best_angles[best_scale_index]
        angle_range = angle_range / 5
        scale_range = scale_range / 5
        if verbose:
            print(
                f"Best scale: {best_scale}. Best angle: {best_angle}",
                flush=True,
            )
    if not upsample:
        shift, cc = phase_corr(
            reference_fft,
            transform_image(target, scale=best_scale, angle=best_angle),
            fft_ref=False,
            max_shift=max_shift,
        )
        if debug:
            debug_info["phase_corr"] = cc
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, scale=best_scale, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    if debug:
        return best_scale, best_angle, shift, debug_info
    return best_scale, best_angle, shift


# Numba makes no difference here
def estimate_rotation_angle(
    reference_fft,
    target,
    angle_range,
    best_angle,
    nangles,
    max_shift=None,
    min_shift=None,
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
        max_shift (int): maximum shift to avoid spurious cross-correlations
        min_shift (int): minimum shift to avoid spurious cross-correlations
        debug (bool): whether to return debug info

    Returns:
        best_angle (float) in degrees
        max_cc (float) maximum cross correlation
        debug_info (dict): dictionary with debug info, only if debug=True
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
        return best_angle, max_cc[best_angle_index], dict(xcorr=all_cc, angles=angles)
    return best_angle, max_cc[best_angle_index]


# TODO: check if this numba is useful here.
@jit(parallel=True, forceobj=True)
def estimate_rotation_translation(
    reference,
    target,
    angle_range=5.0,
    niter=3,
    nangles=20,
    upsample=None,
    min_shift=None,
    max_shift=None,
    iter_range_factor=5.0,
    debug=False,
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
        upsample (bool, or int): whether to upsample the image, and if so by what factor
        min_shift (int): minimum shift. Necessary to avoid spurious cross-correlations
            for images acquired from the same camera
        max_shift (int): maximum shift. Necessary to avoid spurious cross-correlations
        iter_range_factor (float): how much to shrink the angle range for each
            iteration. Default: 5.
        debug (bool): whether to return debug info

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts

    """

    best_angle = 0
    reference_fft = scipy.fft.fft2(reference)
    if debug:
        debug_info = {}
    for i in range(niter):
        out = estimate_rotation_angle(
            reference_fft,
            target,
            angle_range,
            best_angle,
            nangles,
            min_shift=min_shift,
            max_shift=max_shift,
            debug=debug,
        )
        if debug:
            best_angle, max_cc, db_info = out
            debug_info[f"estimate_angle_iter{i}"] = db_info
        else:
            best_angle, max_cc = out
        angle_range = angle_range / iter_range_factor
    if not upsample:
        shift, cc_phase_corr = phase_corr(
            reference,
            transform_image(target, angle=best_angle),
            min_shift=min_shift,
            max_shift=max_shift,
        )
        if debug:
            debug_info["phase_corr"] = cc_phase_corr
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    if debug:
        return best_angle, shift, debug_info
    return best_angle, shift
