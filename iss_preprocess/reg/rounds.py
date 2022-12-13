import numpy as np
import scipy.fft
import scipy.ndimage
from numba import jit, prange
from skimage.transform import SimilarityTransform, warp
from skimage.registration import phase_cross_correlation
from . import phase_corr
from math import cos, sin, radians


def register_channels_and_rounds(stack, ref_ch=0, ref_round=0):
    """
    Estimate transformation matrices for alignment across channels and rounds.

    Args:
        stack:
        ref_ch (int): channel to align to
        ref_round (int): round to align to

    Returns:

    """
    # first register images across rounds within each channel
    angles_within_channels, shifts_within_channels = align_within_channels(
        stack, upsample=5, ref_round=ref_round
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
):
    nrounds = len(angles_within_channels[0])
    nchannels = len(angles_within_channels)
    tforms = np.empty((nchannels, nrounds), dtype=object)

    for ich in range(nchannels):
        for iround in range(nrounds):
            tforms[ich, iround] = make_transform(
                scales_between_channels[ich],
                angles_between_channels[ich],
                shifts_between_channels[ich],
                stack_shape,
            ) @ make_transform(
                1.0,
                angles_within_channels[ich][iround],
                shifts_within_channels[ich][iround],
                stack_shape,
            )
    return tforms


def align_channels_and_rounds(stack, tforms):
    """
    Apply the provided transformations to align images across channels and rounds.

    Args:
        stack:
        tforms:

    Returns:

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


def align_within_channels(stack, upsample=False, ref_round=0):
    # align rounds to each other for each channel
    nchannels, nrounds = stack.shape[2:]
    angles_channels = []
    shifts_channels = []
    for ref_ch in range(nchannels):
        print(f"optimizing angles and shifts for channel {ref_ch}")
        angles = []
        shifts = []
        for iround in range(nrounds):
            if ref_round != iround:
                angle, shift = estimate_rotation_translation(
                    stack[:, :, ref_ch, ref_round],
                    stack[:, :, ref_ch, iround],
                    angle_range=1.0,
                    niter=3,
                    nangles=15,
                    min_shift=2,
                    upsample=upsample,
                )
            else:
                angle, shift = 0.0, [0.0, 0.0]
            angles.append(angle)
            shifts.append(shift)
            print(f"angle: {angle}, shift: {shift}")
        angles_channels.append(angles)
        shifts_channels.append(shifts)
    return angles_channels, shifts_channels


def estimate_shifts_for_tile(
    stack,
    angles_within_channels,
    scales_between_channels,
    angles_between_channels,
    ref_ch=0,
    ref_round=0,
):
    """Use precompute rotations and scale factors to re-estimate shifts for every round and between channels.

    Args:
        stack (_type_): _description_
        angles_within_channels (_type_): _description_
        scales_between_channels (_type_): _description_
        angles_between_channels (_type_): _description_
        ref_ch (int, optional): _description_. Defaults to 0.
        ref_round (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
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


def transform_image(im, scale=1, angle=0, shift=(0, 0)):
    tform = SimilarityTransform(matrix=make_transform(scale, angle, shift, im.shape))
    return warp(im, tform.inverse, preserve_range=True)


def make_transform(s, angle, shift, shape):
    angle = -radians(angle)
    center_x = float(shape[1]) / 2 - 0.5
    center_y = float(shape[0]) / 2 - 0.5
    shift_x = shift[1]
    shift_y = shift[0]
    tform = [
        [
            cos(angle) * s,
            -sin(angle) * s,
            shift_x + (center_x - s * (center_x * cos(angle) - center_y * sin(angle))),
        ],
        [
            sin(angle) * s,
            cos(angle) * s,
            shift_y + (center_y - s * (center_x * sin(angle) + center_y * cos(angle))),
        ],
        [0, 0, 1],
    ]
    return np.array(tform)


def apply_corrections(im, scales, angles, shifts):
    nchannels = im.shape[2]
    im_reg = np.zeros(im.shape)
    for channel, scale, angle, shift in zip(range(nchannels), scales, angles, shifts):
        im_reg[:, :, channel] = transform_image(
            im[:, :, channel], scale=scale, angle=angle, shift=shift
        )

    return im_reg


def estimate_correction(im, ch_to_align=0, upsample=False):
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
        print(f"optimizing rotation and scale for channel {channel}")
        if channel != ch_to_align:
            scale, angle, shift = estimate_scale_rotation_translation(
                im[:, :, ch_to_align],
                im[:, :, channel],
                niter=5,
                nangles=3,
                verbose=True,
                scale_range=scale_range,
                angle_range=1.0,
                upsample=upsample,
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
                reference_fft,
                target_rescaled,
                angle_range,
                best_angle,
                nangles,
                min_shift=None,
            )
        best_scale_index = np.argmax(max_cc)
        best_scale = scales[best_scale_index]
        best_angle = best_angles[best_scale_index]
        angle_range = angle_range / 5
        scale_range = scale_range / 5
        if verbose:
            print(f"Best scale: {best_scale}. Best angle: {best_angle}")
    if not upsample:
        shift, _ = phase_corr(
            reference_fft,
            transform_image(target, scale=best_scale, angle=best_angle),
            fft_ref=False,
        )
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, scale=best_scale, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    return best_scale, best_angle, shift


@jit(parallel=True, forceobj=True)
def estimate_rotation_angle(
    reference_fft, target, angle_range, best_angle, nangles, min_shift=None
):
    angles = np.linspace(-angle_range, angle_range, nangles) + best_angle
    max_cc = np.empty(angles.shape)
    shifts = np.empty((nangles, 2))
    for iangle in prange(nangles):
        shifts[iangle, :], cc = phase_corr(
            reference_fft,
            transform_image(target, angle=angles[iangle]),
            fft_ref=False,
            min_shift=min_shift,
        )
        max_cc[iangle] = np.max(cc)
    best_angle_index = np.argmax(max_cc)
    best_angle = angles[best_angle_index]
    return best_angle, max_cc[best_angle_index]


@jit(parallel=True, forceobj=True)
def estimate_rotation_translation(
    reference,
    target,
    angle_range=5.0,
    niter=3,
    nangles=20,
    min_shift=None,
    upsample=None,
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

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts

    """
    best_angle = 0
    reference_fft = scipy.fft.fft2(reference)
    for i in range(niter):
        best_angle, max_cc = estimate_rotation_angle(
            reference_fft,
            target,
            angle_range,
            best_angle,
            nangles,
            min_shift=min_shift,
        )
        angle_range = angle_range / 5
    if not upsample:
        shift, _ = phase_corr(reference, transform_image(target, angle=best_angle))
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, angle=best_angle),
            upsample_factor=upsample,
        )[0]
    return best_angle, shift
