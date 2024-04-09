import gc
import multiprocessing

import numpy as np
import scipy.fft
import scipy.ndimage
from image_tools.registration import affine_by_block as abb
from image_tools.registration import phase_correlation as mpc
from image_tools.similarity_transforms import make_transform, transform_image
from scipy.ndimage import median_filter
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp


def register_channels_and_rounds(
    stack,
    ref_ch=0,
    ref_round=0,
    median_filter=None,
    min_shift=None,
    max_shift=None,
    debug=False,
    use_masked_correlation=False,
    affine_by_block=True,
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
        use_masked_correlation (bool): whether to use masked phase correlation
        affine_by_block (bool): whether to use affine by block registration instead of
            similarity transforms for channel alignment. Default: True

    Returns:
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        accross_channels_params (list): Nchannels list of affine transformations if
            affine_by_block, a list of scales_between_channels (Nchannels array of
            scales), angles_between_channels (Nchannels array of angles), and
            shifts_between_channels (Nchannels x 2 array of shifts)
        debug_info (dict): dictionary with debug info, only if debug=True
    """

    # first register images across rounds within each channel
    out_within = align_within_channels(
        stack,
        upsample=False,
        ref_round=ref_round,
        median_filter_size=median_filter,
        min_shift=min_shift,
        max_shift=max_shift,
        use_masked_correlation=use_masked_correlation,
        debug=debug,
    )
    if debug:
        angles_within_channels, shifts_within_channels, db_info = out_within
        debug_info = {"align_within_channels": db_info}
    else:
        angles_within_channels, shifts_within_channels = out_within
    gc.collect()
    # use these to compute a reference image for each channel
    std_stack, mean_stack = get_channel_reference_images(
        stack, angles_within_channels, shifts_within_channels
    )
    if affine_by_block:
        out_across = list(
            correct_by_block(
                std_stack,
                ch_to_align=ref_ch,
                median_filter_size=median_filter,
                debug=debug,
            )
        )
        if debug:
            debug_info["correct_by_block"] = out_across.pop(-1)
    else:
        out_across = list(
            estimate_correction(
                std_stack,
                ch_to_align=ref_ch,
                upsample=5,
                max_shift=max_shift,
                median_filter_size=median_filter,
                use_masked_correlation=use_masked_correlation,
                debug=debug,
            )
        )
        if debug:
            debug_info["estimate_correction"] = out_across.pop(-1)
    output = [angles_within_channels, shifts_within_channels] + [out_across]

    if debug:
        output.append(debug_info)
    return tuple(output)


def generate_channel_round_transforms(
    angles_within_channels,
    shifts_within_channels,
    matrix_between_channels,
    stack_shape,
    align_channels=True,
    ref_ch=0,
):
    """Generate transformation matrices for each channel and round.

    Args:
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        shifts_within_channels (np.array): Nchannels x Nrounds x 2 array of shifts
        matrix_between_channels (list): Nchannels list of affine transformations matrices
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
            tforms[ich, iround] = matrix_between_channels[use_ch] @ make_transform(
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
            tform = AffineTransform(matrix=tforms[ich, iround])
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
    use_masked_correlation=False,
    debug=False,
    multiprocess=True,
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
        use_masked_correlation (bool): whether to use masked phase correlation
        debug (bool): whether to return debug info, default: False
        multiprocess (bool): whether to use multiprocessing, default: True

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
            use_masked_correlation,
            debug,
        )
        for ref_ch in range(nchannels)
        for iround in range(nrounds)
    ]

    if multiprocess:
        # TODO: Process tasks in parallel, each process uses ~3Gb RAM so limit to amount
        with multiprocessing.Pool(15) as pool:
            results = pool.map(_process_single_rotation_translation, pool_args)
    else:
        results = [_process_single_rotation_translation(args) for args in pool_args]

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
        use_masked_correlation,
        debug,
    ) = args

    print(f"Processing channel {ref_ch}, round {iround}", flush=True)

    if ref_round != iround:
        if use_masked_correlation:
            reference_mask = reference.astype(np.float32)
            reference_mask = np.where(reference_mask <= 0, 0, 1)
            target_mask = target.astype(np.float32)
            target_mask = np.where(target_mask <= 0, 0, 1)
        else:
            reference_mask = None
            target_mask = None
        out = estimate_rotation_translation(
            reference,
            target,
            angle_range=angle_range,
            niter=niter,
            nangles=nangles,
            upsample=upsample,
            min_shift=min_shift,
            max_shift=max_shift,
            reference_mask=reference_mask,
            target_mask=target_mask,
            debug=debug,
        )
        return ref_ch, iround, *out
    elif debug:
        return ref_ch, iround, 0.0, [0.0, 0.0], {}
    return ref_ch, iround, 0.0, [0.0, 0.0]


def estimate_affine_for_tile(
    stack,
    tform_matrix,
    max_shift=None,
    ref_ch=0,
    debug=False,
):
    """Estimate affine transformations for a single tile

    Args:
        stack (np.array): X x Y x Nchannels images stack
        tform_matrix (np.array): Nchannels list of affine transformations matrices
        max_shift (int): maximum shift to avoid spurious cross-correlations
        ref_ch (int): reference channel
        debug (bool): whether to return debug info, default: False

    Returns:
        matrix (np.array): Nchannels list of affine transformations matrices
        debug_info (dict): dictionary with debug info, only if debug=True
    """
    # run affine by block on image transformed by the matrix
    moving_image = apply_corrections(stack, matrix=tform_matrix)
    # We do the median filtering in the parent function to do it before corrections
    out = correct_by_block(
        moving_image,
        ch_to_align=ref_ch,
        median_filter_size=None,
        block_size=512,
        overlap=0.6,
        correlation_threshold=0.01,
        debug=debug,
    )
    if debug:
        matrix, debug_info = out
    else:
        matrix = out
    # multiply the matrix by the tform_matrix to get the whole transform
    nch = stack.shape[2]
    for ich in range(nch):
        matrix[ich] = matrix[ich] @ tform_matrix[ich]
    if debug:
        return matrix, debug_info
    return matrix


def estimate_shifts_and_angles_for_tile(
    stack,
    scales=None,
    ref_ch=0,
    max_shift=None,
    debug=False,
):
    """Estimate shifts and angles for a single tile

    Args:
        stack (np.array): X x Y x Nchannels images stack
        scales (np.array): Nchannels array of scales
        ref_ch (int): reference channel
        max_shift (int): maximum shift to avoid spurious cross-correlations
        debug (bool): whether to return debug info, default: False

    Returns:
        angles (np.array): Nchannels array of angles, if tfom_matrix is None
        shifts (np.array): Nchannels x 2 array of shifts, if tfom_matrix is None
        debug_info (dict): dictionary with debug info, only if debug=True

    """
    nch = stack.shape[2]
    if debug:
        debug_info = {}
    angles = []
    shifts = []
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
    matrix_between_channels,
    ref_ch=0,
    ref_round=0,
    max_shift=None,
    min_shift=None,
    median_filter_size=None,
):
    """Use precomputed rotations and scale factors to re-estimate shifts for every round
    and between channels.

    Args:
        stack (np.array): X x Y x Nchannels x Nrounds images stack
        angles_within_channels (np.array): Nchannels x Nrounds array of angles
        matrix_between_channels (list): Nchannels list of affine transformations matrices
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
                shift = mpc.phase_correlation(
                    reference_fft,
                    transform_image(
                        stack[:, :, ich, iround],
                        angle=angles_within_channels[ich][iround],
                    ),
                    fixed_image_is_fft=True,
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
    matrix_between_channels_new = matrix_between_channels.copy()
    for ich in range(nchannels):
        # TODO this always uses upsample. Is that what we want?
        moving_image = warp(
            std_stack[:, :, ich],
            AffineTransform(matrix=matrix_between_channels[ich]).inverse,
            preserve_range=True,
            cval=0,
        )
        extra_shifts_between_channels = phase_cross_correlation(
            std_stack[:, :, ref_ch],
            moving_image,
            upsample_factor=5,
        )[0]
        # add extra_shift to the matrix, matrix is x/y, shifts are row/column
        matrix_between_channels_new[ich][:2, 2] += extra_shifts_between_channels[::-1]

    tforms = generate_channel_round_transforms(
        angles_within_channels,
        shifts_within_channels,
        matrix_between_channels_new,
        stack.shape[:2],
    )
    return tforms, shifts_within_channels, matrix_between_channels_new


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
        corrected = apply_corrections(
            stack[:, :, ich, :],
            scales=np.ones((nrounds)),
            angles=angles_channels[ich],
            shifts=shifts_channels[ich],
        )
        std_stack[:, :, ich] = np.std(corrected, axis=2)
        mean_stack[:, :, ich] = np.mean(corrected, axis=2)
    return std_stack, mean_stack


def apply_corrections(im, matrix=None, scales=None, angles=None, shifts=None, cval=0.0):
    """Apply scale, rotation and shift corrections to a multichannel image.

    Args:
        im (np.array): X x Y x Nchannels image
        matrix (np.array): Nchannels list of affine transformations matrices. If
            provided, scales, angles and shifts are ignored.
        scales (np.array): Nchannels array of scale factors
        angles (np.array): Nchannels array of angles
        shifts (np.array): Nchannels x 2 array of shifts
        cval (float): value to fill empty pixels

    Returns:
        im_reg (np.array): X x Y x Nchannels registered image

    """
    nchannels = im.shape[2]
    im_reg = np.zeros(im.shape)
    if matrix is None:
        for ch, scale, angle, shift in zip(range(nchannels), scales, angles, shifts):
            im_reg[:, :, ch] = transform_image(
                im[:, :, ch], scale=scale, angle=angle, shift=shift, cval=cval
            )
    else:
        for ch, mat in enumerate(matrix):
            im_reg[:, :, ch] = warp(
                im[:, :, ch],
                AffineTransform(matrix=mat).inverse,
                preserve_range=True,
                cval=cval,
            )

    return im_reg


def correct_by_block(
    im,
    ch_to_align,
    median_filter_size=None,
    block_size=256,  # todo make ops
    overlap=0.5,
    max_shift=None,
    correlation_threshold=None,
    debug=False,
):
    """Estimate affine transformations by block for each channel of a multichannel image.

    Args:
        im (np.array): X x Y x Nchannels image
        ch_to_align (int): channel to align to
        median_filter_size (int): size of median filter to apply to the stack.
        block_size (int): size of the block to use for registration. Default: 256
        overlap (float): overlap between blocks. Default: 0.5
        max_shift (int): maximum shift to avoid spurious cross-correlations
        correlation_threshold (float): threshold for correlation to use for fitting
            affine transformations. None to keep all values. Default: None
        debug (bool): whether to return debug info, default: False

    Returns:
        output (list): Nchannels list of affine transformations

    """

    nchannels = im.shape[2]
    if median_filter_size is not None:
        print(f"Filtering with median filter of size {median_filter_size}")
        assert isinstance(
            median_filter_size, int
        ), "reg_median_filter must be an integer"
        im = median_filter(im, footprint=disk(median_filter_size), axes=(0, 1))
    reference = im[:, :, ch_to_align]
    matrix_list = []
    if debug:
        db = {}
    for channel in range(nchannels):
        if channel != ch_to_align:
            target = im[:, :, channel]
            params = abb.find_affine_by_block(
                reference,
                target,
                block_size=block_size,
                overlap=overlap,
                max_shift=max_shift,
                correlation_threshold=correlation_threshold,
                debug=debug,
            )
            if debug:
                params, db[channel] = params
        else:
            params = np.array([1, 0, 0, 0, 1, 0])
        # make a 3x3 matrix from the 6 parameters
        matrix = np.zeros((3, 3))
        matrix[0] = params[:3]
        matrix[1] = params[3:]
        matrix[2] = [0, 0, 1]
        matrix_list.append(matrix)

    if debug:
        return matrix_list, db
    return matrix_list


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
    use_masked_correlation=False,
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
        use_masked_correlation (bool): whether to use masked phase correlation

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
            use_masked_correlation,
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
    else:
        scales, angles, shifts = zip(*results)
    # make a matrix from the shifts, scales and angles
    matrix_list = [
        make_transform(scale, angle, shift, im.shape[:2])
        for scale, angle, shift in zip(scales, angles, shifts)
    ]

    if debug:
        return matrix_list, debug_info
    return matrix_list


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
        use_masked_correlation,
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
            use_masked_correlation=use_masked_correlation,
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
    use_masked_correlation=False,
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
        use_masked_correlation (bool): whether to use masked phase correlation

    Returns:
        best_angle (float) in degrees
        shift (tuple) of X and Y shifts
        debug_info (dict): dictionary with debug info, only if debug=True

    """
    best_angle = 0
    best_scale = 1

    if debug:
        debug_info = {}
    if use_masked_correlation:
        target_mask = target.astype(np.float32)
        target_mask = np.where(target_mask <= 0, 0, 1)
        (
            reference_fft,
            reference_squared_fft,
            reference_mask_fft,
        ) = mpc.get_mask_and_ffts(reference)
    else:
        reference_fft = scipy.fft.fft2(reference)
        reference_mask_fft = None
        reference_squared_fft = None

    for i in range(niter):
        scales = np.linspace(-scale_range, scale_range, nscales) + best_scale
        max_cc = np.empty(scales.shape)
        best_angles = np.empty(scales.shape)
        for iscale in range(nscales):
            target_rescaled = transform_image(target, scale=scales[iscale])
            if use_masked_correlation:
                target_mask_rescaled = transform_image(
                    target_mask, scale=scales[iscale]
                )
            else:
                target_mask_rescaled = None
            out = estimate_rotation_angle(
                reference_fft,
                target_rescaled,
                angle_range,
                best_angle,
                nangles,
                max_shift=max_shift,
                debug=debug,
                target_mask=target_mask_rescaled,
                reference_mask_fft=reference_mask_fft,
                reference_squared_fft=reference_squared_fft,
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
        shift, _, cc, _ = mpc.phase_correlation(
            reference_fft,
            transform_image(target, scale=best_scale, angle=best_angle),
            fixed_image_is_fft=True,
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
    reference_mask_fft=None,
    target_mask=None,
    reference_squared_fft=None,
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
        reference_mask_fft (numpy.ndarray): Binary mask for reference image. If not None,
            will compute masked phase correlation. Default: None
        target_mask (numpy.ndarray): Binary mask for target image. If not None,
            will compute masked phase correlation. Default: None
        reference_squared_fft (numpy.ndarray): FFT of the squared reference image. Required
            for masked phase correlation. Default: None


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
        if reference_mask_fft is None:
            move_mask = None
        else:
            move_mask = transform_image(target_mask, angle=angles[iangle])
        shifts[iangle, :], _, cc, _ = mpc.phase_correlation(
            reference_fft,
            transform_image(target, angle=angles[iangle]),
            fixed_mask=reference_mask_fft,
            moving_mask=move_mask,
            min_shift=min_shift,
            max_shift=max_shift,
            fixed_image_is_fft=True,
            fixed_mask_is_fft=True,
            fixed_squared_fft=reference_squared_fft,
        )
        max_cc[iangle] = np.max(cc)
        if debug:
            all_cc[iangle] = cc
    best_angle_index = np.argmax(max_cc)
    best_angle = angles[best_angle_index]
    if debug:
        # to save memory, cut xcorr with max_shift
        _, hrow, hcol = np.array(all_cc.shape) // 2
        all_cc = all_cc[
            :, hrow - max_shift : hrow + max_shift, hcol - max_shift : hcol + max_shift
        ].copy()
        return best_angle, max_cc[best_angle_index], dict(xcorr=all_cc, angles=angles)
    return best_angle, max_cc[best_angle_index]


def estimate_rotation_translation(
    reference,
    target,
    angle_range=5.0,
    niter=3,
    nangles=20,
    reference_mask=None,
    target_mask=None,
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
        reference_mask (numpy.ndarray): Binary mask for reference image. Only used
            if upsample is not None. Default: None
        target_mask (numpy.ndarray): Binary mask for target image. Only used if upsample
            is not None. Default: None
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
    if reference_mask is not None:
        reference_fft, fixed_squared_fft, reference_mask_fft = mpc.get_mask_and_ffts(
            reference, reference_mask
        )
    else:
        reference_fft = scipy.fft.fft2(reference)
        fixed_squared_fft = None
        reference_mask_fft = None
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
            reference_mask_fft=reference_mask_fft,
            reference_squared_fft=fixed_squared_fft,
            target_mask=target_mask,
        )
        if debug:
            best_angle, max_cc, db_info = out
            debug_info[f"estimate_angle_iter{i}"] = db_info
        else:
            best_angle, max_cc = out
        angle_range = angle_range / iter_range_factor
    if not upsample:
        if reference_mask is not None:
            target_mask = transform_image(target_mask, angle=best_angle)

        shift, _, cc_phase_corr, _ = mpc.phase_correlation(
            reference_fft,
            transform_image(target, angle=best_angle),
            min_shift=min_shift,
            max_shift=max_shift,
            fixed_mask=reference_mask_fft,
            moving_mask=target_mask,
            fixed_image_is_fft=True,
            fixed_mask_is_fft=True,
            fixed_squared_fft=fixed_squared_fft,
        )
        if debug:
            hrow, hcol = np.array(cc_phase_corr.shape) // 2
            debug_info["phase_corr"] = cc_phase_corr[
                hrow - max_shift : hrow + max_shift, hcol - max_shift : hcol + max_shift
            ].copy()
    else:
        shift = phase_cross_correlation(
            reference,
            transform_image(target, angle=best_angle),
            upsample_factor=upsample,
            reference_mask=reference_mask,
            target_mask=target_mask,
        )[0]
    if debug:
        return best_angle, shift, debug_info
    return best_angle, shift


def phase_correlation_by_block(
    reference, target, block_size=256, overlap=0.1, upsample_factor=1
):
    """Estimate translation between two images by dividing them into blocks and estimating
    translation for each block.

    Args:
        reference (np.array): reference image
        target (np.array): target image
        block_size (int): size of the blocks
        overlap (float): fraction of overlap between blocks

    Returns:
        shifts (np.array): array of shifts for each block

    """
    # Iterate on blocks
    col, row = 0, 0
    shifts = {}
    while row < reference.shape[0]:
        while col < reference.shape[1]:
            ref_block = reference[row : row + block_size, col : col + block_size]
            target_block = target[row : row + block_size, col : col + block_size]
            shift = phase_cross_correlation(
                ref_block,
                target_block,
            )[0]
            col += int((1 - overlap) * block_size)
            block_center = (row + block_size // 2, col + block_size // 2)
            shifts[block_center] = shift
        row += int((1 - overlap) * block_size)
        col = 0
    return shifts
