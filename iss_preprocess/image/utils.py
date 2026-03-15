from pathlib import Path

import cv2
import numpy as np
import tifffile

from iss_preprocess.io.load import get_channel_round_transforms, get_processed_path, get_processed_path, load_ops, load_tile_by_coors

from ..io import load_stack, write_stack


def flip_channels(im, channels_to_flip, flip_vertical=False, flip_horizontal=False):
    """Flip some channels from an image horizontally, vertically or both

    Used to correct micromanager issues

    Args:
        im (numpy.array): Nx x Ny x Nc x ... array
        channels_to_flip (list): list of channels to flip.
        flip_vertical (bool, optional): Flip vertically. Defaults to False.
        flip_horizontal (bool, optional): Flip horizontally. Defaults to False.

    Returns:
        numpy.array: Same shape as input, with some channels flipped

    """
    if not flip_horizontal and not flip_vertical:
        print("Nothing to do")
        return
    channels_to_flip = list(channels_to_flip)
    assert all([im.shape[2] > c for c in channels_to_flip])

    if flip_horizontal and flip_vertical:
        flip_code = -1
    elif flip_horizontal:
        flip_code = 1
    else:
        flip_code = 0
    out = np.array(im, copy=True)
    for ic in channels_to_flip:
        out[:, :, ic] = cv2.flip(im[:, :, ic], flipCode=flip_code)
    return out


def flip_all_tiffs(
    folder,
    channels_to_flip,
    flip_vertical=False,
    flip_horizontal=False,
    file_filter="*.tif",
    suffix="_flipped",
    target_folder=None,
    overwrite=False,
):
    """Iterate on tiffs in a directory and flip some channels

    See flip_channels

    Args:
        folder (str): Path to folder containing tiffs to process
        channels_to_flip (list): List of channels that need flipping
        flip_vertical (bool, optional): Flip vertically. Defaults to False.
        flip_horizontal (bool, optional): Flip horizontally. Defaults to False.
        file_filter (str, optional): Filter to select tiff (will be passed to glob).
            Defaults to "*.tif".
        suffix (str, optional): Suffix to add to file name. Defaults to "_flipped".
        target_folder (str, optional): Path to folder where to save tiffs. If None,
            saves in folder Defaults to None.
        overwrite (bool, optional): Overwrite existing tifs? Defaults to False.

    """
    folder = Path(folder)
    if target_folder is None:
        target_folder = folder
    else:
        target_folder = Path(target_folder)
        assert target_folder.is_dir()

    for fname in folder.glob(file_filter):
        target = target_folder / f"{fname.stem}{suffix}{fname.suffix}"
        if target.exists() and not overwrite:
            print(f"File already exists. Skipping {target}.")

        img = load_stack(fname)
        flipped = flip_channels(
            img,
            channels_to_flip=channels_to_flip,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )
        write_stack(flipped, target)


def black_out_tiff(input_path, output_path):
    """Load a TIFF, zero all pixels, and save.

    Args:
        input_path (str): Path to input TIFF.
        output_path (str): Path to output TIFF.
    """
    if input_path is None or output_path is None:
        print("Usage: python blackout_tif.py input.tif output.tif")
        return

    # Load all pages
    with tifffile.TiffFile(input_path) as tf:
        pages = [p.asarray() for p in tf.pages]


    # Zero all pages and stack into (X, Y, Npages)
    black_pages = [np.zeros_like(img) for img in pages]
    stack = np.stack(black_pages, axis=2)  # shape: (X, Y, N)

    # Save back as multi-page TIFF (write_stack handles dtype/compression)
    write_stack(stack, output_path, dtype="uint16")

    print(f"Blacked out TIFF written to: {output_path}")

def make_butterworth_highpass_mask(H, W, cutoff, order=2):
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    eps = 1e-8
    # Butterworth high-pass: 0 at center, smoothly rises to 1
    Hmask = 1 / (1 + (cutoff / (R + eps))**(2 * order))
    return Hmask.astype(np.float32)

def highpass_fft_2d(img, cutoff=3.0, order=2, pad=None):
    img = img.astype(np.float32)

    if pad and pad > 0:
        imgp = np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")
    else:
        imgp = img

    Hp, Wp = imgp.shape
    mask = make_butterworth_highpass_mask(Hp, Wp, cutoff=cutoff, order=order)

    F = np.fft.fftshift(np.fft.fft2(imgp))
    outp = np.fft.ifft2(np.fft.ifftshift(F * mask))
    outp = np.real(outp).astype(np.float32)

    if pad and pad > 0:
        return outp[pad:-pad, pad:-pad]
    return outp

def highpass_stack(stack, cutoff=3.0, order=2, pad=None):
    H, W, C, R = stack.shape
    out = np.empty_like(stack, dtype=np.float32)
    for c in range(C):
        for r in range(R):
            out[:, :, c, r] = highpass_fft_2d(stack[:, :, c, r], cutoff=cutoff, order=order, pad=pad)
    return out


