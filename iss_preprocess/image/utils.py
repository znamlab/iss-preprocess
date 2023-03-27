import cv2
import numpy as np
from pathlib import Path
import iss_preprocess as iss


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

        img = iss.io.load_stack(fname)
        flipped = flip_channels(
            img,
            channels_to_flip=channels_to_flip,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
        )
        iss.io.write_stack(flipped, target)

