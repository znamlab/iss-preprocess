import time
import warnings
from pathlib import Path

import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from ..diagnostics import plot_zfocus
from ..diagnostics.diag_stitching import plot_overview_images
from ..io import get_processed_path, get_raw_path, get_tile_ome, load_ops
from .project import project_tile


def crunch_pos_file(
    data_path, pos_file, destination_folder=None, project=False, nproc=4
):
    """Crunch a single position file

    Args:
        data_path (str): Relative path to the data folder
        pos_file (str): Full Path to the position file
        destination_folder (str, optional): Path to the destination folder. If None,
            will use the processed folder. Defaults to None.
    """
    pos_file = Path(pos_file)
    assert pos_file.exists(), f"{pos_file} does not exist"
    assert pos_file.suffix == ".pos", f"{pos_file} is not a .pos file"
    source_folder = pos_file.parent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ops = load_ops(data_path)
    print(f"Crunching {pos_file.name}", flush=True)
    # Position files are name like `CHAMBERID_PREFIX_NPOS_positions.pos`
    # but micromanager adds a _1 or _2 at the end of the prefix later
    prefix = "_".join(pos_file.name.split("_")[1:-2])
    # Find the prefix folder
    prefix_folders = []
    while not prefix_folders:
        prefix_folders = list(source_folder.glob(f"{prefix}_*"))
        if not prefix_folders:
            print(f"No folder found for {prefix}", flush=True)
            # If the processed folder exists, skip, otherwise wait
            if (destination_folder / prefix).exists():
                print(f"{prefix} already processed, skipping", flush=True)
                return
            print("Sleeping 20s", flush=True)
            time.sleep(20)

    for prefix_folder in prefix_folders:
        # To avoid checking multiple times the same folder, make a "done" file
        target_pref_folder = destination_folder / prefix_folder.name
        if not target_pref_folder.exists():
            target_pref_folder.mkdir(parents=True)

        done_file = target_pref_folder / "DONE"
        if done_file.exists():
            print(f"  {prefix_folder.name} already processed, skipping", flush=True)
            continue

        print(f"  Looking at {prefix_folder.name}", flush=True)
        # First build a list of position that should be acquired
        position_list = []
        to_project = []
        with open(pos_file, "r") as f:
            positions = yaml.safe_load(f)
        positions = positions["map"]["StagePositions"]["array"]
        for pos in positions:
            pos_label = pos["Label"]["scalar"]
            position_list.append(pos_label)
            if project:
                projected_file = (
                    f"{target_pref_folder.name}_MMStack_{pos_label}_max.tif"
                )
            else:
                projected_file = (
                    f"{target_pref_folder.name}_MMStack_{pos_label}_zprofile.npz"
                )
            if not (target_pref_folder / projected_file).exists():
                to_project.append(pos_label)
        if not to_project:
            print(f"{prefix_folder.name} already projected, skipping", flush=True)
            done_file.touch()

        print(
            f"    {len(to_project)}/{len(positions)} positions to project", flush=True
        )

        # Find the raw folder
        raw_root = get_raw_path("test").parent
        # Now project the missing positions
        # Initialize the progress bar
        pbar = tqdm(total=len(to_project))

        def process_position(pos):
            raw_file = f"{prefix_folder.name}_MMStack_{pos}"
            raw_file = list(prefix_folder.glob(f"{raw_file}*.tif"))
            if len(raw_file):
                assert len(raw_file) == 1, f"Multiple files found for {raw_file}"
            else:
                return None
            raw_file = raw_file[0]
            fname = str(raw_file.relative_to(raw_root)).replace(".ome.tif", "")
            if project:
                project_tile(fname, ops, overwrite=False)
            else:
                quick_std(fname, overwrite=False, window_size=500)
            return pos

        while len(to_project):
            pbar.set_description("projecting ...")
            processed = Parallel(n_jobs=nproc)(
                delayed(process_position)(pos) for pos in to_project
            )
            processed = [pos for pos in processed if pos is not None]
            for done in processed:
                to_project.remove(done)
                pbar.update(1)
            if len(to_project):
                pbar.set_description("waiting for new positions...")
            time.sleep(1)
        pbar.close()
        print(f"Done projecting {prefix_folder.name}", flush=True)

        # Plot diagnostic Z-stacks
        print(f"Plotting diagnostic Z-stacks for {prefix_folder.name}", flush=True)
        plot_zfocus(data_path, prefix_folder.name, rois=None, verbose=True)

        if project:
            # Plot overview images
            print(f"Plotting overview images for {prefix_folder.name}", flush=True)
            plot_overview_images(
                data_path,
                prefix_folder.name,
                plot_grid=True,
                downsample_factor=25,
                save_raw=False,
                dependency=None,
            )

        # Save the done file
        done_file.touch()
        print(f"Done processing {prefix_folder}", flush=True)


def quick_std(fname, overwrite=False, window_size=500):
    """Calculate the standard deviation of a small part of the image.

    To make it faster, we only take a small window around the center of the image.

    Args:
        fname (str): Name of the file to process.
        overwrite (bool, optional): If True, will overwrite the existing file. Defaults
            to False.
        window_size (int, optional): Size of the window to take around the center
            of the image. Defaults to 500.

    Returns:
        np.array: standard deviation along the Z axis.
    """
    target_name = get_processed_path(fname + "_zprofile.npz")
    if not overwrite and target_name.exists():
        return
    im = get_tile_ome(
        get_raw_path(fname + ".ome.tif"),
        get_raw_path(fname + "_metadata.txt"),  # note that this won't be used
        # if use_indexmap is True
        use_indexmap=True,
    )
    get_processed_path(fname).parent.mkdir(parents=True, exist_ok=True)

    # To check if the focus was correct, we also save a small projectiong along Z
    center = np.array(im.shape[:2]) // 2
    part = im[
        center[0] - window_size : center[0] + window_size,
        center[1] - window_size : center[1] + window_size,
    ]
    std_z = np.std(part, axis=(0, 1))
    np.savez(target_name, std=std_z)

    return np.std(part, axis=(0, 1))
