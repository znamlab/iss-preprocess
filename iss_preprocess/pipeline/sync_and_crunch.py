import time
import warnings
from pathlib import Path

import numpy as np
import re
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from ..diagnostics import plot_zfocus
from ..diagnostics.diag_stitching import plot_overview_images
from ..io import get_processed_path, get_raw_path, get_tile_ome, load_ops
from .project import project_tile


def process_position(data_path, pos_label, prefix_folder, raw_root, ops, project=False):
    """
    Waits for the .tif to appear/unlock, then does projection or quick_std.
    Returns the pos_label if successful, else None.
    """
    # e.g. file name pattern:  <prefix_folder.name>_MMStack_<pos_label>*.tif
    raw_file_pattern = f"{prefix_folder.name}_MMStack_{pos_label}"
    
    while True:
        # Search for the file
        raw_file_candidates = list(prefix_folder.glob(f"{raw_file_pattern}*.tif"))
        if not raw_file_candidates:
            # File not present yet: sleep and try again
            time.sleep(1)
            continue
        
        # If more than one file matches, that's an error in your naming scheme.
        if len(raw_file_candidates) > 1:
            print(f"WARNING: Multiple files found for {raw_file_candidates}. Using first.")
        
        raw_file = raw_file_candidates[0]
        fname = str(raw_file.relative_to(raw_root)).replace(".ome.tif", "")
        #print(f"fname: {fname}")
        # WAIT TO MAKE SURE FILE IS FULLY WRITTEN
        time.sleep(2)
        # Attempt to process
        try:
            #TODO: Add except for: Skipping page XX due to shape mismatch: (0,)
            # which waits for file to be written and then attempts reload
            if project:
                # Extract r, x, y using regex
                match = re.match(r"Pos-(\d+)-(\d{3})_(\d{3})", pos_label)
                if match:
                    r, x, y = match.groups()
                    r = int(r)
                    x = int(x)
                    y = int(y)
                else:
                    raise ValueError(f"Invalid pos_label format: {pos_label}")

                # Construct the target_name
                target_name = f"{data_path}{prefix_folder.name}/{prefix_folder.name}_MMStack_{r}-Pos{x:03d}_{y:03d}"
                #print(f"Target name: {target_name}")
                project_tile(fname, ops, target_name=target_name, overwrite=False, verbose=False)
            else:
                quick_std(fname, overwrite=False, window_size=500)
            return pos_label
        except PermissionError:
            # If the file is still being written to, wait and retry
            print(f"File {raw_file} currently being written. Waiting 5s...")
            time.sleep(1)

def crunch_pos_file(
    data_path, 
    pos_file, 
    destination_folder=None, 
    project=False, 
    nproc=10
):
    """
    Continuously watches for new tiles that match the .pos file positions,
    and processes them as soon as they're available. Only once all positions
    are processed does it proceed to final plotting/overview steps.
    """
    pos_file = Path(pos_file)
    assert pos_file.exists(), f"{pos_file} does not exist"
    assert pos_file.suffix == ".pos", f"{pos_file} is not a .pos file"
    source_folder = pos_file.parent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ops = load_ops(data_path)

    print(f"Crunching {pos_file.name}", flush=True)

    # Example name splitting logic: CHAMBERID_PREFIX_NPOS_positions.pos
    prefix = "_".join(pos_file.name.split("_")[1:-2])

    # Continuously wait for the prefix folder(s) to appear
    prefix_folders = []
    while not prefix_folders:
        prefix_folders = list(source_folder.glob(f"{prefix}_*"))
        if not prefix_folders:
            print(f"No folder found for {prefix}", flush=True)
            # If a processed folder already exists, we can skip
            if destination_folder and (destination_folder / prefix).exists():
                print(f"{prefix} already processed, skipping", flush=True)
                return
            print("Sleeping 20s", flush=True)
            time.sleep(5)

    # We'll assume there could be multiple prefix_folders
    for prefix_folder in prefix_folders:
        # Create a target folder under destination
        target_pref_folder = (destination_folder / prefix_folder.name) if destination_folder else None
        if target_pref_folder:
            target_pref_folder.mkdir(parents=True, exist_ok=True)
            done_file = target_pref_folder / "DONE"
            if done_file.exists():
                print(f"  {prefix_folder.name} already processed, skipping", flush=True)
                
                print(f"Plotting overview images for {prefix_folder.name}", flush=True)
                # transfer metadata file from raw to processed
                raw_path = get_raw_path(data_path) / prefix_folder.name
                processed_path = get_processed_path(data_path)

                # Find all metadata files in the raw folder
                metadata_files = list(raw_path.glob("*_metadata.txt"))
                print(raw_path)
                print(metadata_files)
                print(prefix_folder.name)
                if metadata_files:
                    # Select the first metadata file
                    metadata_file = metadata_files[0]
                    print(metadata_file)
                    if not (processed_path / prefix_folder.name / metadata_file.name).exists():
                        metadata_file.rename(processed_path / prefix_folder.name / metadata_file.name)
                else:
                    print(f"Metadata file not found for {prefix_folder.name}")
                plot_overview_images(
                    data_path,
                    prefix_folder.name,
                    plot_grid=True,
                    downsample_factor=25,
                    save_raw=False,
                    dependency=None,
                    use_slurm=False,
                )

                continue
        else:
            done_file = None

        print(f"  Looking at {prefix_folder.name}", flush=True)

        # Load the positions from the .pos file (YAML)
        with open(pos_file, "r") as f:
            pos_yaml = yaml.safe_load(f)
        positions = pos_yaml["map"]["StagePositions"]["array"]
        
        # We'll collect the "to_project" set, i.e. all positions we expect
        # (but we may skip ones already processed if we find them)
        all_position_labels = []
        for pos in positions:
            label = pos["Label"]["scalar"]
            all_position_labels.append(label)
        
        # figure out which are already done
        to_project = []
        for pos_label in all_position_labels:
            if target_pref_folder:
                match = re.match(r"Pos-(\d+)-(\d{3})_(\d{3})", pos_label)
                if match:
                    r, x, y = match.groups()
                    r = int(r)
                    x = int(x)
                    y = int(y)
                else:
                    raise ValueError(f"Invalid pos_label format: {pos_label}")

                # Construct the target_name
                target_name = f"{data_path}{prefix_folder.name}_MMStack_{r}-Pos{x:03d}_{y:03d}"
                f"{prefix}_MMStack_{r}-Pos{x:03d}_{y:03d}"

                # either the max-projection or the zprofile
                if project:
                    outname = f"{target_pref_folder.name}_MMStack_{pos_label}_max.tif"
                else:
                    outname = f"{target_pref_folder.name}_MMStack_{pos_label}_zprofile.npz"
                
                outpath = target_pref_folder / outname
                if not outpath.exists():
                    to_project.append(pos_label)
            else:
                # If no destination folder is provided, we assume we do them all
                to_project.append(pos_label)
        
        if not to_project:
            print(f"All positions for {prefix_folder.name} already processed, skipping.", flush=True)

            if done_file:
                done_file.touch()
            continue
        
        print(f"    {len(to_project)}/{len(all_position_labels)} positions to project")

        # Create a progress bar
        pbar = tqdm(total=len(to_project), desc=f"Projecting {prefix_folder.name}", dynamic_ncols=True)
        processed_positions = set()
        submitted_positions = set()
        future_to_position = {}
        raw_root = get_raw_path("test").parent

        # Start a single ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            # Keep looping until we've processed all positions
            while len(processed_positions) < len(to_project):
                # 1) Submit tasks for any new positions that appear
                for pos_label in to_project:
                    if pos_label not in processed_positions and pos_label not in submitted_positions:
                        fut = executor.submit(
                            process_position,
                            data_path,
                            pos_label,
                            prefix_folder,
                            raw_root,
                            ops,
                            project
                        )
                        future_to_position[fut] = pos_label
                        submitted_positions.add(pos_label)
                
                # 2) Check any futures that are done
                done_futs = []
                for fut in future_to_position:
                    if fut.done():
                        done_futs.append(fut)
                
                for fut in done_futs:
                    pos_label = future_to_position[fut]
                    future_to_position.pop(fut)
                    try:
                        result = fut.result()
                        if result is not None:
                            processed_positions.add(result)
                            pbar.update(1)
                    except Exception as e:
                        print(f"Error processing position {pos_label}: {e}")
                
                # 3) If we haven't finished everything, sleep briefly and check again
                if len(processed_positions) < len(to_project):
                    time.sleep(1)
        pbar.close()

        print(f"Done projecting {prefix_folder.name}", flush=True)

        # Next do your plotting, etc.
        print(f"Plotting diagnostic Z-stacks for {prefix_folder.name}", flush=True)
        plot_zfocus(data_path, prefix_folder.name, rois=None, verbose=True)

        if project:
            print(f"Plotting overview images for {prefix_folder.name}", flush=True)
            # transfer metadata file from raw to processed
            raw_path = get_raw_path(data_path + "/" + prefix_folder.name)
            processed_path = get_processed_path(data_path)
            # Find all metadata files in the raw folder
            metadata_files = list(raw_path.glob("*_metadata.txt"))
            if metadata_files:
                # Select the first metadata file
                metadata_file = metadata_files[0]
                print(metadata_file)
                metadata_file.rename(processed_path / prefix_folder.name / metadata_file.name)
            else:
                print(f"Metadata file not found for {prefix_folder.name}")
            plot_overview_images(
                data_path,
                prefix_folder.name,
                plot_grid=True,
                downsample_factor=25,
                save_raw=False,
                dependency=None,
            )

        # Mark done
        if done_file:
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
