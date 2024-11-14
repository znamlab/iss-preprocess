"""WARNING THESE ARE DRAFT UTILITY FUNCTIONS AND WILL NEED TO BE ADAPTED"""

from pathlib import Path

import numpy as np
import tifffile
from cellpose import io, models, train
from znamutils import slurm_it

from ..pipeline.segment import get_stack_for_cellpose


def save_tile_for_cellpose_gui(data_path, prefix, tile_coors, channels, target_folder):
    """Save a tile for the Cellpose GUI.

    This requires to reorder the channels and save the tile.

    Args:
        data_path (Path): path to the data.
        prefix (str): prefix of the tile.
        tile_coors (tuple): coordinates of the tile.
        channels (list): list of channels.

    Returns:
        None
    """
    stack = get_stack_for_cellpose(data_path, prefix, tile_coors)
    # reorder into cellpose order and save
    part = stack.copy()
    part = np.moveaxis(part, 2, 0)
    part = np.moveaxis(part, 3, 0)
    part = np.nan_to_num(part, 0)
    part = np.clip(part, 0, 2**16 - 1).astype(np.uint16)
    tname = "_".join(map(str, tile_coors))
    fname = f"{prefix}_{tname}_all_planes_part.tif"
    print(f"Saving {fname} to {target_folder}")
    tifffile.imwrite(
        target_folder / fname, part[..., 512:1024, 512:1024], photometric="minisblack"
    )


def split_stack_label_in_2D_training(model_folder, zstack_filter="all_planes_part_seg"):
    """Split the stack and label in 2D training data.

    Cellpose cannot train in 3d, it trains plane by plane.

    Args:
        model_folder (Path): folder containing the stack and labels.
        zstack_filter (str): filter for the zstacks.

    Returns:
        list: list of mask files.
    """
    model_folder = Path(model_folder)
    assert model_folder.exists(), f"{model_folder} does not exist"
    mask_files = []
    for fname in model_folder.glob(f"*{zstack_filter}.npy"):
        print(f"Processing {fname.name}")
        seg_data = np.load(fname, allow_pickle=True).item()
        masks = seg_data["masks"]
        tiff_name = fname.with_name(fname.name.replace("_seg.npy", ".tif"))
        stack = tifffile.imread(tiff_name)
        for plane in np.arange(masks.shape[0]):
            tname = f"{fname.stem}_plane_{plane}.tif"
            tifffile.imwrite(
                model_folder / tname, stack[plane], photometric="minisblack"
            )
            tname = f"{fname.stem}_plane_{plane}_mask.tif"
            mask_files.append(model_folder / tname)
            tifffile.imwrite(
                model_folder / tname, masks[plane], photometric="minisblack"
            )
    return mask_files


@slurm_it(
    conda_env="iss-cellpos",
    slurm_options=dict(partition="gpu", gres="gpu:1", time="5:00:00", mem="16G"),
    print_job_id=True,
)
def train_model(train_dir, test_dir=None):
    """Train a Cellpose model.

    Args:
        train_dir (Path): folder containing the training data.
        test_dir (Path, optional): folder containing the test data.

    Returns:
        Path: path to the trained model.
    """
    io.logger_setup()
    output = io.load_train_test_data(
        train_dir,
        test_dir,
        image_filter="_plane_",
        mask_filter="_masks",
        look_one_level_down=False,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output

    # e.g. retrain a Cellpose model
    model = models.CellposeModel(model_type="cyto3")

    model_path = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channels=[1, 2],
        normalize=True,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=1e-4,
        SGD=True,
        learning_rate=0.1,
        n_epochs=500,
        model_name="my_new_model",
    )
    return model_path
