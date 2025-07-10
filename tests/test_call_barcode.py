import os

import pytest

from .pytest_fixture import RAW_DATA_DIR


def test_estimate_channel_correction(barcode_data):
    from iss_preprocess.pipeline.sequencing import estimate_channel_correction

    root = barcode_data.parent.parent
    data_path = barcode_data.relative_to(root)

    # with default ops it crashes cause no ref tile set
    with pytest.raises(AssertionError):
        estimate_channel_correction(data_path, prefix="barcode_round")

    # Write an ops.yml in the barcode_data folder
    ops_data = dict(barcode_ref_tiles=(1, 2, 6))
    with open(barcode_data / "ops.yml", "w") as f:
        for k, v in ops_data.items():
            f.write(f"{k}: {v}\n")
    from iss_preprocess.io import load_ops

    _ = load_ops(data_path)


def test_call(barcode_data):
    # Check if the output directory contains the expected number of tiles
    from iss_preprocess.pipeline.pipeline import call_spots

    root = barcode_data.parent.parent
    data_path = barcode_data.relative_to(root)
    call_spots(data_path, use_slurm=False, genes=False, hybridisation=False)
    tiles = [f for f in os.listdir(barcode_data) if f.endswith(".tif")]
    assert len(tiles) == 9 * len(
        [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".tif")]
    )
